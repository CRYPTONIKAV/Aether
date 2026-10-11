"""Agent-loop tests — scripted FakeProvider turns, everything else faked.

The core paths: a proposal → authz ALLOW executes; a risky proposal parks
for approval and a decision runs the held call; a denied proposal never
executes; inbound chat becomes memory and reaches the model; new events
become observations; configured source polls run on schedule.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from fakes import (
    FakeAgentSettings,
    FakeApprovals,
    FakeAudit,
    FakeComposioBridge,
    FakeConfigManager,
    FakeContextBuilder,
    FakeEntities,
    FakeEventStore,
    FakeProvider,
    FakeRegistry,
    FakeRoutines,
    FakeSalience,
    FakeScheduler,
    FakeSurfaceConnector,
    FakeTraces,
    FakeTranscript,
)

from aether.agent.loop import (
    _HUB_RESET_AFTER_SECONDS,
    _MISSING,
    AgentLoop,
    CaptureRequestBox,
    SurfaceFanout,
    _coerce_config_value,
    _parse_config_command,
)
from aether.agent.prompts import SYSTEM_PROMPT
from aether.agent.tools import register_native_tools
from aether.authz.approvals import APPROVED, DENIED
from aether.authz.audit import ChainVerification
from aether.authz.policy import Policy
from aether.composio_bridge import SERVER_ENTRY
from aether.config import (
    AgentConfig,
    AppConfig,
    AuthzRule,
    MCPServerConfig,
    MessagingConfig,
    PlatformToggle,
    PollTool,
    TransportConfig,
)
from aether.connectors.base import InboundMessage
from aether.connectors.registry import ToolRegistry
from aether.llm.types import ToolCall, ToolResult, ToolSpec, Turn
from aether.memory.events import Event, IngestResult
from aether.scheduler.jobs import ScheduledAction
from aether.secret_env import EnvResolver
from aether.workspace import Workspace


def _msg(
    text: str,
    handle: str = "@vedant",
    surface: str = "telegram",
    reply_to_id: str = "",
    reply_to_text: str = "",
) -> InboundMessage:
    return InboundMessage(
        surface=surface,
        handle=handle,
        text=text,
        chat_ref="1",
        reply_to_id=reply_to_id,
        reply_to_text=reply_to_text,
    )


def _event(
    event_id: int, source: str = "mail", kind: str = "poll:unread", salience: float = 0.0
) -> Event:
    return Event(
        id=event_id,
        source=source,
        kind=kind,
        occurred_at=datetime.now(UTC),
        payload={"result": "3 unread"},
        salience_score=salience,
        memorable=False,
        meta={},
    )


class LoopKit:
    """One agent loop wired to every fake, for one test."""

    def __init__(
        self,
        provider: FakeProvider | None,
        *,
        rules: list[AuthzRule] | None = None,
        config: AppConfig | None = None,
        agent_settings: FakeAgentSettings | None = None,
        config_manager: FakeConfigManager | None = None,
        workspace: Any = None,
        transcript: Any = None,
    ):
        self.tools = ToolRegistry()
        self.events = FakeEventStore()
        self.approvals = FakeApprovals()
        self.audit = FakeAudit()
        self.salience = FakeSalience()
        self.context = FakeContextBuilder()
        self.scheduler = FakeScheduler()
        self.routines = FakeRoutines()
        self.traces = FakeTraces()
        self.connector = FakeSurfaceConnector()
        self.capture_box = CaptureRequestBox()
        self.executed: list[tuple[str, dict]] = []
        self.loop = AgentLoop(
            providers=FakeRegistry(provider) if provider else None,
            tools=self.tools,
            policy=Policy(rules or []),
            approvals=self.approvals,
            audit=self.audit,
            events=self.events,
            salience=self.salience,
            context=self.context,
            scheduler=self.scheduler,
            surfaces=SurfaceFanout([self.connector]),
            capture_box=self.capture_box,
            config=config or AppConfig(),
            host=None,
            routines=self.routines,
            traces=self.traces,
            agent_settings=agent_settings,
            config_manager=config_manager,
            workspace=workspace,
            transcript=transcript,
        )
        if config_manager is not None:
            # the real native config tools, so a /config write goes through
            # the true closure → manager path the production loop uses (and
            # the park test's set_config must be runnable, or the executor
            # would answer "unavailable" instead of parking)
            register_native_tools(
                registry=self.tools,
                events=self.events,
                entities=FakeEntities(),
                approvals=self.approvals,
                scheduler=self.scheduler,
                surfaces=SurfaceFanout([self.connector]),
                capture_box=self.capture_box,
                config_manager=config_manager,
            )

    def add_tool(self, name: str, result: str = "ok") -> None:
        executed = self.executed

        async def handler(params: dict) -> str:
            executed.append((name, dict(params)))
            return result

        self.tools.add_native(ToolSpec(name=name, description=f"test {name}"), handler)

    def add_send_chat_message(self) -> None:
        """The real native tool's mirror: fans out through the loop's own
        surfaces and refuses empty text — so tests exercise the one-delivery
        guard against the true delivery shape."""

        async def send(params: dict) -> str:
            text = str(params.get("text", "")).strip()
            if not text:
                return "send_chat_message needs text."
            await self.loop._surfaces.send_to_user(text)
            return "Sent to the user's chat surfaces."

        self.tools.add_native(ToolSpec(name="send_chat_message", description="send"), send)


# ---------------------------------------------------------------------------
# the authz-gated executor: allow / park / deny
# ---------------------------------------------------------------------------


async def test_allow_classified_calls_execute_and_are_audited() -> None:
    kit = LoopKit(FakeProvider([Turn(text="done")]))
    kit.add_tool("mail__list_messages", result="2 unread: alice, bob")

    result = await kit.loop._execute(
        ToolCall(id="t1", name="mail__list_messages", arguments={"limit": 2})
    )
    assert result.is_error is False
    assert result.content == "2 unread: alice, bob"
    assert kit.executed == [("mail__list_messages", {"limit": 2})]
    assert kit.audit.entries[-1]["decision"] == "allow"
    assert kit.audit.entries[-1]["rules_matched"] == "builtin:read-only"


async def test_a_calls_plain_line_is_recorded_and_never_reaches_the_tool() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__list_messages", result="2 unread")
    trace: list[dict] = []

    result = await kit.loop._execute(
        ToolCall(
            id="t1",
            name="mail__list_messages",
            arguments={"limit": 2, "_plain": "Checking mail for Sam's reply"},
        ),
        trace=trace,
    )
    assert result.is_error is False
    # `_plain` is stripped before the tool, the policy, and the audit see
    # the arguments — it is display text, never part of the call
    assert kit.executed == [("mail__list_messages", {"limit": 2})]
    assert kit.audit.entries[-1]["params"] == {"limit": 2}
    assert trace[0]["plain"] == "Checking mail for Sam's reply"
    assert trace[0]["params"] == {"limit": 2}


async def test_a_call_without_plain_gets_the_backend_sentence() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__list_messages", result="2 unread")
    trace: list[dict] = []

    await kit.loop._execute(
        ToolCall(id="t1", name="mail__list_messages", arguments={}), trace=trace
    )
    assert trace[0]["plain"] == "Checking mail"


async def test_a_parked_call_keeps_its_plain_line_and_clean_params() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__send_message")
    trace: list[dict] = []

    await kit.loop._execute(
        ToolCall(
            id="t1",
            name="mail__send_message",
            arguments={"to": "a@b.c", "_plain": "Emailing Sam the Thursday confirmation"},
        ),
        trace=trace,
    )
    assert kit.executed == []  # parked, not run
    assert kit.approvals.created[0].params == {"to": "a@b.c"}  # no `_plain` in the record
    assert trace[0]["plain"] == "Emailing Sam the Thursday confirmation"


async def test_risky_calls_are_parked_and_presented_not_executed() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__send_message")

    result = await kit.loop._execute(
        ToolCall(id="t1", name="mail__send_message", arguments={"to": "a@b.c", "body": "hi"})
    )
    assert kit.executed == []  # nothing ran
    assert len(kit.approvals.created) == 1
    approval = kit.approvals.created[0]
    assert approval.tool_name == "mail__send_message"
    assert approval.params == {"to": "a@b.c", "body": "hi"}
    assert kit.connector.approvals_presented[0][0] == approval.id
    assert "held for approval" in result.content
    assert result.is_error is False  # parked is a valid outcome, not an error


async def test_an_approved_decision_runs_the_held_call() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__send_message", result="sent")

    await kit.loop._execute(
        ToolCall(id="t1", name="mail__send_message", arguments={"to": "a@b.c", "body": "hi"})
    )
    approval = kit.approvals.created[0]
    approval.status = APPROVED  # the connector's decide() already flipped it

    await kit.loop.execute_decision(approval.id, APPROVED)
    assert kit.executed == [("mail__send_message", {"to": "a@b.c", "body": "hi"})]
    assert kit.approvals.executed == [approval.id]
    assert any("Sent the email" in s for s in kit.connector.sent)
    # the result, plainly — no "Done — tool executed" ceremony
    assert not any("Done —" in s for s in kit.connector.sent)


async def test_a_denied_decision_never_runs() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__send_message")

    await kit.loop._execute(
        ToolCall(id="t1", name="mail__send_message", arguments={"to": "a@b.c", "body": "hi"})
    )
    approval = kit.approvals.created[0]
    approval.status = DENIED

    await kit.loop.execute_decision(approval.id, DENIED)
    assert kit.executed == []
    # a plain okay, not a report — "Not run — you denied …" is the log line
    assert any("Okay — not sending an email" in s for s in kit.connector.sent)


async def test_a_carried_out_action_closes_the_loop_in_memory() -> None:
    # the model's only picture of past turns is the event store — the
    # request lands there (chat_message), so the fulfillment must too, or
    # every later turn still sees the ask as open
    kit = LoopKit(None)
    kit.add_tool("mail__send_message", result="sent")

    await kit.loop._execute(
        ToolCall(id="t1", name="mail__send_message", arguments={"to": "a@b.c", "body": "hi"})
    )
    approval = kit.approvals.created[0]
    approval.status = APPROVED

    await kit.loop.execute_decision(approval.id, APPROVED)
    remembered = [e for e in kit.events.ingested if e["kind"] == "action_done"]
    assert len(remembered) == 1
    assert remembered[0]["source"] == "agent"
    assert remembered[0]["payload"]["tool"] == "mail__send_message"
    assert remembered[0]["payload"]["params"] == {"to": "a@b.c", "body": "hi"}
    assert remembered[0]["payload"]["approval_id"] == approval.id
    assert "sent" in remembered[0]["payload"]["detail"]


async def test_a_denied_action_is_remembered_as_denied() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__send_message")

    await kit.loop._execute(ToolCall(id="t1", name="mail__send_message", arguments={"to": "a@b.c"}))
    approval = kit.approvals.created[0]
    approval.status = DENIED

    await kit.loop.execute_decision(approval.id, DENIED)
    remembered = [e for e in kit.events.ingested if e["kind"] == "action_denied"]
    assert len(remembered) == 1
    assert remembered[0]["payload"]["tool"] == "mail__send_message"


async def test_a_failed_carry_out_is_remembered_as_failed() -> None:
    kit = LoopKit(None)
    approval = await kit.approvals.create(tool_name="gmail__send_message", params={"to": "a@b.c"})
    approval.status = APPROVED

    await kit.loop.execute_decision(approval.id, APPROVED)
    remembered = [e for e in kit.events.ingested if e["kind"] == "action_failed"]
    assert len(remembered) == 1
    assert remembered[0]["payload"]["tool"] == "gmail__send_message"
    assert remembered[0]["payload"]["approval_id"] == approval.id


def _agent_outcome_event(event_id: int) -> Event:
    """The event _remember_outcome writes after a carried-out approval —
    what the next tick's turn is told about its own act."""
    return Event(
        id=event_id,
        source="agent",
        kind="action_done",
        occurred_at=datetime.now(UTC),
        payload={
            "tool": "mail__send_message",
            "params": {"to": "a@b.c"},
            "plain": "Sent the email (to a@b.c)",
            "detail": "sent",
            "approval_id": 1,
        },
        salience_score=5.0,
        memorable=True,
        meta={},
    )


def _held_contents(provider: FakeProvider, call_index: int) -> list[str]:
    """The tool-result texts the model saw before its Nth completion."""
    return [
        b.content
        for m in provider.calls[call_index][1]
        for b in m.blocks
        if isinstance(b, ToolResult)
    ]


async def test_an_unprompted_turn_cannot_re_park_a_just_reported_action() -> None:
    """The reported loop: an approved send runs, its outcome lands as an
    event, and the next turn — told exactly that — drafts the send again,
    so the user is asked to approve the same thing over and over. The
    executor refuses: a re-proposal of a just-settled act never parks."""
    provider = FakeProvider(
        [
            Turn(
                text="",
                tool_calls=[
                    ToolCall(id="t1", name="mail__send_message", arguments={"to": "a@b.c"})
                ],
            ),
            Turn(text=""),
        ]
    )
    kit = LoopKit(provider)
    kit.add_tool("mail__send_message")
    kit.events.events[1] = _agent_outcome_event(1)

    await kit.loop._tick()

    assert kit.approvals.created == []  # the loop ends here — no second ask
    assert kit.connector.approvals_presented == []
    # and the model heard why, in plain words, not silence
    assert any("already settled" in c for c in _held_contents(provider, 1))
    assert any("Sent the email (to a@b.c)" in c for c in _held_contents(provider, 1))


async def test_a_user_asking_again_still_parks() -> None:
    """The guard is scoped to turns nobody asked for: the user typing
    "send it again" right after the outcome event must still reach the
    one-tap approval — their message is a fresh request."""
    provider = FakeProvider(
        [
            Turn(
                text="",
                tool_calls=[
                    ToolCall(id="t1", name="mail__send_message", arguments={"to": "a@b.c"})
                ],
            ),
            Turn(text=""),
        ]
    )
    kit = LoopKit(provider)
    kit.add_tool("mail__send_message")
    kit.events.events[1] = _agent_outcome_event(1)
    kit.loop.submit_message(_msg("send it again"))

    await kit.loop._tick()

    assert len(kit.approvals.created) == 1
    assert kit.connector.approvals_presented[0][0] == kit.approvals.created[0].id


async def test_the_same_call_is_never_parked_twice_in_one_turn() -> None:
    """Two identical proposals inside one turn put one question in front
    of the user, not two approval rows — the second gets the first's
    held answer instead of a fresh park."""
    provider = FakeProvider(
        [
            Turn(
                text="",
                tool_calls=[
                    ToolCall(id="t1", name="mail__send_message", arguments={"to": "a@b.c"})
                ],
            ),
            Turn(
                text="",
                tool_calls=[
                    ToolCall(id="t2", name="mail__send_message", arguments={"to": "a@b.c"})
                ],
            ),
            Turn(text=""),
        ]
    )
    kit = LoopKit(provider)
    kit.add_tool("mail__send_message")
    kit.loop.submit_message(_msg("send an email to sam"))

    await kit.loop._tick()

    assert len(kit.approvals.created) == 1
    assert len(kit.connector.approvals_presented) == 1
    # the model saw the same held-for-approval answer both times
    assert sum("held for approval" in c for c in _held_contents(provider, 2)) == 2


def _stub_hub(*, ready: bool) -> Any:
    """A host-shaped stub holding one connection named like the real hub."""
    conn = SimpleNamespace(name="composio", ready=ready)
    return SimpleNamespace(connections=[conn])


async def test_a_stale_hub_session_gets_one_fresh_cut() -> None:
    # a wedged router answers initialize but never tools/list, so the
    # reconnect loop rides the same dead URL forever — after sustained
    # unreadiness the watchdog has the bridge cut one fresh session
    kit = LoopKit(None)
    bridge = FakeComposioBridge()
    kit.loop._composio = bridge
    kit.loop._host = _stub_hub(ready=False)

    await kit.loop._watch_hub()  # the first sighting only arms the timer
    assert bridge.resets == 0

    kit.loop._hub_unready_since = time.monotonic() - _HUB_RESET_AFTER_SECONDS - 1
    await kit.loop._watch_hub()
    assert bridge.resets == 1
    assert any("fresh session" in s for s in kit.connector.sent)


async def test_a_ready_hub_never_resets() -> None:
    kit = LoopKit(None)
    bridge = FakeComposioBridge()
    kit.loop._composio = bridge
    kit.loop._host = _stub_hub(ready=True)

    kit.loop._hub_unready_since = time.monotonic() - 10_000
    await kit.loop._watch_hub()
    assert bridge.resets == 0
    assert kit.loop._hub_unready_since is None  # the timer clears


async def test_the_tick_expires_overdue_approvals() -> None:
    # a pending row past its TTL must never read as waiting — the tick
    # flips it to expired (and the store audits it), so no surface keeps
    # offering a dead decision
    kit = LoopKit(None)
    approval = await kit.approvals.create(tool_name="mail__send_message", params={"to": "a@b.c"})
    assert [approval] == await kit.approvals.list_pending()  # staged, waiting
    approval.expires_at = datetime.now(UTC) - timedelta(hours=1)

    await kit.loop._tick()
    assert approval.status == "expired"
    assert await kit.approvals.list_pending() == []


async def test_list_pending_never_shows_an_expired_row() -> None:
    # even before the tick's expire_overdue runs, a pending-but-expired
    # row is hidden from every reader — the store's own filter
    kit = LoopKit(None)
    fresh = await kit.approvals.create(tool_name="mail__send_message", params={"to": "a@b.c"})
    stale = await kit.approvals.create(tool_name="mail__send_message", params={"to": "d@e.f"})
    stale.expires_at = datetime.now(UTC) - timedelta(minutes=5)

    pending = await kit.approvals.list_pending()
    assert pending == [fresh]


async def test_hub_resets_respect_the_cooldown() -> None:
    # a Composio-side outage won't be fixed by minting sessions — after the
    # first cut the watchdog leaves it alone for the cooldown
    kit = LoopKit(None)
    bridge = FakeComposioBridge()
    kit.loop._composio = bridge
    kit.loop._host = _stub_hub(ready=False)

    kit.loop._hub_unready_since = time.monotonic() - 10_000
    await kit.loop._watch_hub()
    assert bridge.resets == 1

    kit.loop._hub_unready_since = time.monotonic() - 10_000
    await kit.loop._watch_hub()
    assert bridge.resets == 1  # swallowed by the cooldown


async def test_the_web_panel_decide_path_carries_out_a_fresh_approval() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__send_message", result="sent")

    await kit.loop._execute(
        ToolCall(id="t1", name="mail__send_message", arguments={"to": "a@b.c", "body": "hi"})
    )
    approval = kit.approvals.created[0]
    approval.status = APPROVED
    kit.approvals.result = approval  # what the store hands back on a fresh decide

    assert await kit.loop.decide(approval.id, "approve") is approval
    assert kit.executed == [("mail__send_message", {"to": "a@b.c", "body": "hi"})]
    assert kit.approvals.executed == [approval.id]
    assert any("Sent the email" in s for s in kit.connector.sent)


async def test_a_stale_web_decide_touches_nothing() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__send_message")
    kit.approvals.result = None  # already decided or expired

    assert await kit.loop.decide(1, "approve") is None
    assert kit.executed == []
    assert kit.approvals.executed == []


async def test_denied_by_policy_calls_are_refused_up_front() -> None:
    kit = LoopKit(
        None,
        rules=[AuthzRule(tool_pattern="mail__send_message", decision="deny", note="no sends")],
    )
    kit.add_tool("mail__send_message")

    result = await kit.loop._execute(ToolCall(id="t1", name="mail__send_message", arguments={}))
    assert result.is_error is True
    assert "denied by policy" in result.content
    assert kit.executed == []
    assert kit.audit.entries[-1]["decision"] == "deny"


async def test_unavailable_tools_come_back_in_plain_words() -> None:
    # the model relays these results to the user verbatim, so they carry no
    # internal tool names and no raw error text — the trace keeps the details
    kit = LoopKit(None)

    # an app that isn't linked at all (allow-classified, so the gate lets it
    # through to the registry, which is where the miss becomes a plain word)
    result = await kit.loop._execute(ToolCall(id="t1", name="gmail__list_messages", arguments={}))
    assert result.is_error is True
    assert result.content == "gmail isn't connected right now"
    assert "gmail__list_messages" not in result.content
    assert "unknown tool" not in result.content

    # a linked app that failed to answer mid-call
    from aether.connectors.base import ConnectorUnavailableError

    async def down(params: dict) -> str:
        raise ConnectorUnavailableError("mail server is not connected")

    kit.tools.add_native(ToolSpec(name="mail__list_down", description=""), down)
    result = await kit.loop._execute(ToolCall(id="t2", name="mail__list_down", arguments={}))
    assert result.is_error is True
    assert result.content == "mail isn't responding right now"
    assert "connector unavailable" not in result.content
    assert "mcp server" not in result.content.lower()


async def test_a_wrong_name_in_a_linked_app_is_plain_about_what_is_missing() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__list_messages", result="2 unread: alice, bob")

    # the model hallucinated an action a linked app doesn't offer
    result = await kit.loop._execute(ToolCall(id="t1", name="mail__search_ghost", arguments={}))
    assert result.is_error is True
    assert result.content == "that action isn't available in mail right now"
    assert "mail__search_ghost" not in result.content

    # a made-up native name gets the same treatment, without naming anything
    result = await kit.loop._execute(ToolCall(id="t2", name="memory_search_fast", arguments={}))
    assert result.is_error is True
    assert result.content == "that action isn't available right now"
    assert "memory_search_fast" not in result.content


async def test_a_risky_call_that_cannot_run_never_parks_an_approval() -> None:
    # the regression this guards: a risky name that isn't registered parked
    # an approval anyway, the user approved certain failure, and the error
    # that followed leaked the raw tool name. A call the namespace can't
    # run answers plainly up front — consent to certain failure is not a
    # decision worth interrupting the user for.
    kit = LoopKit(None)  # no servers, no tools

    result = await kit.loop._execute(
        ToolCall(id="t1", name="gmail__send_message", arguments={"to": "a@b.c"})
    )
    assert result.is_error is True
    assert result.content == "gmail isn't connected right now"
    assert "gmail__send_message" not in result.content
    assert kit.approvals.created == []  # nothing was parked
    assert kit.connector.approvals_presented == []  # and nobody was asked


async def test_a_failed_carry_out_speaks_plainly_and_closes_the_approval() -> None:
    # rows parked before the up-front check — or an app unlinked while one
    # sat pending — can still fail at carry-out: the note speaks plainly, the
    # approval is closed as failed, and the record keeps exactly what the
    # chat line left out
    kit = LoopKit(None)
    approval = await kit.approvals.create(tool_name="gmail__send_message", params={"to": "a@b.c"})
    approval.status = APPROVED

    await kit.loop.execute_decision(approval.id, APPROVED)
    assert kit.executed == []
    assert kit.approvals.failed == [approval.id]
    assert kit.approvals.executed == []  # it didn't run; it must not read as ran

    note = kit.connector.sent[-1]
    assert "couldn't run" in note
    assert "gmail isn't connected right now" in note
    assert "gmail__send_message" not in note  # never the raw name in chat
    assert "unknown tool" not in note  # never the raw error either

    payload = kit.traces.created[-1]["payload"]
    assert payload["tool"] == "gmail__send_message"  # the record keeps it
    assert "unknown tool" in payload["result"]
    assert payload["is_error"] is True


async def test_a_linked_app_that_dies_before_carry_out_is_said_plainly() -> None:
    from aether.connectors.base import ConnectorUnavailableError

    kit = LoopKit(None)

    async def down(params: dict) -> str:
        raise ConnectorUnavailableError("mail server is not connected")

    kit.tools.add_native(ToolSpec(name="mail__send_message", description=""), down)
    # registered, so the gate parks it in good faith — the server is only
    # discovered down when the approved call runs
    result = await kit.loop._execute(
        ToolCall(id="t1", name="mail__send_message", arguments={"to": "a@b.c"})
    )
    assert "held for approval" in result.content
    approval = kit.approvals.created[0]
    approval.status = APPROVED

    await kit.loop.execute_decision(approval.id, APPROVED)
    assert kit.approvals.failed == [approval.id]
    note = kit.connector.sent[-1]
    assert "mail isn't responding right now" in note
    assert "mail__send_message" not in note
    assert "connector unavailable" not in note


# ---------------------------------------------------------------------------
# the tick: messages, observations, quiet
# ---------------------------------------------------------------------------


async def test_inbound_message_becomes_memory_and_reaches_the_model() -> None:
    provider = FakeProvider([Turn(text="hello back")])
    kit = LoopKit(provider)
    kit.loop.submit_message(_msg("what do you remember?"))
    await kit.loop._tick()

    assert len(kit.events.ingested) == 1
    ingested = kit.events.ingested[0]
    assert ingested["source"] == "telegram"
    assert ingested["kind"] == "chat_message"
    assert ingested["sender"].handle == "@vedant"
    assert kit.salience.scored  # the chat was scored for memorability

    messages = provider.calls[0][1]
    assert any("what do you remember?" in m.text for m in messages)
    assert any("[message from @vedant via telegram]" in m.text for m in messages)
    assert kit.connector.sent == ["hello back"]


async def test_a_dying_solicited_turn_says_so_in_plain_words() -> None:
    # the model failing after the user spoke must never leave silence —
    # plain words go out, then the failure still propagates to the tick
    provider = FakeProvider([Turn(text="unused")])
    kit = LoopKit(provider)

    async def dead(system: str, messages: list, tools: list) -> Turn:
        raise RuntimeError("provider is down")

    provider.complete = dead  # type: ignore[method-assign]
    kit.loop.submit_message(_msg("what's on my calendar?"))
    try:
        await kit.loop._tick()
        raise AssertionError("the turn failure must propagate")
    except RuntimeError:
        pass

    assert kit.connector.sent == [
        "Something went wrong on my side — I couldn't finish "
        "that just now. Nothing else is affected."
    ]


async def test_a_dying_unprompted_turn_stays_quiet() -> None:
    # a background tick that fails is a log line, never an interruption —
    # nobody asked, so nobody is told
    provider = FakeProvider([Turn(text="unused")])
    kit = LoopKit(provider)

    async def dead(system: str, messages: list, tools: list) -> Turn:
        raise RuntimeError("provider is down")

    provider.complete = dead  # type: ignore[method-assign]
    kit.events.events[1] = _event(1)
    try:
        await kit.loop._tick()
        raise AssertionError("the turn failure must propagate")
    except RuntimeError:
        pass

    assert kit.connector.sent == []


async def test_handle_inbound_is_awaitable_and_lands_the_message() -> None:
    # Connectors await their InboundHandler; the loop's entry for them must
    # return an awaitable or every DM crashes ("'NoneType' object can't be
    # awaited") — awaiting here is the regression.
    provider = FakeProvider([Turn(text="hello back")])
    kit = LoopKit(provider)
    await kit.loop.handle_inbound(_msg("what do you remember?"))
    await kit.loop._tick()

    assert len(kit.events.ingested) == 1
    assert kit.connector.sent == ["hello back"]


async def test_filtered_senders_never_reach_the_model() -> None:
    provider = FakeProvider([])  # would AssertionError if the model were called
    kit = LoopKit(provider)
    kit.events.ingest_result = IngestResult(stored=False, reason="filtered:contacts")
    kit.loop.submit_message(_msg("you should not see this"))
    await kit.loop._tick()

    assert provider.calls == []
    assert kit.connector.sent == []


# ---------------------------------------------------------------------------
# conversation memory: the transcript (chat_messages) re-enters every turn
# ---------------------------------------------------------------------------


async def test_transcript_reaches_the_model_with_roles() -> None:
    provider = FakeProvider([Turn(text="the atoms one")])
    transcript = FakeTranscript(
        rows=[
            {"surface": "telegram", "direction": "in", "text": "tell a joke", "at": ""},
            {
                "surface": "telegram",
                "direction": "out",
                "text": "why don't scientists trust atoms?",
                "at": "",
            },
        ]
    )
    kit = LoopKit(provider, transcript=transcript)
    kit.loop.submit_message(_msg("what was the joke?"))
    await kit.loop._tick()

    messages = provider.calls[0][1]
    assert any(m.role == "user" and "[telegram] tell a joke" in m.text for m in messages)
    assert any(m.role == "assistant" and "atoms" in m.text for m in messages)


async def test_current_turn_inbound_is_not_echoed_from_the_transcript() -> None:
    provider = FakeProvider([Turn(text="hi")])
    transcript = FakeTranscript()
    kit = LoopKit(provider, transcript=transcript)
    kit.loop.submit_message(_msg("hello there"))
    await kit.loop._tick()

    # the tick persisted the inbound line...
    assert transcript.appended == [("telegram", "in", "hello there")]
    # ...and the model still saw it exactly once, as the rich
    # [message from …] line — not again as a transcript echo
    messages = provider.calls[0][1]
    assert sum("hello there" in m.text for m in messages) == 1
    assert any("[message from @vedant via telegram]" in m.text for m in messages)


async def test_turn_works_without_a_transcript() -> None:
    provider = FakeProvider([Turn(text="hello back")])
    kit = LoopKit(provider)  # transcript=None — the default everywhere else
    kit.loop.submit_message(_msg("hi"))
    await kit.loop._tick()

    messages = provider.calls[0][1]
    assert any("[message from @vedant via telegram]" in m.text for m in messages)


async def test_web_inbound_is_not_double_persisted() -> None:
    provider = FakeProvider([Turn(text="hi")])
    transcript = FakeTranscript()
    kit = LoopKit(provider, transcript=transcript)
    kit.loop.submit_message(_msg("from the browser", surface="web"))
    await kit.loop._tick()

    # the websocket handler owns web rows; the loop must not write them again
    assert transcript.appended == []


async def test_filtered_senders_stay_out_of_the_transcript() -> None:
    provider = FakeProvider([])  # would AssertionError if the model were called
    transcript = FakeTranscript()
    kit = LoopKit(provider, transcript=transcript)
    kit.events.ingest_result = IngestResult(stored=False, reason="filtered:contacts")
    kit.loop.submit_message(_msg("you should not see this"))
    await kit.loop._tick()

    # a dropped sender never enters conversation memory either — their words
    # must not leak into a later prompt
    assert transcript.appended == []


async def test_send_to_user_persists_each_delivered_surface() -> None:
    transcript = FakeTranscript()
    connector = FakeSurfaceConnector("telegram")
    fanout = SurfaceFanout([connector], history=transcript)

    refs = await fanout.send_to_user("going out")

    assert refs == {"telegram": "m1"}
    assert transcript.appended == [("telegram", "out", "going out")]


async def test_a_failing_transcript_never_breaks_a_send() -> None:
    class BrokenTranscript(FakeTranscript):
        async def append(self, surface: str, direction: str, text: str) -> None:
            raise RuntimeError("db down")

    connector = FakeSurfaceConnector("telegram")
    fanout = SurfaceFanout([connector], history=BrokenTranscript())

    refs = await fanout.send_to_user("still delivered")

    assert refs == {"telegram": "m1"}
    assert connector.sent == ["still delivered"]


async def test_new_events_become_scored_observations() -> None:
    provider = FakeProvider([Turn(text="noted the mail")])
    kit = LoopKit(provider)
    kit.events.events[1] = _event(1)
    await kit.loop._tick()

    assert kit.salience.scored == [1]
    messages = provider.calls[0][1]
    assert any("[new event]" in m.text and "poll:unread" in m.text for m in messages)


async def test_a_quiet_tick_calls_no_llm() -> None:
    provider = FakeProvider([])  # would AssertionError if called
    kit = LoopKit(provider)
    await kit.loop._tick()
    assert provider.calls == []
    assert kit.context.builds == 0


async def test_no_provider_leaves_a_note_instead_of_silence() -> None:
    kit = LoopKit(None)
    kit.loop.submit_message(_msg("hi"))
    await kit.loop._tick()
    assert any("without an LLM provider" in s for s in kit.connector.sent)


# ---------------------------------------------------------------------------
# one delivery per turn: send_chat_message vs the final reply
# ---------------------------------------------------------------------------


async def test_a_turn_that_sent_via_tool_does_not_also_fan_out_the_reply() -> None:
    # The model can answer through send_chat_message AND write a final
    # reply — both reach the fanout, which reads as the same thing twice
    # in different wording. One delivery per turn: the tool send replaces
    # the reply, which stays in the trace for replay.
    provider = FakeProvider(
        [
            Turn(
                text="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="send_chat_message",
                        arguments={"text": "no mail server connected"},
                    )
                ],
            ),
            Turn(text="No mail server is connected right now."),
        ]
    )
    kit = LoopKit(provider)
    kit.add_send_chat_message()
    kit.loop.submit_message(_msg("can you mail"))
    await kit.loop._tick()

    assert kit.connector.sent == ["no mail server connected"]
    payload = kit.traces.created[-1]["payload"]
    assert payload["reply"] == "No mail server is connected right now."


async def test_a_denied_send_does_not_swallow_the_reply() -> None:
    # Only a send that actually delivered replaces the reply — one denied
    # by policy sent nothing, so the model's final words must still go out.
    provider = FakeProvider(
        [
            Turn(
                text="",
                tool_calls=[
                    ToolCall(id="c1", name="send_chat_message", arguments={"text": "psst"})
                ],
            ),
            Turn(text="I couldn't send that."),
        ]
    )
    kit = LoopKit(
        provider,
        rules=[AuthzRule(tool_pattern="send_chat_message", decision="deny", note="quiet mode")],
    )
    kit.add_send_chat_message()
    kit.loop.submit_message(_msg("tell me something"))
    await kit.loop._tick()

    assert kit.connector.sent == ["I couldn't send that."]


async def test_an_empty_send_does_not_swallow_the_reply() -> None:
    # The handler refuses empty text — nothing goes out — so the final
    # reply is still the turn's one delivery.
    provider = FakeProvider(
        [
            Turn(
                text="",
                tool_calls=[ToolCall(id="c1", name="send_chat_message", arguments={"text": "  "})],
            ),
            Turn(text="Nothing to send."),
        ]
    )
    kit = LoopKit(provider)
    kit.add_send_chat_message()
    kit.loop.submit_message(_msg("say something"))
    await kit.loop._tick()

    assert kit.connector.sent == ["Nothing to send."]


# ---------------------------------------------------------------------------
# the interruption budget: quiet hours
# ---------------------------------------------------------------------------


def _quiet_kit(provider: FakeProvider | None) -> LoopKit:
    """A loop inside quiet hours around the clock ("00:00-23:59" is quiet for
    all but the day's final minute) — every waiting path of the budget is on."""
    return LoopKit(
        provider,
        config=AppConfig(agent=AgentConfig(quiet_hours="00:00-23:59", quiet_urgent_salience=8.0)),
    )


async def test_quiet_hours_hold_unprompted_commentary() -> None:
    provider = FakeProvider([Turn(text="saw the mail")])
    kit = _quiet_kit(provider)
    kit.events.events[1] = _event(1)  # salience 0 — routine feed noise
    await kit.loop._tick()

    assert kit.connector.sent == []
    assert kit.loop._held == ["saw the mail"]
    # the words exist in the record — they just haven't interrupted anyone yet
    payload = kit.traces.created[-1]["payload"]
    assert payload["reply"] == "saw the mail"


async def test_urgent_events_still_interrupt_quiet_hours() -> None:
    provider = FakeProvider([Turn(text="prod is down — on it")])
    kit = _quiet_kit(provider)
    kit.events.events[1] = _event(1, kind="alert:down", salience=9.0)
    await kit.loop._tick()

    assert kit.connector.sent == ["prod is down — on it"]
    assert kit.loop._held == []


async def test_replies_to_the_user_never_wait_out_quiet_hours() -> None:
    provider = FakeProvider([Turn(text="here now")])
    kit = _quiet_kit(provider)
    kit.loop.submit_message(_msg("you up?"))
    await kit.loop._tick()

    assert kit.connector.sent == ["here now"]
    assert kit.loop._held == []


async def test_held_notes_ship_as_one_message_when_the_window_ends() -> None:
    provider = FakeProvider([Turn(text="first held")])
    kit = _quiet_kit(provider)
    kit.events.events[1] = _event(1)
    await kit.loop._tick()
    provider.turns.append(Turn(text="second held"))
    kit.events.events[2] = _event(2)
    await kit.loop._tick()
    assert kit.connector.sent == []
    assert kit.loop._held == ["first held", "second held"]

    # the window ends: the backlog ships as ONE message, even on a tick with
    # nothing else to say
    kit.loop._quiet = None
    await kit.loop._tick()
    assert kit.connector.sent == ["While it was quiet:\n\nfirst held\n\nsecond held"]
    assert kit.loop._held == []


async def test_a_routine_acts_at_night_but_its_note_waits() -> None:
    # a taught routine is solicited by the teaching — its action still runs
    # during quiet hours; only the meta-commentary about the fire waits
    kit = _quiet_kit(None)
    kit.add_tool("note_entity", result="noted")
    kit.routines.add(
        label="billing",
        trigger={"source": "mail"},
        action={"type": "tool", "tool": "note_entity", "params": {"note": "invoice"}},
    )
    kit.events.events[1] = _event(1)
    await kit.loop._tick()

    assert kit.executed == [("note_entity", {"note": "invoice"})]
    assert kit.connector.sent == []
    assert any("routine" in h and "billing" in h for h in kit.loop._held)


# ---------------------------------------------------------------------------
# source polls
# ---------------------------------------------------------------------------


def _poll_host(kit: LoopKit, every_minutes: float = 5.0, handler=None) -> SimpleNamespace:
    """A host-shaped double: one server with one configured poll tool.
    `handler` overrides the poll call itself (e.g. to make it fail)."""
    config = MCPServerConfig(
        name="mail",
        poll_tools=[PollTool(tool="list_unread", args={"limit": 5}, every_minutes=every_minutes)],
    )
    connection = SimpleNamespace(name="mail", poll_tools=config.poll_tools)
    calls: list[tuple[str, dict]] = []

    async def call(qualified: str, args: dict) -> str:
        calls.append((qualified, dict(args)))
        return "3 unread: alice re: demo, bob, carol"

    kit.loop._poll_targets["mail:list_unread:0"] = ("mail", config.poll_tools[0])
    kit.loop._next_poll["mail:list_unread:0"] = datetime.now(UTC)
    kit.loop._host = SimpleNamespace(connections=[connection], call=handler or call)
    return SimpleNamespace(calls=calls)


async def test_source_polls_run_ingest_and_audit() -> None:
    kit = LoopKit(FakeProvider([Turn(text="watching")]))
    host = _poll_host(kit)

    await kit.loop._run_due_polls()
    assert host.calls == [("mail__list_unread", {"limit": 5})]
    assert kit.events.ingested[-1]["source"] == "mail"
    assert kit.events.ingested[-1]["kind"] == "poll:list_unread"
    assert "3 unread" in kit.events.ingested[-1]["payload"]["result"]
    entry = kit.audit.entries[-1]
    assert entry["actor"] == "scheduler"
    assert entry["rules_matched"] == "config:poll_tools"

    # the schedule advanced — an immediate second pass does not re-poll
    await kit.loop._run_due_polls()
    assert len(host.calls) == 1


async def test_the_watchdog_announces_a_dead_connector_once_and_its_recovery() -> None:
    # polls are the heartbeat: the first failure is silent, the second
    # announces the server is down once, further failures don't repeat it,
    # and the next good poll announces the recovery — which re-arms the
    # cycle, so a later outage is announced again.
    from aether.connectors.base import ConnectorUnavailableError

    kit = LoopKit(FakeProvider([]))
    attempts = 0

    async def flaky(qualified: str, args: dict) -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 3 or attempts >= 5:
            raise ConnectorUnavailableError("mail server is not connected")
        return "3 unread: alice re: demo, bob, carol"

    _poll_host(kit, handler=flaky)
    key = "mail:list_unread:0"

    def due_again() -> None:
        kit.loop._next_poll[key] = datetime.now(UTC)

    due_again()
    await kit.loop._run_due_polls()
    assert kit.connector.sent == []  # one failure alone says nothing

    due_again()
    await kit.loop._run_due_polls()
    assert kit.connector.sent == [
        "mail hasn't been responding — I've paused relying on it and will say when it's back."
    ]

    due_again()
    await kit.loop._run_due_polls()
    assert len(kit.connector.sent) == 1  # still down, still just the one note

    due_again()
    await kit.loop._run_due_polls()
    assert kit.connector.sent[-1] == "mail is responding again."

    due_again()
    await kit.loop._run_due_polls()
    assert len(kit.connector.sent) == 2  # fresh cycle: first failure is silent

    due_again()
    await kit.loop._run_due_polls()
    assert len(kit.connector.sent) == 3  # and the second outage announces again


# ---------------------------------------------------------------------------
# scheduled actions
# ---------------------------------------------------------------------------


async def test_scheduled_actions_pass_the_gate_at_fire_time() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__send_message", result="sent")
    when = datetime.now(UTC) + timedelta(hours=1)

    # an allowed scheduled action runs through the executor
    action = ScheduledAction(
        id=7,
        label="evening summary",
        run_at=when,
        status="pending",
        payload={"type": "tool", "tool": "mail__list_messages", "params": {"limit": 1}},
        created_at=when,
    )
    kit.add_tool("mail__list_messages", result="all caught up")
    content = await kit.loop.execute_scheduled(action)
    assert content == "all caught up"
    assert kit.executed[-1] == ("mail__list_messages", {"limit": 1})

    # a risky scheduled action parks for approval instead of running
    risky = ScheduledAction(
        id=8,
        label="send the file",
        run_at=when,
        status="pending",
        payload={"type": "tool", "tool": "mail__send_message", "params": {"to": "a@b.c"}},
        created_at=when,
    )
    content = await kit.loop.execute_scheduled(risky)
    assert "held for approval" in content
    assert kit.approvals.created[-1].tool_name == "mail__send_message"


async def test_scheduled_denials_raise_for_the_worker_to_record() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__send_message")
    kit.loop._policy = Policy([AuthzRule(tool_pattern="mail__send_message", decision="deny")])
    action = ScheduledAction(
        id=9,
        label="never",
        run_at=datetime.now(UTC),
        status="pending",
        payload={"type": "tool", "tool": "mail__send_message", "params": {}},
        created_at=datetime.now(UTC),
    )
    try:
        await kit.loop.execute_scheduled(action)
        raise AssertionError("a denied scheduled action must raise, not pass")
    except RuntimeError as exc:
        assert "denied by policy" in str(exc)


# ---------------------------------------------------------------------------
# the system prompt: the model knows what it can reach
# ---------------------------------------------------------------------------


def test_the_system_prompt_is_formatted_safely() -> None:
    text = SYSTEM_PROMPT.format(owner="the user", apps="none")
    assert "You are Aether" in text
    assert "plain words" in text  # failures are worded for the user, not the log
    assert "Never invent a tool name" in text  # no promised capabilities
    assert "{" not in text.replace("{owner}", "")  # no stray braces left behind


async def test_the_prompt_tells_the_model_which_apps_are_connected() -> None:
    # "can you send emails?" with no mail server must get an honest no —
    # the prompt states the live namespace, so the model never promises a
    # capability it doesn't have
    provider = FakeProvider([Turn(text="no mail is connected right now")])
    kit = LoopKit(provider)
    kit.tools.attach_mcp(
        SimpleNamespace(tool_specs=lambda: [ToolSpec(name="calendar__list_events", description="")])
    )
    kit.tools.sync_mcp_tools()
    kit.loop.submit_message(_msg("can you send emails?"))
    await kit.loop._tick()

    system = provider.calls[0][0]
    assert "Apps connected right now: calendar." in system
    assert "Apps connected right now: none" not in system


async def test_with_no_apps_connected_the_prompt_says_so() -> None:
    provider = FakeProvider([Turn(text="nothing is connected")])
    kit = LoopKit(provider)
    kit.loop.submit_message(_msg("what can you reach?"))
    await kit.loop._tick()

    assert "Apps connected right now: none." in provider.calls[0][0]


async def test_system_prompt_appends_the_owners_personality_text() -> None:
    # the personality block from the settings store rides on top of the
    # prompt — a tone overlay the owner writes, never a way around the rules
    kit = LoopKit(None, agent_settings=FakeAgentSettings("Be extremely terse. Use no emoji."))
    prompt = await kit.loop._system_prompt()
    base = SYSTEM_PROMPT.format(owner="the user", apps=kit.loop._connected_apps())
    assert prompt.startswith(base)
    assert "Be extremely terse. Use no emoji." in prompt


async def test_system_prompt_is_unchanged_with_no_personality_configured() -> None:
    kit = LoopKit(None, agent_settings=FakeAgentSettings(""))
    assert await kit.loop._system_prompt() == SYSTEM_PROMPT.format(
        owner="the user", apps=kit.loop._connected_apps()
    )


async def test_system_prompt_is_unchanged_with_no_settings_store_wired() -> None:
    kit = LoopKit(None)  # agent_settings defaults to None
    assert await kit.loop._system_prompt() == SYSTEM_PROMPT.format(
        owner="the user", apps=kit.loop._connected_apps()
    )


async def test_system_prompt_carries_the_workspace_context(tmp_path) -> None:
    workspace = Workspace(tmp_path / "ws")
    workspace.ensure_scaffold()
    workspace.remember("prefers concise notifications", "preference")
    kit = LoopKit(None, workspace=workspace)
    prompt = await kit.loop._system_prompt()
    base = SYSTEM_PROMPT.format(owner="the user", apps=kit.loop._connected_apps())
    assert prompt.startswith(base)
    assert "[workspace]" in prompt
    assert "prefers concise notifications" in prompt
    # the first-run BOOTSTRAP.md that ensure_scaffold() just created rides
    # along too, so the agent actually sees the setup questions
    assert "First-run setup" in prompt


async def test_system_prompt_is_unchanged_with_no_workspace_wired() -> None:
    kit = LoopKit(None)  # workspace defaults to None
    assert await kit.loop._system_prompt() == SYSTEM_PROMPT.format(
        owner="the user", apps=kit.loop._connected_apps()
    )


# ---------------------------------------------------------------------------
# the deterministic chat vocabulary: /verify and the why? reply
# ---------------------------------------------------------------------------


async def test_verify_command_answers_from_the_record_without_waking_the_model() -> None:
    provider = FakeProvider([])  # any LLM call would fail the test
    kit = LoopKit(provider)
    await kit.audit.append(actor="agent", tool_name="mail__list_unread", decision="allow")
    kit.loop.submit_message(_msg("/verify"))
    await kit.loop._tick()

    note = kit.connector.sent[-1]
    assert "decision record" in note and "chain intact" in note
    assert "moments ago" in note  # the row was just written
    assert provider.calls == []  # deterministic — no LLM turn
    assert kit.events.ingested == []  # and nothing became memory
    assert kit.traces.created == []  # a read leaves no record of its own


async def test_verify_command_reports_a_broken_chain_in_plain_words() -> None:
    provider = FakeProvider([])  # any LLM call would fail the test
    kit = LoopKit(provider)
    kit.audit.verification = ChainVerification(
        ok=False,
        entries=892,
        first_bad_seq=892,
        problem="seq 892: stored hash does not match the entry contents",
    )
    kit.loop.submit_message(_msg("/verify"))
    await kit.loop._tick()

    note = kit.connector.sent[-1]
    assert "decision record" in note and "BROKEN at entry #892" in note
    assert provider.calls == []


async def test_verify_command_survives_a_dead_record() -> None:
    provider = FakeProvider([])
    kit = LoopKit(provider)

    async def dead() -> object:
        raise RuntimeError("the database is down")

    kit.audit.verify_chain = dead  # type: ignore[method-assign]
    kit.loop.submit_message(_msg("/verify"))
    await kit.loop._tick()

    assert "couldn't verify" in kit.connector.sent[-1]
    assert provider.calls == []


async def test_verify_answers_even_during_quiet_hours() -> None:
    # the user asked for the check — solicited words never wait out the window
    kit = _quiet_kit(FakeProvider([]))
    await kit.audit.append(actor="agent", tool_name="t", decision="allow")
    kit.loop.submit_message(_msg("/verify"))
    await kit.loop._tick()

    assert "decision record" in kit.connector.sent[-1]
    assert kit.loop._held == []


async def test_an_unknown_command_flows_to_the_model() -> None:
    provider = FakeProvider([Turn(text="there's no /help — just ask me")])
    kit = LoopKit(provider)
    kit.loop.submit_message(_msg("/help"))
    await kit.loop._tick()

    assert len(kit.events.ingested) == 1  # an unhandled command is just words
    assert any("/help" in m.text for m in provider.calls[0][1])
    assert kit.connector.sent == ["there's no /help — just ask me"]


async def test_a_replied_turn_records_where_the_words_landed() -> None:
    provider = FakeProvider([Turn(text="hello back")])
    kit = LoopKit(provider)
    kit.loop.submit_message(_msg("you there?"))
    await kit.loop._tick()

    payload = kit.traces.created[-1]["payload"]
    assert payload["chat_refs"] == {"fake": "m1"}  # what a later why? finds


async def test_a_held_note_records_no_chat_refs() -> None:
    provider = FakeProvider([Turn(text="saw the mail")])
    kit = _quiet_kit(provider)
    kit.events.events[1] = _event(1)
    await kit.loop._tick()

    payload = kit.traces.created[-1]["payload"]
    assert payload["reply"] == "saw the mail"
    assert "chat_refs" not in payload  # nothing landed anywhere yet


async def test_a_why_reply_is_answered_from_the_record() -> None:
    provider = FakeProvider([])  # any LLM call would fail the test
    kit = LoopKit(provider)
    kit.traces.add(
        kind="turn",
        label="email alice",
        payload={
            "trigger": {"messages": [], "observations": []},
            "calls": [],
            "reply": "on it — sent once you approve",
            "chat_refs": {"telegram": "101"},
        },
    )
    kit.loop.submit_message(_msg("why?", reply_to_id="101"))
    await kit.loop._tick()

    replay = kit.connector.sent[-1]
    assert replay.startswith("that message, from the record")
    assert "on it — sent once you approve" in replay
    assert provider.calls == []  # the record answered, not the model
    assert kit.traces.created == []  # a replay is a read — no new trace
    assert kit.events.ingested == []  # and not memory either


async def test_a_why_reply_to_an_unrecorded_message_reaches_the_model() -> None:
    provider = FakeProvider([Turn(text="I'm not sure what you mean")])
    kit = LoopKit(provider)
    kit.loop.submit_message(_msg("why?", reply_to_id="999", reply_to_text="the earlier words"))
    await kit.loop._tick()

    # nothing linked the reply, so the model answers — and it can see which
    # of its own messages the question was about
    history = provider.calls[0][1]
    assert any("(replying to my earlier message:" in m.text for m in history)
    assert kit.connector.sent == ["I'm not sure what you mean"]


async def test_a_carried_out_approval_records_where_the_news_landed() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__send_message", result="sent")
    await kit.loop._execute(
        ToolCall(id="t1", name="mail__send_message", arguments={"to": "a@b.c", "body": "hi"})
    )
    approval = kit.approvals.created[0]
    approval.status = APPROVED

    await kit.loop.execute_decision(approval.id, APPROVED)

    payload = kit.traces.created[-1]["payload"]
    assert payload["chat_refs"] == {"fake": "m1"}


async def test_a_routine_fire_records_where_its_ping_landed() -> None:
    kit = LoopKit(None)  # routines fire before any LLM turn
    kit.add_tool("note_entity", result="noted")
    kit.routines.add(
        label="billing",
        trigger={"source": "mail"},
        action={"type": "tool", "tool": "note_entity", "params": {"note": "invoice"}},
    )
    kit.events.events[1] = _event(1)
    await kit.loop._tick()

    payload = kit.traces.created[-1]["payload"]
    assert payload["chat_refs"] == {"fake": "m1"}


# ---------------------------------------------------------------------------
# /config — the whole vocabulary, deterministic (no model, no ingest)
# ---------------------------------------------------------------------------


async def test_config_opens_the_guided_walk_without_waking_the_model() -> None:
    provider = FakeProvider([])  # any LLM call would fail the test
    manager = FakeConfigManager(show_reply="current configuration")
    kit = LoopKit(provider, config_manager=manager)
    kit.loop.submit_message(_msg("/config"))
    await kit.loop._tick()

    # bare /config is the friendly front door: the walk's opening menu, with
    # no ingest, no model, no trace — exactly like the rest of the vocabulary
    assert kit.connector.sent == [
        (
            "let's set me up — answer each question with a number, "
            'or "stop" any time.\n'
            "1 — see my settings\n"
            "2 — change something\n"
            "3 — put something back the way config.yaml had it"
        )
    ]
    assert provider.calls == []
    assert kit.events.ingested == []  # a read never becomes memory
    assert kit.traces.created == []

    # the overview arrives inside the walk: "see" relays it, then the menu
    kit.loop.submit_message(_msg("1"))
    await kit.loop._tick()
    assert kit.connector.sent[-1] == (
        "current configuration\n\n"
        "anything else? 1 — see my settings  2 — change something  "
        '3 — reset something  (or "done")'
    )
    assert manager.show_calls == [None]  # the bare overview
    assert provider.calls == []
    assert kit.events.ingested == []


async def test_config_show_one_path_delegates_it() -> None:
    provider = FakeProvider([])
    manager = FakeConfigManager(show_reply="agent.tick_seconds = 30.0")
    kit = LoopKit(provider, config_manager=manager)
    kit.loop.submit_message(_msg("/config show agent.tick_seconds"))
    await kit.loop._tick()

    assert manager.show_calls == ["agent.tick_seconds"]
    assert kit.connector.sent == ["agent.tick_seconds = 30.0"]


async def test_config_set_tuning_applies_through_the_gate() -> None:
    provider = FakeProvider([])  # the command answers; the model never turns
    manager = FakeConfigManager(set_reply="tick interval set to 10s")
    kit = LoopKit(provider, config_manager=manager)
    kit.loop.submit_message(_msg("/config set agent.tick_seconds 10"))
    await kit.loop._tick()

    # the write went through the executor as one synthetic set_config call
    assert manager.set_calls == [
        {"op": "set", "path": "agent.tick_seconds", "value": 10, "source": "tool"}
    ]
    assert kit.connector.sent == ["tick interval set to 10s"]
    row = kit.audit.entries[-1]
    assert row["tool_name"] == "set_config"
    assert row["decision"] == "allow"
    assert row["rules_matched"] == "builtin:config-tune"
    assert provider.calls == []
    assert kit.events.ingested == []


async def test_config_set_security_parks_for_one_tap() -> None:
    provider = FakeProvider([])
    manager = FakeConfigManager()
    kit = LoopKit(provider, config_manager=manager)
    kit.loop.submit_message(_msg("/config set messaging.telegram.enabled true"))
    await kit.loop._tick()

    # parked, not applied — nothing reached the manager
    assert manager.set_calls == []
    approval = kit.approvals.created[-1]
    assert approval.tool_name == "set_config"
    assert approval.params == {"op": "set", "path": "messaging.telegram.enabled", "value": True}
    # the card went to the surfaces as one plain question
    presented = kit.connector.approvals_presented[-1]
    assert presented[0] == approval.id
    assert presented[1] == "Change my configuration (path messaging.telegram.enabled)?"
    # the reply is in the user's own words, never the model-directed wording
    note = kit.connector.sent[-1]
    assert f"one-tap approval (#{approval.id})" in note
    assert "Do not propose" not in note
    assert provider.calls == []
    assert kit.events.ingested == []


async def test_a_model_proposed_connect_still_parks() -> None:
    """The origin argument is loop-internal: a set_config the model
    proposes carries no origin, so a connect parks exactly as before —
    the direct apply belongs to the /apps walk and the callback alone."""
    kit = LoopKit(None, config_manager=FakeConfigManager())
    await kit.loop._execute(
        ToolCall(
            id="t1",
            name="set_config",
            arguments={
                "op": "add",
                "path": "mcp_servers",
                "value": {"name": "github", "transport": {}},
            },
        )
    )
    assert kit.approvals.created[-1].tool_name == "set_config"
    assert kit.approvals.created[-1].params == {
        "op": "add",
        "path": "mcp_servers",
        "value": {"name": "github", "transport": {}},
    }


async def test_config_set_without_a_value_is_refused_before_the_gate() -> None:
    provider = FakeProvider([])
    manager = FakeConfigManager()
    kit = LoopKit(provider, config_manager=manager)
    kit.loop.submit_message(_msg("/config set agent.tick_seconds"))
    await kit.loop._tick()

    assert "set needs a value" in kit.connector.sent[-1]
    assert manager.set_calls == []
    assert kit.audit.entries == []  # refused up front — nothing was executed


async def test_config_garbage_gets_the_usage_not_the_model() -> None:
    provider = FakeProvider([])
    manager = FakeConfigManager()
    kit = LoopKit(provider, config_manager=manager)
    kit.loop.submit_message(_msg("/config frobnicate the moon"))
    await kit.loop._tick()

    assert "guided walk" in kit.connector.sent[-1]  # the usage text
    assert manager.set_calls == [] and manager.show_calls == []
    assert provider.calls == []
    assert kit.events.ingested == []


async def test_config_when_unwired_answers_deterministically() -> None:
    provider = FakeProvider([])  # still no LLM turn — the command never lands
    kit = LoopKit(provider)  # no config_manager wired
    kit.loop.submit_message(_msg("/config"))
    await kit.loop._tick()

    assert kit.connector.sent == ["config management isn't wired on this instance."]
    assert provider.calls == []
    assert kit.events.ingested == []


# the guided walk — same gate, friendlier front door


async def test_a_guided_tuning_walk_applies_through_the_gate() -> None:
    provider = FakeProvider([])  # the walk answers; the model never turns
    manager = FakeConfigManager(set_reply="tick interval set to 10s")
    kit = LoopKit(provider, config_manager=manager)
    kit.loop.submit_message(_msg("/config"))
    await kit.loop._tick()
    for answer in ("2", "2", "1", "10"):  # change → agent → tick_seconds → 10
        kit.loop.submit_message(_msg(answer))
        await kit.loop._tick()

    # the write went through the executor as one synthetic set_config call —
    # the same gate as the one-liner and the model's own proposals
    assert manager.set_calls == [
        {"op": "set", "path": "agent.tick_seconds", "value": 10, "source": "tool"}
    ]
    row = kit.audit.entries[-1]
    assert row["tool_name"] == "set_config"
    assert row["decision"] == "allow"
    assert row["rules_matched"] == "builtin:config-tune"
    assert provider.calls == []
    assert kit.events.ingested == []
    # the walk relayed the confirmation and is back at the menu
    assert kit.connector.sent[-1] == (
        "tick interval set to 10s\n"
        "anything else? 1 — see my settings  2 — change something  "
        '3 — reset something  (or "done")'
    )


async def test_a_guided_security_walk_parks_for_one_tap() -> None:
    provider = FakeProvider([])
    manager = FakeConfigManager()
    kit = LoopKit(provider, config_manager=manager)
    kit.loop.submit_message(_msg("/config"))
    await kit.loop._tick()
    for answer in ("2", "4", "1", "on"):  # change → messaging → telegram → on
        kit.loop.submit_message(_msg(answer))
        await kit.loop._tick()

    # parked, not applied — nothing reached the manager
    assert manager.set_calls == []
    approval = kit.approvals.created[-1]
    assert approval.tool_name == "set_config"
    assert approval.params == {"op": "set", "path": "messaging.telegram.enabled", "value": True}
    presented = kit.connector.approvals_presented[-1]
    assert presented[0] == approval.id
    assert presented[1] == "Change my configuration (path messaging.telegram.enabled)?"
    # the walk relays the park card, then offers the menu again
    note = kit.connector.sent[-1]
    assert f"one-tap approval (#{approval.id})" in note
    assert "anything else?" in note
    assert provider.calls == []
    assert kit.events.ingested == []


async def test_an_expert_command_mid_walk_honors_expert_and_ends_the_walk() -> None:
    provider = FakeProvider([Turn(text="done")])  # the stray "2" reaches the model
    manager = FakeConfigManager(show_reply="agent.tick_seconds = 30.0")
    kit = LoopKit(provider, config_manager=manager)
    kit.loop.submit_message(_msg("/config"))
    await kit.loop._tick()
    kit.loop.submit_message(_msg("2"))
    await kit.loop._tick()
    assert "which area?" in kit.connector.sent[-1]

    # the user spoke expert — the one-liner runs, and the walk is over
    kit.loop.submit_message(_msg("/config show agent.tick_seconds"))
    await kit.loop._tick()
    assert manager.show_calls == ["agent.tick_seconds"]

    # a following "2" is normal chat now — no menu is waiting for it
    assert provider.calls == []
    kit.loop.submit_message(_msg("2"))
    await kit.loop._tick()
    assert len(provider.calls) == 1
    assert kit.events.ingested[0]["payload"]["text"] == "2"


async def test_a_stray_sentence_mid_walk_falls_through_to_normal_chat() -> None:
    provider = FakeProvider([Turn(text="done"), Turn(text="done")])
    manager = FakeConfigManager()
    kit = LoopKit(provider, config_manager=manager)
    kit.loop.submit_message(_msg("/config"))
    await kit.loop._tick()
    kit.loop.submit_message(_msg("2"))
    await kit.loop._tick()
    assert "which area?" in kit.connector.sent[-1]

    # a sentence at a menu is a changed subject — the walk steps aside
    kit.loop.submit_message(_msg("hey what did mom say about dinner?"))
    await kit.loop._tick()
    assert len(provider.calls) == 1
    assert kit.events.ingested[-1]["payload"]["text"] == "hey what did mom say about dinner?"

    # and the walk is gone: a later "1" is chat too, not a menu answer
    kit.loop.submit_message(_msg("1"))
    await kit.loop._tick()
    assert len(provider.calls) == 2


async def test_reparse_quiet_hours_flips_the_window_live() -> None:
    kit = _quiet_kit(None)
    assert kit.loop._quiet_now() is True

    kit.loop._config.agent.quiet_hours = None  # the chat-made change
    kit.loop.reparse_quiet_hours()
    assert kit.loop._quiet_now() is False

    kit.loop._config.agent.quiet_hours = "00:00-23:59"  # and back again
    kit.loop.reparse_quiet_hours()
    assert kit.loop._quiet_now() is True


# the /config line parser and value coercion — pure functions


def test_config_value_coercion() -> None:
    assert _coerce_config_value("10") == 10
    assert _coerce_config_value("2.5") == 2.5
    assert _coerce_config_value("true") is True
    assert _coerce_config_value("on") is True
    assert _coerce_config_value("off") is False
    assert _coerce_config_value("none") is None  # the explicit clear
    assert _coerce_config_value('"claude-opus-5"') == "claude-opus-5"
    assert _coerce_config_value("'23:00-07:00'") == "23:00-07:00"  # quoted is literal
    assert _coerce_config_value("00:00-23:59") == "00:00-23:59"  # not a number — raw
    assert _coerce_config_value('["a", "b"]') == ["a", "b"]
    assert _coerce_config_value('{"type": "stdio"}') == {"type": "stdio"}
    assert _coerce_config_value("claude-sonnet-5") == "claude-sonnet-5"  # a model name stays words


def test_config_command_parser_shapes() -> None:
    assert _parse_config_command("/config") == ("show", None, _MISSING)  # the bare overview
    assert _parse_config_command("/config frobnicate") == ("usage", None, _MISSING)
    assert _parse_config_command("/config show") == ("show", None, _MISSING)
    assert _parse_config_command("/config show agent") == ("show", "agent", _MISSING)
    assert _parse_config_command("/config show agent.tick_seconds") == (
        "show",
        "agent.tick_seconds",
        _MISSING,
    )
    assert _parse_config_command("/config set agent.tick_seconds 10") == (
        "set",
        "agent.tick_seconds",
        10,
    )
    assert _parse_config_command("/config set agent.quiet_hours none") == (
        "set",
        "agent.quiet_hours",
        None,
    )
    assert _parse_config_command("/config set llm.model 'gpt-5.1'") == (
        "set",
        "llm.model",
        "gpt-5.1",
    )
    assert _parse_config_command("/config reset salience.threshold") == (
        "reset",
        "salience.threshold",
        _MISSING,
    )
    assert _parse_config_command("/config add contacts.allowlist telegram @friend") == (
        "add",
        "contacts.allowlist",
        "telegram @friend",  # the raw remainder — ConfigManager reads the tokens
    )
    assert _parse_config_command("/config remove authz.rules ^mail__send") == (
        "remove",
        "authz.rules",
        "^mail__send",
    )
    assert _parse_config_command("/config set") == ("set", None, _MISSING)


# ---------------------------------------------------------------------------
# /apps — the app front door, and the promises it keeps (no model, no ingest)
# ---------------------------------------------------------------------------

_ADD_HINT = "add one with /apps add <name> — like /apps add gmail"
_GATE_LINE = "the gate: reads run free — sends, deletes and anything new ask first."


class FakeMcpHost:
    """The host's surface as the reconcile and the status read it: named
    connections with readiness flags and live tool lists (a hub grows after
    connect — `tools` and `tool_version` move together, like the real one)."""

    def __init__(self, *connections: SimpleNamespace) -> None:
        self.connections = list(connections)

    def tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name=f"{c.name}__{tool}", description="one action", source=c.name)
            for c in self.connections
            if c.ready
            for tool in getattr(c, "tools", ("act",))
        ]


async def test_apps_opens_the_walk_without_waking_the_model() -> None:
    provider = FakeProvider([])  # any LLM call would fail the test
    kit = LoopKit(provider)
    kit.loop.submit_message(_msg("/apps"))
    await kit.loop._tick()

    # bare /apps is the status, the gate line, and the add hint — no
    # question is outstanding, so nothing intercepts the next message
    assert kit.connector.sent == [f"your apps: nothing connected yet.\n{_GATE_LINE}\n{_ADD_HINT}"]
    assert kit.loop._apps_walk is None
    assert provider.calls == []
    assert kit.events.ingested == []  # a read never becomes memory
    assert kit.traces.created == []


async def test_the_apps_status_is_honest_about_both_halves(tmp_path) -> None:
    provider = FakeProvider([])
    config = AppConfig(
        messaging=MessagingConfig(
            telegram=PlatformToggle(enabled=True),
            discord=PlatformToggle(enabled=True),
        ),
        mcp_servers=[
            MCPServerConfig(name="mail", transport=TransportConfig(type="http", url="https://x")),
            MCPServerConfig(
                name="calendar", transport=TransportConfig(type="http", url="https://y")
            ),
        ],
    )
    kit = LoopKit(provider, config=config)
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=yes\n")  # discord's token is still missing
    kit.loop._resolver = EnvResolver(env_file)
    host = FakeMcpHost(
        SimpleNamespace(name="mail", ready=True),
        SimpleNamespace(name="calendar", ready=False),
    )
    kit.tools.attach_mcp(host)
    kit.tools.sync_mcp_tools()  # the boot sync: mail is in, calendar isn't
    kit.loop._host = host

    kit.loop.submit_message(_msg("/apps"))
    await kit.loop._tick()
    assert kit.connector.sent == [
        (
            "your apps:\n"
            "· telegram — on\n"
            "· discord — on — no token yet\n"
            "· mail — connected · 1 action\n"
            "· calendar — still connecting — I'll tell you the moment it's up\n"
            f"{_GATE_LINE}\n{_ADD_HINT}"
        )
    ]
    assert provider.calls == []
    assert kit.events.ingested == []


class FakeSecretStore:
    """SecretStore double: takes the paste, keeps it by name — the loop's
    store call is the only place the value ever lands. `get`/`names` are
    the resolver's read side: a stored paste is immediately known."""

    def __init__(self) -> None:
        self.saved: dict[str, str] = {}

    async def set(self, name: str, value: str) -> None:
        self.saved[name] = value

    async def get(self, name: str) -> str | None:
        return self.saved.get(name)

    async def names(self) -> set[str]:
        return set(self.saved)


async def test_a_pasted_key_connects_directly_and_never_reaches_the_model(
    tmp_path,
) -> None:
    """The headline /apps transcript, end to end: the paste is consumed
    before ingest (no event, no model turn, never echoed), stored
    encrypted with the audit naming the key only, and the pick-and-paste
    applies the connect directly — origin "walk" through the same gate."""
    provider = FakeProvider([Turn(text="done")])  # only the post-walk chat turn
    manager = FakeConfigManager(
        set_reply="messaging.telegram.enabled set to true (config.yaml says false)."
    )
    kit = LoopKit(provider, config_manager=manager)
    kit.loop._resolver = EnvResolver(tmp_path / ".env")  # empty .env: nothing set
    store = FakeSecretStore()
    kit.loop._secret_store = store

    kit.loop.submit_message(_msg("/apps add telegram"))
    await kit.loop._tick()
    kit.loop.submit_message(_msg("123:AAbbCC"))  # the paste
    await kit.loop._tick()

    # consumed before ingest: no event, no model turn, never echoed back
    assert kit.events.ingested == []
    assert provider.calls == []
    assert not any("123:AAbbCC" in text for text in kit.connector.sent)

    # it went to the encrypted store, and the audit row names it only
    assert store.saved == {"TELEGRAM_BOT_TOKEN": "123:AAbbCC"}
    stored = [e for e in kit.audit.entries if e["tool_name"] == "store_app_secret"][-1]
    assert stored["actor"] == "owner"
    assert stored["tool_name"] == "store_app_secret"
    assert stored["decision"] == "allow"
    assert stored["params"] == {"name": "TELEGRAM_BOT_TOKEN", "app": "telegram"}
    assert stored["outcome"] == "stored encrypted — value never recorded"
    assert "123:AAbbCC" not in str(kit.audit.entries)

    # the pick-and-paste was the approval: applied, not parked
    assert kit.approvals.created == []
    assert manager.set_calls == [
        {
            "op": "set",
            "path": "messaging.telegram.enabled",
            "value": True,
            "source": "tool",
        }
    ]
    applied = kit.audit.entries[-1]
    assert applied["tool_name"] == "set_config"
    assert applied["decision"] == "allow"
    assert applied["rules_matched"] == "builtin:walk-connect"
    assert kit.connector.sent[-1] == (
        "stored — switching telegram on now.\n"
        "messaging.telegram.enabled set to true (config.yaml says false)."
    )
    assert kit.loop._apps_walk is None  # done — nothing is waiting

    # normal chat flows again
    kit.loop.submit_message(_msg("thanks"))
    await kit.loop._tick()
    assert len(provider.calls) == 1


async def test_an_unwired_store_refuses_a_paste_honestly(tmp_path) -> None:
    """A loop without the store wired (a bare test loop) can't save a
    paste — and says so instead of pretending."""
    provider = FakeProvider([])
    kit = LoopKit(provider, config_manager=FakeConfigManager())
    kit.loop._resolver = EnvResolver(tmp_path / ".env")
    kit.loop.submit_message(_msg("/apps add telegram"))
    await kit.loop._tick()
    kit.loop.submit_message(_msg("123:AAbbCC"))
    await kit.loop._tick()
    assert kit.connector.sent[-1] == (
        "that didn't work — nothing was stored. add TELEGRAM_BOT_TOKEN "
        'to .env and reply "done", or paste it again.'
    )
    assert kit.loop._apps_walk is not None  # still waiting — nothing was lost


async def test_a_user_rule_still_parks_a_walk_connect_and_chat_flows_again(
    tmp_path,
) -> None:
    """Origin "walk" is the loop's argument, not a promise: the user's own
    rule forces the hold the origin would have skipped, and the walk
    relays the familiar card verbatim before letting go."""
    provider = FakeProvider([Turn(text="done")])  # only the post-walk chat turns
    manager = FakeConfigManager()
    kit = LoopKit(
        provider,
        rules=[AuthzRule(tool_pattern=r"set_config", decision="approve")],
        config_manager=manager,
    )
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=123:present\n")
    kit.loop._resolver = EnvResolver(env_file)

    kit.loop.submit_message(_msg("/apps add telegram"))
    await kit.loop._tick()

    # the walk's write went through the same gate — and the user's rule won
    assert kit.approvals.created[-1].params == {
        "op": "set",
        "path": "messaging.telegram.enabled",
        "value": True,
    }
    note = kit.connector.sent[-1]
    assert note.startswith("the key's already there — connecting telegram now.")
    assert "one-tap approval" in note  # the card, relayed by the walk verbatim
    assert manager.set_calls == []  # parked, not applied — the user's rule
    assert provider.calls == []  # the whole walk: no model, no ingest
    assert kit.events.ingested == []

    # the walk ended at the card — normal chat flows again
    kit.loop.submit_message(_msg("thanks, tapping approve now"))
    await kit.loop._tick()
    assert len(provider.calls) == 1


async def test_bare_apps_and_bare_config_replace_each_other() -> None:
    provider = FakeProvider([Turn(text="ok")])
    kit = LoopKit(provider, config_manager=FakeConfigManager())

    # bare /apps is just the status: it replaces a live config walk and
    # leaves nothing waiting — the next message is normal chat
    kit.loop.submit_message(_msg("/config"))
    await kit.loop._tick()
    assert "let's set me up" in kit.connector.sent[-1]
    kit.loop.submit_message(_msg("/apps"))
    await kit.loop._tick()
    assert kit.connector.sent[-1].endswith(_ADD_HINT)
    assert kit.loop._apps_walk is None
    kit.loop.submit_message(_msg("1"))  # no menu is waiting — this is chat
    await kit.loop._tick()
    assert len(provider.calls) == 1

    # and a live apps-add walk is replaced by bare /config
    kit.loop.submit_message(_msg("/apps add telegram"))
    await kit.loop._tick()
    assert kit.connector.sent[-1].startswith("telegram — talk to me there.")
    kit.loop.submit_message(_msg("/config"))
    await kit.loop._tick()
    kit.loop.submit_message(_msg("2"))
    await kit.loop._tick()
    assert kit.connector.sent[-1].startswith("which area?")


async def test_a_command_mid_apps_walk_ends_it_and_falls_through() -> None:
    provider = FakeProvider([Turn(text="done"), Turn(text="done")])
    kit = LoopKit(provider, config_manager=FakeConfigManager())
    kit.loop.submit_message(_msg("/apps add telegram"))
    await kit.loop._tick()
    assert kit.connector.sent[-1].startswith("telegram — talk to me there.")

    # a slash command ends the walk quietly, and the message itself is chat
    kit.loop.submit_message(_msg("/remind me to water the plants"))
    await kit.loop._tick()
    assert len(provider.calls) == 1
    assert kit.events.ingested[-1]["payload"]["text"] == "/remind me to water the plants"

    # the walk is gone: a later "1" reaches the model, not the walk
    kit.loop.submit_message(_msg("1"))
    await kit.loop._tick()
    assert len(provider.calls) == 2


async def test_the_tick_announces_a_late_ready_server_once() -> None:
    kit = LoopKit(None)  # a quiet tick — no model is needed or wanted
    host = FakeMcpHost(
        SimpleNamespace(name="stub", ready=True),
        SimpleNamespace(name="mail", ready=False),
    )
    kit.tools.attach_mcp(host)
    kit.loop._host = host
    assert not kit.tools.has_server("stub")  # the gap: ready, but not callable

    await kit.loop._tick()
    # it joined the namespace and said so, once, with its action count
    assert kit.connector.sent == ["stub is up — 1 action in my vocabulary"]
    assert kit.tools.has_server("stub")
    assert not kit.tools.has_server("mail")  # still silent — no false promise

    await kit.loop._tick()
    assert len(kit.connector.sent) == 1  # once per boot, never again


async def test_the_composio_hub_joins_and_grows_silently() -> None:
    """The directive: composio never announces its ups in chat — it speaks
    only when it's down, and the watchdog owns that side. Joining and
    growing are housekeeping."""
    kit = LoopKit(None)
    host = FakeMcpHost(SimpleNamespace(name="composio", ready=True))
    kit.tools.attach_mcp(host)
    kit.loop._host = host

    await kit.loop._tick()
    assert kit.connector.sent == []  # joined the namespace, said nothing
    assert kit.tools.has_server("composio")

    # the user approved an app on the hub — growth is quiet too; the
    # connect flow is what tells them, not the tick
    hub = host.connections[0]
    hub.tools = ["act", "gmail_send"]
    hub.tool_version = 2
    await kit.loop._tick()
    assert kit.connector.sent == []
    assert kit.tools.get("composio__gmail_send") is not None  # callable regardless
    """The hub promise: an app approved after connect shows up as callable
    actions — the tick notices the changed tool list, syncs, and says so."""
    kit = LoopKit(None)
    hub = SimpleNamespace(name="hub", ready=True, tools=["act"], tool_version=1)
    host = FakeMcpHost(hub)
    kit.tools.attach_mcp(host)
    kit.tools.sync_mcp_tools()  # the boot sync
    kit.loop._host = host

    await kit.loop._tick()  # already in the namespace: recorded, quiet
    assert kit.connector.sent == []

    # the user approved an app on the hub's dashboard — its tools grew
    hub.tools.append("calendar_add")
    hub.tool_version = 2

    await kit.loop._tick()
    assert kit.connector.sent == ["hub — 1 new action in my vocabulary"]
    assert kit.tools.get("hub__calendar_add") is not None  # actually callable

    await kit.loop._tick()
    assert len(kit.connector.sent) == 1  # the growth is said once, not repeated


async def test_a_tool_that_vanished_leaves_the_namespace() -> None:
    """The other half of honesty: a hub that lost an app doesn't keep its
    actions callable — and a loss is fixed quietly, not announced."""
    kit = LoopKit(None)
    hub = SimpleNamespace(name="hub", ready=True, tools=["act", "extra"], tool_version=1)
    host = FakeMcpHost(hub)
    kit.tools.attach_mcp(host)
    kit.tools.sync_mcp_tools()
    kit.loop._host = host
    await kit.loop._tick()  # the boot sync is the baseline now

    hub.tools.remove("extra")
    hub.tool_version = 2
    await kit.loop._tick()

    assert kit.tools.get("hub__extra") is None  # pruned, not callable
    assert kit.tools.get("hub__act") is not None  # the rest of the hub stayed
    assert kit.connector.sent == []  # quiet — nothing was gained


async def test_the_hub_connect_announces_and_asks_the_gates_question(tmp_path) -> None:
    """The hub transcript end to end: /apps add gmail pastes the hub key
    once, serves the Connect Link, and the out-of-band waiter announces
    the connect and asks the gate's question. 'Everything asks first'
    writes its authz rule through the same executor — and authz being a
    security root, it parks for the one-tap even from a walk."""
    provider = FakeProvider([])
    manager = FakeConfigManager(set_reply="applied.")
    kit = LoopKit(provider, config_manager=manager)
    store = FakeSecretStore()
    kit.loop._resolver = EnvResolver(tmp_path / ".env", secrets=store)  # no key yet
    kit.loop._secret_store = store
    kit.loop._composio = FakeComposioBridge()

    kit.loop.submit_message(_msg("/apps add gmail"))
    await kit.loop._tick()
    # no hub key yet — the one-time paste comes before any link
    assert "one key needed (one time — it connects every app)" in kit.connector.sent[-1]

    kit.loop.submit_message(_msg("csk-123"))
    await kit.loop._tick()
    assert store.saved == {"COMPOSIO_API_KEY": "csk-123"}
    link = kit.connector.sent[-1]
    assert link.startswith("Gmail — last step, let me in:")
    assert "https://hub.example.test/connect/gmail" in link
    assert kit.events.ingested == []  # the paste never became memory
    assert provider.calls == []  # and the model never woke
    # the first connect pins the composio server entry, origin "walk"
    assert manager.set_calls == [
        {
            "op": "add",
            "path": "mcp_servers",
            "value": dict(SERVER_ENTRY),
            "source": "tool",
        }
    ]

    # the click lands — the waiter announces and asks the gate's question
    while kit.loop._connect_waits:
        await asyncio.wait(kit.loop._connect_waits)
    announcement = kit.connector.sent[-1]
    assert announcement.startswith("gmail connected — gmail@example.test.")
    assert "what may I do with gmail?" in announcement

    # "everything asks first" → an authz rule — which parks, honestly relayed
    kit.loop.submit_message(_msg("2"))
    await kit.loop._tick()
    assert kit.approvals.created[-1].params == {
        "op": "add",
        "path": "authz.rules",
        "value": {
            "tool_pattern": "composio__GMAIL_",
            "decision": "approve",
            "note": "gmail connected in chat — everything asks first",
        },
    }
    assert "one-tap approval" in kit.connector.sent[-1]
    assert kit.loop._apps_walk is None  # done — nothing is waiting


async def test_a_failed_connect_says_so_in_chat(tmp_path) -> None:
    """A link that never gets its click doesn't hang the walk or stay
    silent — the waiter's honest line lands in chat."""
    kit = LoopKit(None)
    store = FakeSecretStore()
    kit.loop._resolver = EnvResolver(tmp_path / ".env", secrets=store)
    kit.loop._secret_store = store
    kit.loop._composio = FakeComposioBridge(wait_error=TimeoutError("expired"))

    kit.loop.submit_message(_msg("/apps add gmail"))
    await kit.loop._tick()
    kit.loop.submit_message(_msg("csk-123"))
    await kit.loop._tick()
    while kit.loop._connect_waits:
        await asyncio.wait(kit.loop._connect_waits)
    assert kit.connector.sent[-1] == (
        "the gmail connection didn't finish — the link may have expired. "
        "/apps add gmail to try again."
    )
