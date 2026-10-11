"""Decision-trace tests.

Unit: every act of the loop leaves a trace — a turn (calls with their
rulings, the model's words, the reply), a routine fire, a scheduled fire
(even one that fails), an approved call carried out — a quiet tick leaves
none, and a poisoned trace store never breaks the act it records. The
explain_decision tool reads them back by trace id, by the approval a call
was held as, by topic, or as the recent list. Integration: the real store —
encrypted at rest, round-tripped by a fresh instance, recent newest first.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fakes import FakeProvider, FakeRoutines, FakeTraces
from test_agent_loop import LoopKit, _event, _msg
from test_native_tools import NativeKit

from aether.agent.traces import Traces
from aether.authz.approvals import APPROVED
from aether.authz.policy import Policy
from aether.config import AuthzRule
from aether.llm.types import ToolCall, ToolSpec, Turn
from aether.memory.crypto import Cipher, generate_key_b64
from aether.scheduler.jobs import ScheduledAction

# ---------------------------------------------------------------------------
# unit: a turn leaves a trace
# ---------------------------------------------------------------------------


def _scripted_turn() -> FakeProvider:
    """One turn proposing a read-only call, then the wrap-up reply."""
    return FakeProvider(
        [
            Turn(
                text="checking mail",
                tool_calls=[ToolCall(id="t1", name="mail__list_messages", arguments={"limit": 2})],
            ),
            Turn(text="2 unread"),
        ]
    )


def _scripted_risky() -> FakeProvider:
    """The same shape, proposing a call the gate will park."""
    return FakeProvider(
        [
            Turn(
                text="drafting the reply",
                tool_calls=[
                    ToolCall(id="t1", name="mail__send_message", arguments={"to": "a@b.c"})
                ],
            ),
            Turn(text="held for your approval"),
        ]
    )


async def test_a_turn_leaves_a_trace_with_calls_rulings_and_reply() -> None:
    provider = _scripted_turn()
    kit = LoopKit(provider)
    kit.add_tool("mail__list_messages", result="2 unread: alice, bob")
    kit.loop.submit_message(_msg("what's unread?"))
    await kit.loop._tick()

    assert len(kit.traces.created) == 1
    created = kit.traces.created[0]
    assert created["kind"] == "turn"
    assert created["label"] == "what's unread?"
    payload = created["payload"]
    assert payload["trigger"]["messages"] == [
        {
            "surface": "telegram",
            "handle": "@vedant",
            "text": "what's unread?",
            "reply_to_id": "",
            "reply_to_text": "",
        }
    ]
    assert payload["trigger"]["observations"] == []
    call = payload["calls"][0]
    assert call["name"] == "mail__list_messages"
    assert call["params"] == {"limit": 2}
    assert call["decision"] == "allow"
    assert call["matched_rule"] == "builtin:read-only"
    assert call["audit_seq"] == 1
    assert call["approval_id"] is None
    assert call["result"] == "2 unread: alice, bob"
    assert call["is_error"] is False
    assert payload["reasoning"] == ["checking mail"]  # words between calls
    assert payload["reply"] == "2 unread"  # and the reply is its own field
    assert "error" not in payload


def FakeProvider_scripted_turn():
    """One turn proposing a read-only call, then the wrap-up reply."""
    from fakes import FakeProvider

    return FakeProvider(
        [
            Turn(
                text="checking mail",
                tool_calls=[ToolCall(id="t1", name="mail__list_messages", arguments={"limit": 2})],
            ),
            Turn(text="2 unread"),
        ]
    )


async def test_an_observation_only_turn_is_traced_with_what_was_seen() -> None:
    provider = _scripted_turn()
    kit = LoopKit(provider)
    kit.add_tool("mail__list_messages")
    kit.events.events[1] = _event(1)  # no message, just a new event
    await kit.loop._tick()

    created = kit.traces.created[0]
    assert created["label"] == "observed 1 new event(s)"
    payload = created["payload"]
    assert payload["trigger"]["messages"] == []
    observation = payload["trigger"]["observations"][0]
    assert observation["id"] == 1
    assert observation["source"] == "mail"
    assert observation["kind"] == "poll:unread"
    assert "3 unread" in observation["line"]


async def test_a_parked_call_records_its_approval_id() -> None:
    provider = _scripted_risky()
    kit = LoopKit(provider)
    kit.add_tool("mail__send_message", result="sent")
    kit.loop.submit_message(_msg("reply to alice"))
    await kit.loop._tick()

    payload = kit.traces.created[0]["payload"]
    call = payload["calls"][0]
    assert call["name"] == "mail__send_message"
    assert call["decision"] == "require_approval"
    assert call["approval_id"] == 1  # the trace points at the held approval
    assert call["audit_seq"] is None  # parking is not an audit decision row
    assert kit.approvals.created[0].id == 1


async def test_a_denied_call_is_traced_with_its_reason() -> None:
    provider = _scripted_risky()
    kit = LoopKit(provider, rules=[AuthzRule(tool_pattern="mail__send_message", decision="deny")])
    kit.add_tool("mail__send_message")
    kit.loop.submit_message(_msg("reply to alice"))
    await kit.loop._tick()

    call = kit.traces.created[0]["payload"]["calls"][0]
    assert call["decision"] == "deny"
    assert call["matched_rule"] == "user:mail__send_message"  # the user rule won
    assert call["is_error"] is True
    assert "denied by policy" in call["result"]


async def test_a_crashed_turn_still_persists_its_partial_trace() -> None:
    provider = FakeProvider([])  # complete() raises: out of scripted turns
    kit = LoopKit(provider)
    kit.add_tool("mail__list_messages")
    kit.loop.submit_message(_msg("hello?"))

    with pytest.raises(AssertionError):
        await kit.loop._tick()

    created = kit.traces.created[0]
    assert created["kind"] == "turn"
    payload = created["payload"]
    assert payload["calls"] == []  # nothing got as far as the gate
    assert "AssertionError" in payload["error"]
    assert "reply" not in payload


async def test_a_quiet_tick_leaves_no_trace() -> None:
    kit = LoopKit(None)
    await kit.loop._tick()
    assert kit.traces.created == []


# ---------------------------------------------------------------------------
# unit: routines, schedules, and carried-out approvals leave traces
# ---------------------------------------------------------------------------


async def test_a_routine_fire_is_traced() -> None:
    kit = LoopKit(None)  # routines fire before any LLM is involved
    kit.add_tool("note_entity")
    kit.routines.add(
        label="billing",
        trigger={"source": "mail"},
        action={"type": "tool", "tool": "note_entity", "params": {"note": "invoice"}},
    )
    kit.events.events[1] = _event(1)

    await kit.loop._tick()

    created = kit.traces.created[0]
    assert created["kind"] == "routine"
    assert created["label"] == "billing"
    payload = created["payload"]
    assert payload["routine"] == {"id": 1, "label": "billing", "trigger": {"source": "mail"}}
    assert payload["event"]["id"] == 1
    assert payload["event"]["source"] == "mail"
    call = payload["calls"][0]
    assert call["name"] == "note_entity"
    assert call["decision"] == "allow"
    assert call["matched_rule"] == "builtin:internal"
    # the provenance row's seq, so the trace points into the audit log
    assert payload["audit_seq"] == kit.audit.entries[-1]["seq"]


def _sched(action_id: int, tool: str, params: dict) -> ScheduledAction:
    when = datetime.now(UTC) + timedelta(hours=1)
    return ScheduledAction(
        id=action_id,
        label="evening summary",
        run_at=when,
        status="pending",
        payload={"type": "tool", "tool": tool, "params": params},
        created_at=when,
    )


async def test_a_scheduled_fire_is_traced() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__list_messages", result="all caught up")

    content = await kit.loop.execute_scheduled(_sched(7, "mail__list_messages", {"limit": 1}))

    assert content == "all caught up"
    created = kit.traces.created[0]
    assert created["kind"] == "scheduled"
    assert created["label"] == "evening summary"
    payload = created["payload"]
    assert payload["action"]["id"] == 7
    assert payload["action"]["label"] == "evening summary"
    assert payload["result"] == "all caught up"
    assert payload["is_error"] is False
    assert payload["calls"][0]["name"] == "mail__list_messages"


async def test_a_failed_scheduled_fire_is_traced_before_the_raise() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__send_message")
    kit.loop._policy = Policy([AuthzRule(tool_pattern="mail__send_message", decision="deny")])

    with pytest.raises(RuntimeError):
        await kit.loop.execute_scheduled(_sched(9, "mail__send_message", {}))

    created = kit.traces.created[0]  # written before the worker sees the raise
    assert created["kind"] == "scheduled"
    payload = created["payload"]
    assert payload["is_error"] is True
    assert payload["calls"][0]["decision"] == "deny"


async def test_a_carried_out_approval_is_traced() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__send_message", result="sent")
    await kit.loop._execute(ToolCall(id="t1", name="mail__send_message", arguments={"to": "a@b.c"}))
    approval = kit.approvals.created[0]
    approval.status = APPROVED
    approval.decided_by = "user"

    await kit.loop.execute_decision(approval.id, APPROVED)

    created = kit.traces.created[0]
    assert created["kind"] == "carry_out"
    assert created["label"] == "approval #1 — mail__send_message"
    payload = created["payload"]
    assert payload["approval_id"] == 1
    assert payload["tool"] == "mail__send_message"
    assert payload["params"] == {"to": "a@b.c"}
    assert payload["decided_by"] == "user"
    assert payload["result"] == "sent"
    assert payload["is_error"] is False


async def test_a_failed_carry_out_is_traced_as_an_error() -> None:
    kit = LoopKit(None)

    async def boom(params: dict) -> str:
        raise ValueError("mail down")

    kit.tools.add_native(ToolSpec(name="mail__send_message", description="test"), boom)
    await kit.loop._execute(ToolCall(id="t1", name="mail__send_message", arguments={"to": "a@b.c"}))
    approval = kit.approvals.created[0]
    approval.status = APPROVED

    await kit.loop.execute_decision(approval.id, APPROVED)

    payload = kit.traces.created[0]["payload"]
    assert payload["is_error"] is True
    assert "mail down" in payload["result"]


# ---------------------------------------------------------------------------
# unit: a trace write must never break the act it records
# ---------------------------------------------------------------------------


async def test_a_poisoned_trace_store_never_breaks_the_act() -> None:
    provider = _scripted_turn()
    kit = LoopKit(provider)
    kit.add_tool("mail__list_messages", result="2 unread: alice, bob")

    async def poison(*, kind: str, label: str = "", payload: dict) -> None:
        raise RuntimeError("trace db down")

    kit.traces.create = poison  # type: ignore[method-assign]
    kit.loop.submit_message(_msg("what's unread?"))
    await kit.loop._tick()

    assert kit.executed == [("mail__list_messages", {"limit": 2})]
    assert kit.connector.sent == ["2 unread"]  # the act stands, trace or no trace


# ---------------------------------------------------------------------------
# unit: the explain_decision tool
# ---------------------------------------------------------------------------


async def test_eight_native_tools_register_when_traces_are_wired() -> None:
    kit = NativeKit(traces=FakeTraces())
    assert kit.count == 8
    assert len(kit.registry) == 8
    both = NativeKit(routines=FakeRoutines(), traces=FakeTraces())
    assert both.count == 12
    assert len(both.registry) == 12


def _seed_turn(kit: NativeKit, *, approval_id: int | None = None) -> None:
    kit.traces.add(
        kind="turn",
        label="unread check",
        payload={
            "trigger": {
                "messages": [
                    {"surface": "telegram", "handle": "@vedant", "text": "what's unread?"}
                ],
                "observations": [],
            },
            "calls": [
                {
                    "id": "t1",
                    "name": "mail__list_messages",
                    "params": {"limit": 2},
                    "decision": "allow" if approval_id is None else "require_approval",
                    "matched_rule": "builtin:read-only" if approval_id is None else "builtin:risky",
                    "reason": "",
                    "audit_seq": 2 if approval_id is None else None,
                    "approval_id": approval_id,
                    "result": "2 unread: alice, bob",
                    "is_error": False,
                }
            ],
            "reply": "2 unread",
        },
    )


async def test_explain_by_trace_id_replays_the_record() -> None:
    kit = NativeKit(traces=FakeTraces())
    _seed_turn(kit)

    out = await kit.run("explain_decision", {"trace_id": 1})

    assert "Trace #1 (turn) — unread check" in out
    assert "asked: @vedant via telegram: what's unread?" in out
    assert "→ mail__list_messages" in out
    assert "gate: allow (builtin:read-only) [audit #2]" in out
    assert "reply: 2 unread" in out


async def test_explain_reports_unknown_ids_and_an_empty_store() -> None:
    kit = NativeKit(traces=FakeTraces())
    assert await kit.run("explain_decision", {}) == "No decision traces recorded yet."
    assert "No recorded trace 9." in await kit.run("explain_decision", {"trace_id": 9})


async def test_explain_by_approval_id_finds_the_origin_and_the_carry_out() -> None:
    kit = NativeKit(traces=FakeTraces())
    _seed_turn(kit, approval_id=5)  # the act that proposed the held call
    kit.traces.add(
        kind="carry_out",
        label="approval #5 — mail__send_message",
        payload={
            "approval_id": 5,
            "tool": "mail__send_message",
            "params": {},
            "decided_by": "user",
            "result": "sent",
            "is_error": False,
        },
    )

    out = await kit.run("explain_decision", {"approval_id": 5})

    assert out.index("Trace #1 (turn)") < out.index("Trace #2 (carry_out)")
    assert "held as approval #5" in out
    assert "decided by user" in out
    assert "No recorded trace involving approval #9." in await kit.run(
        "explain_decision", {"approval_id": 9}
    )


async def test_explain_by_topic_searches_recent_traces() -> None:
    kit = NativeKit(traces=FakeTraces())
    kit.traces.add(kind="routine", label="billing", payload={"routine": {"label": "billing"}})
    kit.traces.add(kind="turn", label="unread check", payload={"reply": "2 unread"})

    out = await kit.run("explain_decision", {"about": "billing"})

    assert "Trace #1 (routine) — billing" in out
    assert "Trace #2" not in out  # limit defaults to 3, but only matches show
    assert "No recorded trace mentioning 'zzz'." in await kit.run(
        "explain_decision", {"about": "zzz"}
    )


async def test_explain_with_no_arguments_lists_recent_traces() -> None:
    kit = NativeKit(traces=FakeTraces())
    for i in range(1, 4):
        kit.traces.add(kind="turn", label=f"turn {i}", payload={})

    listed = await kit.run("explain_decision", {})
    assert "Trace #3 (turn) — turn 3" in listed  # newest first
    assert "Trace #2 (turn) — turn 2" in listed
    assert "Trace #1 (turn) — turn 1" in listed

    two = await kit.run("explain_decision", {"limit": 2})
    assert "Trace #3" in two
    assert "Trace #1" not in two  # the limit is respected

    non_numeric = await kit.run("explain_decision", {"limit": "many"})
    assert "Trace #3" in non_numeric
    assert "Trace #1" in non_numeric  # a bad limit falls back to the default of 3, not an error


# ---------------------------------------------------------------------------
# integration: the real store
# ---------------------------------------------------------------------------


def _make_traces(db) -> Traces:
    return Traces(db, Cipher.from_b64(generate_key_b64()))


@pytest.mark.integration
async def test_traces_persist_encrypted_and_roundtrip(db) -> None:
    store = _make_traces(db)
    trace = await store.create(
        kind="turn", label="unread check", payload={"reply": "2 unread", "calls": []}
    )
    assert trace.id > 0

    # encrypted at rest: the blob is not the plaintext it stands for
    raw = await db.fetchval("SELECT trace_enc FROM decision_traces WHERE id = $1", trace.id)
    assert b"2 unread" not in raw

    # a fresh store instance over the same key reads it back — restart
    # survival for the "why" (a restart re-reads the same env key)
    fresh = Traces(db, store._cipher)
    back = await fresh.get(trace.id)
    assert back is not None
    assert back.kind == "turn"
    assert back.label == "unread check"
    assert back.payload == {"reply": "2 unread", "calls": []}


@pytest.mark.integration
async def test_recent_lists_newest_first(db) -> None:
    store = _make_traces(db)
    first = await store.create(kind="turn", label="one", payload={})
    second = await store.create(kind="routine", label="two", payload={})

    rows = await store.recent(limit=10)

    assert [t.id for t in rows] == [second.id, first.id]
    missing = await store.get(first.id + 100)
    assert missing is None
