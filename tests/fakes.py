"""In-memory doubles for unit tests — no network, no database, no SDK calls."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aether.agent.traces import Trace
from aether.authz.approvals import PENDING, Approval
from aether.llm.types import Message, ToolCall, ToolResult, ToolSpec, Turn
from aether.memory.context import AgentContext
from aether.memory.entities import Entity, PersonRef
from aether.memory.events import Event, IngestResult
from aether.routines import Routine
from aether.scheduler.jobs import PENDING as JOB_PENDING
from aether.scheduler.jobs import ScheduledAction


class FakeProvider:
    """Scripted provider: returns turns[i] on the i-th complete() call.

    Records every call for assertions. A complete() with tools=[] (the
    loop's forced wrap-up) is treated like any other call, so a script must
    account for it.
    """

    name = "fake"
    supports_tools = True
    supports_vision = True

    def __init__(self, turns: list[Turn], model: str = "fake-model") -> None:
        self.turns = list(turns)
        self.model = model
        self.calls: list[tuple[str, list[Message], list[ToolSpec]]] = []

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> Turn:
        self.calls.append((system, list(messages), list(tools)))
        if not self.turns:
            raise AssertionError("FakeProvider ran out of scripted turns")
        return self.turns.pop(0)


class FakeExecutor:
    """Callable tool executor returning canned results per tool name."""

    def __init__(self, results: dict[str, str] | None = None, default: str = "ok") -> None:
        self.results = results or {}
        self.default = default
        self.calls: list[ToolCall] = []

    async def __call__(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=self.results.get(call.name, self.default),
        )


class CrashingExecutor:
    """Executor that always raises — the loop must convert this into an
    error ToolResult instead of crashing."""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    async def __call__(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        raise RuntimeError("executor exploded")


class FakeJudge:
    """Scripted salience judge: returns judgments[i] on the i-th call."""

    def __init__(self, judgments: list[object]) -> None:
        self.judgments = list(judgments)
        self.calls: list[str] = []

    async def judge(self, text: str):
        self.calls.append(text)
        if not self.judgments:
            raise AssertionError("FakeJudge ran out of scripted judgments")
        item = self.judgments.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakePersonJudge:
    """Scripted same-person judge: returns confidences[i] on the i-th call."""

    def __init__(self, confidences: list[float]) -> None:
        self.confidences = list(confidences)
        self.calls: list[tuple[PersonRef, PersonRef]] = []

    async def confirm(self, a: PersonRef, b: PersonRef) -> float:
        self.calls.append((a, b))
        if not self.confidences:
            raise AssertionError("FakePersonJudge ran out of scripted confidences")
        return self.confidences.pop(0)


class FakeRegistry:
    """ProviderRegistry double: always answers for_role() with one provider."""

    def __init__(self, provider: object | None) -> None:
        self._provider = provider

    def for_role(self, role: str):
        return self._provider


class FakeApprovals:
    """Approvals double: records decide() calls, returns a canned result.

    Also implements the rest of the Approvals surface (create/get/
    mark_executed/mark_failed/list_pending) so agent-loop tests can run the
    full park-then-decide flow without a database."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []
        self.created: list[Approval] = []
        self.executed: list[int] = []
        self.failed: list[int] = []
        self.ttl_updates: list[float] = []
        self.result: object = object()  # what decide() returns (set to None to simulate stale)
        self._now = datetime.now(UTC)

    def set_ttl(self, hours: float) -> None:
        self.ttl_updates.append(hours)

    async def create(
        self,
        *,
        tool_name: str,
        params: dict,
        actor: str = "agent",
        rules_matched: str = "",
        note: str = "",
    ) -> Approval:
        approval = Approval(
            id=len(self.created) + 1,
            tool_name=tool_name,
            params=dict(params),
            status=PENDING,
            created_at=self._now,
            expires_at=self._now + timedelta(hours=24),
            decided_at=None,
            decided_by=None,
            rules_matched=rules_matched,
            note=note,
        )
        self.created.append(approval)
        return approval

    async def decide(self, approval_id: int, decision: str, decided_by: str = "user"):
        self.calls.append((approval_id, decision))
        if self.result is None:
            return None
        if isinstance(self.result, Approval):
            return self.result  # the caller scripted the decided row
        # realistic default: a row created via create() is only decided when
        # it's genuinely still pending and unexpired — mirrors the real
        # store's WHERE clause closely enough that get() reflects the
        # outcome afterward (tests exercising stale/expired feedback rely
        # on this), and the row is mutated in place like the real UPDATE
        for i, approval in enumerate(self.created):
            if approval.id != approval_id:
                continue
            if approval.status != PENDING or approval.expires_at <= datetime.now(UTC):
                return None
            updated = Approval(
                id=approval.id,
                tool_name=approval.tool_name,
                params=approval.params,
                status=decision,
                created_at=approval.created_at,
                expires_at=approval.expires_at,
                decided_at=datetime.now(UTC),
                decided_by=decided_by,
                rules_matched=approval.rules_matched,
                note=approval.note,
            )
            self.created[i] = updated
            return updated
        # no matching created row — the id was scripted directly, not via
        # create(); keep the old synthetic-approval behavior so those tests
        # (which never call create()) are unaffected
        return Approval(
            id=approval_id,
            tool_name=f"tool-{approval_id}",
            params={},
            status=decision,
            created_at=self._now,
            expires_at=self._now + timedelta(hours=24),
            decided_at=self._now,
            decided_by=decided_by,
        )

    async def get(self, approval_id: int) -> Approval | None:
        for approval in self.created:
            if approval.id == approval_id:
                return approval
        return None

    async def mark_executed(self, approval_id: int) -> None:
        self.executed.append(approval_id)

    async def mark_failed(self, approval_id: int) -> None:
        self.failed.append(approval_id)

    async def list_pending(self) -> list[Approval]:
        # mirrors the store: pending rows past their TTL never read as
        # waiting, even before expire_overdue has flipped them
        now = datetime.now(UTC)
        return [a for a in self.created if a.status == PENDING and a.expires_at > now]

    async def expire_overdue(self) -> int:
        flipped = 0
        for a in self.created:
            if a.status == PENDING and a.expires_at <= datetime.now(UTC):
                a.status = "expired"
                flipped += 1
        return flipped


class FakeAudit:
    """AuditLog double: records appends; recent()/verify_chain() feed the
    panel. `verification` overrides what verify_chain() reports, so a test
    can stage a broken chain."""

    def __init__(self, verification: object | None = None) -> None:
        self.entries: list[dict] = []
        self.verification = verification

    async def append(
        self,
        *,
        actor: str,
        tool_name: str,
        decision: str,
        rules_matched: str = "",
        params: dict | None = None,
        outcome: str = "",
    ) -> int:
        self.entries.append(
            {
                "seq": len(self.entries) + 1,
                "actor": actor,
                "tool_name": tool_name,
                "decision": decision,
                "rules_matched": rules_matched,
                "params": dict(params or {}),
                "outcome": outcome,
                "entry_hash": f"fake-hash-{len(self.entries) + 1}",
                "created_at": datetime.now(UTC),
            }
        )
        return len(self.entries)

    async def recent(self, limit: int = 100) -> list[dict]:
        return list(reversed(self.entries))[:limit]  # newest first, matching the real AuditLog

    async def verify_chain(self) -> object:
        if self.verification is not None:
            return self.verification
        from aether.authz.audit import ChainVerification

        return ChainVerification(ok=True, entries=len(self.entries))


class FakeSalience:
    """Salience double: records score_event calls, returns a canned score."""

    def __init__(self, score: float = 5.0) -> None:
        self.score = score
        self.scored: list[int] = []

    async def score_event(self, event_id: int) -> float:
        self.scored.append(event_id)
        return self.score


class FakeContextBuilder:
    """ContextBuilder double: hands back a canned AgentContext, counts builds."""

    def __init__(self, context: AgentContext | None = None) -> None:
        self.context = context or AgentContext()
        self.builds = 0

    async def build(self) -> AgentContext:
        self.builds += 1
        return self.context


class FakeScheduler:
    """Scheduler double: records creates, hands back ScheduledActions."""

    def __init__(self) -> None:
        self.created: list[dict] = []

    async def create(self, *, label: str, run_at, payload: dict, actor: str = "agent"):
        self.created.append({"label": label, "run_at": run_at, "payload": dict(payload)})
        return ScheduledAction(
            id=len(self.created),
            label=label,
            run_at=run_at,
            status=JOB_PENDING,
            payload=dict(payload),
            created_at=datetime.now(UTC),
        )


class FakeRoutines:
    """Routines double: `add` arms rows the loop's matcher sees; create/
    set_enabled/delete/mark_fired record what the tools and the loop did."""

    def __init__(self) -> None:
        self.routines: list[Routine] = []
        self.created: list[dict] = []
        self.fired: list[int] = []
        self.toggles: list[tuple[int, bool]] = []
        self.deleted: list[int] = []

    def add(
        self,
        *,
        label: str = "routine",
        trigger: dict | None = None,
        action: dict | None = None,
        cooldown_seconds: int = 0,
        last_fired_at: datetime | None = None,
        enabled: bool = True,
    ) -> Routine:
        routine = Routine(
            id=len(self.routines) + 1,
            label=label,
            trigger=dict(trigger or {}),
            action=dict(action or {"type": "tool", "tool": "note_entity", "params": {}}),
            enabled=enabled,
            cooldown_seconds=cooldown_seconds,
            fire_count=0,
            last_fired_at=last_fired_at,
            created_at=datetime.now(UTC),
        )
        self.routines.append(routine)
        return routine

    async def create(
        self,
        *,
        label: str,
        trigger: dict,
        action: dict,
        cooldown_seconds: int = 300,
        actor: str = "agent",
    ) -> Routine:
        self.created.append(
            {
                "label": label,
                "trigger": dict(trigger),
                "action": dict(action),
                "cooldown_seconds": cooldown_seconds,
                "actor": actor,
            }
        )
        return self.add(
            label=label, trigger=trigger, action=action, cooldown_seconds=cooldown_seconds
        )

    async def get(self, routine_id: int) -> Routine | None:
        return next((r for r in self.routines if r.id == routine_id), None)

    async def list(self, limit: int = 50) -> list[Routine]:
        return list(reversed(self.routines))[:limit]

    async def list_enabled(self) -> list[Routine]:
        return [r for r in self.routines if r.enabled]

    async def mark_fired(self, routine_id: int) -> None:
        self.fired.append(routine_id)
        for r in self.routines:
            if r.id == routine_id:
                r.fire_count += 1
                r.last_fired_at = datetime.now(UTC)

    async def set_enabled(self, routine_id: int, enabled: bool) -> Routine | None:
        self.toggles.append((routine_id, enabled))
        routine = await self.get(routine_id)
        if routine is None:
            return None
        routine.enabled = enabled
        return routine

    async def delete(self, routine_id: int) -> bool:
        self.deleted.append(routine_id)
        routine = await self.get(routine_id)
        if routine is None:
            return False
        self.routines.remove(routine)
        return True


class FakeTraces:
    """Traces double: `add` pre-seeds rows the explain tool can find;
    create() records what the loop persisted."""

    def __init__(self) -> None:
        self.traces: list[Trace] = []
        self.created: list[dict] = []

    def add(self, *, kind: str = "turn", label: str = "", payload: dict | None = None) -> Trace:
        trace = Trace(
            id=len(self.traces) + 1,
            kind=kind,
            label=label,
            payload=dict(payload or {}),
            created_at=datetime.now(UTC),
        )
        self.traces.append(trace)
        return trace

    async def create(self, *, kind: str, label: str = "", payload: dict) -> Trace:
        self.created.append({"kind": kind, "label": label, "payload": dict(payload)})
        return self.add(kind=kind, label=label, payload=payload)

    async def get(self, trace_id: int) -> Trace | None:
        return next((t for t in self.traces if t.id == trace_id), None)

    async def recent(self, limit: int = 20) -> list[Trace]:
        return list(reversed(self.traces))[:limit]


class FakeTranscript:
    """ChatHistory double: append() records (surface, direction, text);
    recent() answers oldest-first, bounded to the newest `limit` rows —
    the shape the loop reads back as conversation memory."""

    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows: list[dict] = list(rows or [])
        self.appended: list[tuple[str, str, str]] = []

    async def append(self, surface: str, direction: str, text: str) -> None:
        self.appended.append((surface, direction, text))
        self.rows.append({"surface": surface, "direction": direction, "text": text, "at": ""})

    async def recent(self, limit: int = 50) -> list[dict]:
        return self.rows[-limit:]


class FakeEntities:
    """Entities double: resolve mints fresh identities, note() records."""

    def __init__(self) -> None:
        self.resolved: list[tuple[str, str]] = []
        self.notes: list[tuple[int, dict]] = []

    async def resolve(self, platform: str, handle: str, display_name: str = ""):
        self.resolved.append((platform, handle))
        return Entity(
            id=len(self.resolved),
            display_name=display_name or handle,
            confidence=1.0,
            handles=[(platform, handle)],
        )

    async def note(self, identity_id: int, payload: dict) -> None:
        self.notes.append((identity_id, dict(payload)))


class FakeSurfaceConnector:
    """MessagingConnector double: records sends, approval presentations, and
    start/stop cycles (the config manager's messaging diff stops and starts
    connectors live). send_to_user answers with a fresh platform message id,
    the way the real surfaces do — enough for the loop to link a later why?
    back."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.sent: list[str] = []
        self.approvals_presented: list[tuple[int, str]] = []
        self.starts = 0
        self.stops = 0
        self._ids = 0

    async def start(self) -> None:
        self.starts += 1

    async def stop(self) -> None:
        self.stops += 1

    async def send_to_user(self, text: str) -> str:
        self.sent.append(text)
        self._ids += 1
        return f"m{self._ids}"

    async def present_approval(self, approval_id: int, text: str) -> None:
        self.approvals_presented.append((approval_id, text))


class FakeSchedStore:
    """Scheduler-store double for worker tests: claims due, records marks."""

    def __init__(self, actions: list[ScheduledAction]) -> None:
        self.actions = actions
        self.marks: list[tuple[int, str]] = []

    async def overdue_pending(self) -> list[ScheduledAction]:
        now = datetime.now(UTC)
        return [a for a in self.actions if a.status == JOB_PENDING and a.run_at <= now]

    async def claim_due(self, limit: int = 20) -> list[ScheduledAction]:
        now = datetime.now(UTC)
        claimed = [a for a in self.actions if a.status == JOB_PENDING and a.run_at <= now][:limit]
        for action in claimed:
            action.status = "running"
        return claimed

    async def mark(self, action_id: int, status: str, result_digest: str = "") -> None:
        self.marks.append((action_id, status))
        for action in self.actions:
            if action.id == action_id:
                action.status = status


class FakeEventStore:
    """EventStore double for salience-pipeline tests: canned events, canned
    per-source recent counts, recorded salience updates."""

    def __init__(
        self,
        events: dict[int, Event] | None = None,
        recent_counts: dict[str, int] | None = None,
        search_results: list[Event] | None = None,
    ) -> None:
        self.events = dict(events or {})
        self.recent_counts = dict(recent_counts or {})
        self.search_results = list(search_results or [])
        self.updates: list[tuple[int, float, bool, dict]] = []
        self.ingested: list[dict] = []
        # set to override what ingest() returns (e.g. filtered:contacts)
        self.ingest_result: IngestResult | None = None

    async def ingest(self, *, source, kind, payload, sender=None, occurred_at=None, meta=None):
        self.ingested.append({"source": source, "kind": kind, "payload": payload, "sender": sender})
        if self.ingest_result is not None:
            return self.ingest_result
        return IngestResult(
            stored=True,
            reason="new",
            # distinct from pre-populated event ids so both can coexist
            event_id=max(self.events, default=0) + len(self.ingested),
        )

    async def get(self, event_id: int) -> Event | None:
        return self.events.get(event_id)

    async def list_since(self, after_id: int, limit: int = 50) -> list[Event]:
        return [self.events[i] for i in sorted(self.events) if i > after_id][:limit]

    async def max_id(self) -> int:
        return max(self.events, default=0)

    async def recent(self, limit: int = 50) -> list[Event]:
        return [self.events[i] for i in sorted(self.events, reverse=True)][:limit]

    async def search(self, query: str, limit: int = 10, scan_window: int = 200) -> list[Event]:
        return self.search_results[:limit]

    async def count_recent(self, source: str, hours: float = 1.0) -> int:
        return self.recent_counts.get(source, 0)

    async def update_salience(
        self,
        event_id: int,
        score: float,
        memorable: bool,
        category: str = "",
        actionability: str = "",
    ) -> None:
        self.updates.append(
            (event_id, score, memorable, {"category": category, "actionability": actionability})
        )


class FakeAgentSettings:
    """AgentSettings double: an in-memory personality string, no database."""

    def __init__(self, personality: str = "") -> None:
        self.personality = personality

    async def get_personality(self) -> str:
        return self.personality

    async def set_personality(self, text: str) -> str:
        self.personality = text.strip()
        return self.personality


class FakeConfigStore:
    """ConfigOverrides double: rows in a dict, the same four-call surface
    (load/upsert/delete/delete_prefixed) so ConfigManager tests run with no
    database."""

    def __init__(self, rows: dict[str, object] | None = None) -> None:
        self.rows = dict(rows or {})
        self.upserts: list[tuple[str, object]] = []
        self.deletes: list[str] = []

    async def load(self) -> dict[str, object]:
        return dict(self.rows)

    async def upsert(self, path: str, value: object) -> None:
        self.upserts.append((path, value))
        self.rows[path] = value

    async def delete(self, path: str) -> None:
        self.deletes.append(path)
        self.rows.pop(path, None)

    async def delete_prefixed(self, prefix: str) -> None:
        for path in [p for p in self.rows if p.startswith(prefix)]:
            self.rows.pop(path, None)


class FakeConfigManager:
    """ConfigManager double: records what the /config command and the native
    tools asked for, answers with canned replies. effective_payload is what
    GET /api/config serves; yaml_values is what the guided walk's reset
    question reads (path → the config.yaml value)."""

    def __init__(
        self,
        show_reply: str = "fake config view",
        set_reply: str = "fake set confirmation",
        effective_payload: dict | None = None,
        yaml_values: dict[str, object] | None = None,
    ) -> None:
        self.show_reply = show_reply
        self.set_reply = set_reply
        self.effective_payload = (
            effective_payload
            if effective_payload is not None
            else {
                "sections": {},
                "overrides": [],
            }
        )
        self.yaml_values = yaml_values or {}
        self.show_calls: list[str | None] = []
        self.set_calls: list[dict] = []

    async def show(self, path: str | None = None) -> str:
        self.show_calls.append(path)
        return self.show_reply

    def yaml_value(self, path: str) -> object:
        return self.yaml_values.get(path)

    async def set(self, *, op: str, path: str, value: object = None, source: str = "chat") -> str:
        self.set_calls.append({"op": op, "path": path, "value": value, "source": source})
        return self.set_reply

    async def effective(self) -> dict:
        return self.effective_payload


class FakeComposioBridge:
    """ComposioBridge double: scripted catalog, accounts, and authorize
    answers — no SDK, no network. `wait_and_enable` returns a fresh ACTIVE
    account (or raises the scripted error); `disconnect` removes only ids
    the fake holds, the way the real one verifies ownership."""

    def __init__(
        self,
        *,
        toolkits: list | None = None,
        accounts: list | None = None,
        available: bool = True,
        authorize_url: str = "https://hub.example.test/connect",
        wait_error: Exception | None = None,
    ) -> None:
        from aether.composio_bridge import ToolkitInfo

        self.toolkit_list = list(
            toolkits
            if toolkits is not None
            else [
                ToolkitInfo(
                    slug="gmail",
                    name="Gmail",
                    logo="https://logo.test/gmail.png",
                    description="your mail",
                ),
                ToolkitInfo(slug="github", name="GitHub", logo="", description="your code"),
            ]
        )
        self.account_list = list(accounts or [])
        self._available = available
        self._authorize_url = authorize_url
        self._wait_error = wait_error
        self.authorized: list[str] = []
        self.disconnected: list[str] = []
        self.resets = 0
        self._requests = 0

    async def available(self) -> bool:
        return self._available

    async def ensure(self) -> bool:
        return self._available

    async def reset_session(self) -> bool:
        self.resets += 1
        return self._available

    async def accounts(self) -> list:
        return list(self.account_list)

    async def toolkits(self) -> list:
        return list(self.toolkit_list)

    async def find_toolkits(self, query: str, limit: int = 5) -> list:
        needle = query.strip().lower()
        exact = [t for t in self.toolkit_list if needle in (t.slug.lower(), t.name.lower())]
        if exact:
            return exact[:1]
        return [
            t for t in self.toolkit_list if needle in t.slug.lower() or needle in t.name.lower()
        ][:limit]

    async def authorize(self, toolkit: str) -> tuple[str, str]:
        from aether.composio_bridge import ComposioNotConfigured

        if not self._available:
            raise ComposioNotConfigured("no key in the fake")
        self.authorized.append(toolkit)
        self._requests += 1
        return f"req-{self._requests}", f"{self._authorize_url}/{toolkit}"

    async def wait_and_enable(self, toolkit: str, request_id: str, timeout: float = 0):
        from aether.composio_bridge import ConnectedApp

        if self._wait_error is not None:
            raise self._wait_error
        app = ConnectedApp(
            id=f"acc-{toolkit}",
            toolkit=toolkit,
            status="ACTIVE",
            identity=f"{toolkit}@example.test",
        )
        self.account_list.append(app)
        return app

    async def disconnect(self, account_id: str):
        for app in self.account_list:
            if app.id == account_id:
                self.account_list.remove(app)
                self.disconnected.append(account_id)
                return app
        return None
