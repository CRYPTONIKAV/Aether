"""API route tests — the REST surface against a stub app.state."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
from fakes import (
    FakeAgentSettings,
    FakeApprovals,
    FakeAudit,
    FakeComposioBridge,
    FakeConfigManager,
    FakeEventStore,
    FakeTraces,
)
from fastapi import FastAPI

from aether.agent.loop import CaptureRequestBox
from aether.api.routes import router
from aether.authz.approvals import Approval
from aether.composio_bridge import ConnectedApp
from aether.config import AppConfig, AuthzConfig, AuthzRule, Settings
from aether.memory.entities import Entity, EntityNote
from aether.memory.events import Event, IngestResult
from aether.scheduler.jobs import ScheduledAction
from aether.workspace import Workspace


class FakeScreenVision:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str]] = []

    async def record(self, png: bytes, note: str = ""):
        self.calls.append((png, note))
        return IngestResult(stored=True, reason="new", event_id=1)


class FakeAgent:
    """The loop as the decide/notify surface needs it."""

    def __init__(self, decision_result: object = None) -> None:
        self.decision_result = decision_result
        self.calls: list[tuple[int, str]] = []
        self.notified = 0
        self.watched: list[tuple[str, str]] = []

    async def decide(self, approval_id: int, decision: str):
        self.calls.append((approval_id, decision))
        if isinstance(self.decision_result, Exception):
            raise self.decision_result
        return self.decision_result

    def notify(self) -> None:
        self.notified += 1

    def watch_connect(self, toolkit: str, request_id: str) -> None:
        self.watched.append((toolkit, request_id))


def _approval(approval_id: int = 1, status: str = "approved") -> Approval:
    now = datetime.now(UTC)
    return Approval(
        id=approval_id,
        tool_name="mail__send_message",
        params={"to": "a@b.c"},
        status=status,
        created_at=now,
        expires_at=now + timedelta(hours=24),
        decided_at=None,
        decided_by=None,
    )


def _event(event_id: int, kind: str = "chat_message") -> Event:
    return Event(
        id=event_id,
        source="telegram",
        kind=kind,
        occurred_at=datetime(2026, 9, 25, 10, event_id, tzinfo=UTC),
        payload={"text": f"number {event_id}"},
        salience_score=5.0,
        memorable=event_id == 2,
        meta={},
    )


def _app(
    vision=None,
    *,
    approvals: FakeApprovals | None = None,
    audit: FakeAudit | None = None,
    events: FakeEventStore | None = None,
    capture_box: CaptureRequestBox | None = None,
    agent: FakeAgent | None = None,
    config: AppConfig | None = None,
    agent_settings: FakeAgentSettings | None = None,
    composio: FakeComposioBridge | None = None,
    config_manager: FakeConfigManager | None = None,
    entities=None,
    scheduler=None,
    workspace=None,
    settings: Settings | None = None,
    traces: FakeTraces | None = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.settings = settings or Settings(_env_file=None, api_token="secret")
    app.state.config = config or AppConfig()
    if entities is not None:
        app.state.entities = entities
    if scheduler is not None:
        app.state.scheduler = scheduler
    if vision is not None:
        app.state.screen_vision = vision
    if approvals is not None:
        app.state.approvals = approvals
    if audit is not None:
        app.state.audit = audit
    if events is not None:
        app.state.events = events
    if capture_box is not None:
        app.state.capture_box = capture_box
    if agent is not None:
        app.state.agent = agent
    if agent_settings is not None:
        app.state.agent_settings = agent_settings
    if composio is not None:
        app.state.composio = composio
    if config_manager is not None:
        app.state.config_manager = config_manager
    if workspace is not None:
        app.state.workspace = workspace
    if traces is not None:
        app.state.traces = traces
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_screen_capture_rejects_a_bad_token() -> None:
    vision = FakeScreenVision()
    async with _client(_app(vision)) as client:
        response = await client.post(
            "/api/screen-capture",
            params={"token": "wrong"},
            files={"file": ("s.png", b"png", "image/png")},
            data={"note": ""},
        )
    assert response.status_code == 401
    assert vision.calls == []


async def test_screen_capture_accepts_the_upload() -> None:
    vision = FakeScreenVision()
    async with _client(_app(vision)) as client:
        response = await client.post(
            "/api/screen-capture",
            params={"token": "secret"},
            files={"file": ("s.png", b"png-bytes", "image/png")},
            data={"note": "keep an eye on this"},
        )
    assert response.status_code == 202
    assert response.json() == {"stored": True, "reason": "new", "event_id": 1}
    assert vision.calls == [(b"png-bytes", "keep an eye on this")]


async def test_screen_capture_token_via_header() -> None:
    vision = FakeScreenVision()
    async with _client(_app(vision)) as client:
        response = await client.post(
            "/api/screen-capture",
            headers={"x-aether-token": "secret"},
            files={"file": ("s.png", b"png", "image/png")},
        )
    assert response.status_code == 202


async def test_screen_capture_without_vision_configured_is_503() -> None:
    async with _client(_app(None)) as client:  # no screen_vision on state
        response = await client.post(
            "/api/screen-capture",
            params={"token": "secret"},
            files={"file": ("s.png", b"png", "image/png")},
        )
    assert response.status_code == 503


async def test_screen_capture_rejects_an_empty_upload() -> None:
    vision = FakeScreenVision()
    async with _client(_app(vision)) as client:
        response = await client.post(
            "/api/screen-capture",
            params={"token": "secret"},
            files={"file": ("s.png", b"", "image/png")},
        )
    assert response.status_code == 400


async def test_screen_capture_wakes_the_agent() -> None:
    vision = FakeScreenVision()
    agent = FakeAgent()
    async with _client(_app(vision, agent=agent)) as client:
        response = await client.post(
            "/api/screen-capture",
            params={"token": "secret"},
            files={"file": ("s.png", b"png-bytes", "image/png")},
        )
    assert response.status_code == 202
    assert agent.notified == 1


# ---------------------------------------------------------------------------
# /api/screen-request — the companion's poll
# ---------------------------------------------------------------------------


async def test_screen_request_hands_over_one_capture() -> None:
    box = CaptureRequestBox()
    box.set("check the deploy log")
    async with _client(_app(None, capture_box=box)) as client:
        first = await client.get("/api/screen-request", params={"token": "secret"})
        second = await client.get("/api/screen-request", params={"token": "secret"})
    assert first.status_code == 200
    assert first.json() == {"note": "check the deploy log"}
    assert second.status_code == 204  # consumed — one capture per ask


async def test_screen_request_guards_and_degrades() -> None:
    box = CaptureRequestBox()
    async with _client(_app(None, capture_box=box)) as client:
        assert (await client.get("/api/screen-request")).status_code == 401
    async with _client(_app(None)) as client:  # no box wired
        response = await client.get("/api/screen-request", params={"token": "secret"})
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# /api/approvals + decide — the web panel's buttons
# ---------------------------------------------------------------------------


async def test_approvals_feed_lists_pending_rows() -> None:
    approvals = FakeApprovals()
    async with _client(_app(None, approvals=approvals)) as client:
        assert (await client.get("/api/approvals")).status_code == 401
        empty = await client.get("/api/approvals", params={"token": "secret"})
        assert empty.json() == {"pending": []}

        await approvals.create(
            tool_name="mail__send_message",
            params={"to": "a@b.c"},
            rules_matched="builtin:risky",
            note="reaches an external system or is hard to undo",
        )
        filled = await client.get("/api/approvals", params={"token": "secret"})
    pending = filled.json()["pending"]
    assert len(pending) == 1
    assert pending[0]["tool_name"] == "mail__send_message"
    assert pending[0]["params"] == {"to": "a@b.c"}
    # why the gate parked it — the card's plain-language reason
    assert pending[0]["rules_matched"] == "builtin:risky"
    assert pending[0]["note"] == "reaches an external system or is hard to undo"
    assert pending[0]["created_at"].endswith("+00:00")  # ISO datetimes


async def test_decide_runs_a_fresh_approval() -> None:
    agent = FakeAgent(decision_result=_approval(3, status="approved"))
    async with _client(_app(None, agent=agent)) as client:
        response = await client.post(
            "/api/approvals/3/decide",
            params={"token": "secret"},
            json={"decision": "approve"},
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True, "status": "approved"}
    assert agent.calls == [(3, "approve")]


async def test_decide_reports_a_stale_one_without_running_it() -> None:
    agent = FakeAgent(decision_result=None)  # already decided or expired
    async with _client(_app(None, agent=agent)) as client:
        response = await client.post(
            "/api/approvals/9/decide",
            params={"token": "secret"},
            json={"decision": "deny"},
        )
    assert response.status_code == 200
    assert response.json() == {"ok": False, "reason": "already decided or expired"}


async def test_decide_rejects_a_bad_decision_and_a_missing_agent() -> None:
    agent = FakeAgent(decision_result=ValueError("decision must be 'approve' or 'deny'"))
    async with _client(_app(None, agent=agent)) as client:
        response = await client.post(
            "/api/approvals/1/decide",
            params={"token": "secret"},
            json={"decision": "maybe"},
        )
    assert response.status_code == 400

    async with _client(_app(None)) as client:  # no agent on state
        response = await client.post(
            "/api/approvals/1/decide",
            params={"token": "secret"},
            json={"decision": "approve"},
        )
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# /api/events + /api/audit — the feeds
# ---------------------------------------------------------------------------


async def test_events_feed_is_newest_first() -> None:
    store = FakeEventStore({1: _event(1), 2: _event(2), 3: _event(3, "poll:unread")})
    async with _client(_app(None, events=store)) as client:
        assert (await client.get("/api/events")).status_code == 401
        response = await client.get("/api/events", params={"token": "secret"})
    feed = response.json()["events"]
    assert [e["id"] for e in feed] == [3, 2, 1]
    assert feed[1]["memorable"] is True  # only event 2 was memorable
    assert feed[0]["kind"] == "poll:unread"
    assert feed[2]["payload"] == {"text": "number 1"}


async def test_audit_feed_carries_the_chain_verdict() -> None:
    audit = FakeAudit()
    await audit.append(
        actor="agent",
        tool_name="mail__list_messages",
        decision="allow",
        rules_matched="builtin:read-only",
        outcome="read-only tool",
    )
    async with _client(_app(None, audit=audit)) as client:
        assert (await client.get("/api/audit")).status_code == 401
        response = await client.get("/api/audit", params={"token": "secret"})
    data = response.json()
    assert data["chain"] == {
        "ok": True,
        "entries": 1,
        "first_bad_seq": None,
        "problem": None,
        "head_hash": "fake-hash-1",
    }
    entry = data["entries"][0]
    assert entry["seq"] == 1
    assert entry["hash"] == "fake-hash-1"
    assert entry["tool"] == "mail__list_messages"
    assert entry["decision"] == "allow"
    assert entry["at"].endswith("+00:00")


# ---------------------------------------------------------------------------
# /api/policy — the read-only view of config.yaml's authz rules
# ---------------------------------------------------------------------------


async def test_policy_feed_is_token_guarded() -> None:
    async with _client(_app(None)) as client:
        response = await client.get("/api/policy")
    assert response.status_code == 401


async def test_policy_feed_reports_configured_rules_and_the_builtin_ladder() -> None:
    config = AppConfig(
        authz=AuthzConfig(
            rules=[
                AuthzRule(
                    tool_pattern="telegram__send_message",
                    decision="approve",
                    note="telegram sends are one-tap",
                ),
            ],
            approval_ttl_hours=12,
        )
    )
    async with _client(_app(None, config=config)) as client:
        response = await client.get("/api/policy", params={"token": "secret"})
    data = response.json()
    assert data["approval_ttl_hours"] == 12
    assert data["rules"] == [
        {
            "tool_pattern": "telegram__send_message",
            "param_pattern": None,
            "decision": "approve",
            "note": "telegram sends are one-tap",
        }
    ]
    assert len(data["builtin_rules"]) == 6
    by_id = {r["id"]: r for r in data["builtin_rules"]}
    assert set(by_id) == {
        "builtin:config-tune",
        "builtin:config-security",
        "builtin:internal",
        "builtin:risky",
        "builtin:read-only",
        "default:fail-safe",
    }
    # the chat-configuration gate, as the panel explains it
    assert by_id["builtin:config-tune"]["decision"] == "allow"
    assert by_id["builtin:config-security"]["decision"] == "require_approval"


# ---------------------------------------------------------------------------
# /api/config — the read-only live config + chat-made overrides
# ---------------------------------------------------------------------------


async def test_config_feed_is_token_guarded() -> None:
    async with _client(_app(None, config_manager=FakeConfigManager())) as client:
        assert (await client.get("/api/config")).status_code == 401


async def test_config_feed_is_503_without_a_manager_wired() -> None:
    async with _client(_app(None)) as client:  # no config_manager on state
        response = await client.get("/api/config", params={"token": "secret"})
    assert response.status_code == 503


async def test_config_feed_serves_the_effective_shape() -> None:
    payload = {
        "sections": {"agent": {"tick_seconds": 10.0}},
        "overrides": [{"path": "agent.tick_seconds", "value": 10.0}],
    }
    manager = FakeConfigManager(effective_payload=payload)
    async with _client(_app(None, config_manager=manager)) as client:
        response = await client.get("/api/config", params={"token": "secret"})
    assert response.status_code == 200
    assert response.json() == payload


# ---------------------------------------------------------------------------
# /api/settings/personality — the owner's custom behavior instructions
# ---------------------------------------------------------------------------


async def test_personality_feed_is_token_guarded() -> None:
    async with _client(_app(None, agent_settings=FakeAgentSettings())) as client:
        assert (await client.get("/api/settings/personality")).status_code == 401
        assert (
            await client.put("/api/settings/personality", json={"text": "x"})
        ).status_code == 401


async def test_personality_feed_is_503_without_a_store_wired() -> None:
    async with _client(_app(None)) as client:  # no agent_settings on state
        response = await client.get("/api/settings/personality", params={"token": "secret"})
    assert response.status_code == 503


async def test_personality_round_trips_through_get_and_put() -> None:
    store = FakeAgentSettings()
    async with _client(_app(None, agent_settings=store)) as client:
        empty = await client.get("/api/settings/personality", params={"token": "secret"})
        assert empty.json() == {"text": ""}

        saved = await client.put(
            "/api/settings/personality",
            params={"token": "secret"},
            json={"text": "  Be warm and a little informal.  "},
        )
        assert saved.json() == {"ok": True, "text": "Be warm and a little informal."}

        fetched = await client.get("/api/settings/personality", params={"token": "secret"})
    assert fetched.json() == {"text": "Be warm and a little informal."}
    assert store.personality == "Be warm and a little informal."


# ---------------------------------------------------------------------------
# /api/settings/apps — real configured connectors + capabilities
# ---------------------------------------------------------------------------


async def test_apps_feed_is_token_guarded() -> None:
    async with _client(_app(None)) as client:
        assert (await client.get("/api/settings/apps")).status_code == 401


async def test_apps_feed_reports_configured_connectors_and_capabilities() -> None:
    from aether.config import (
        AppConfig,
        ContactsConfig,
        MCPServerConfig,
        MessagingConfig,
        PlatformToggle,
        TransportConfig,
    )

    config = AppConfig(
        messaging=MessagingConfig(
            telegram=PlatformToggle(enabled=True),
            discord=PlatformToggle(enabled=False),
            slack=PlatformToggle(enabled=False),
        ),
        mcp_servers=[
            MCPServerConfig(name="mail", transport=TransportConfig(type="http", url="https://x")),
        ],
        contacts=ContactsConfig(mode="enforce"),
    )
    async with _client(_app(config=config, vision=FakeScreenVision())) as client:
        response = await client.get("/api/settings/apps", params={"token": "secret"})
    data = response.json()
    # no resolver wired → no keys known → has_token is honestly False
    assert {"name": "telegram", "enabled": True, "has_token": False} in data["connectors"]
    assert {"name": "discord", "enabled": False, "has_token": False} in data["connectors"]
    # no tools registry wired → not connected, zero actions
    assert data["mcp_servers"] == [
        {"name": "mail", "transport": "http", "enabled": True, "connected": False, "actions": 0}
    ]
    assert data["screen_vision_available"] is True
    assert data["contacts_mode"] == "enforce"
    # the catalog rides along so the panel never hardcodes the chat-surface
    # list — hub apps are /api/connections' job, fetched live
    assert {c["key"] for c in data["catalog"]} == {"telegram", "discord", "slack"}
    assert all(set(c) == {"key", "name", "blurb", "kind"} for c in data["catalog"])


async def test_apps_feed_marks_connectors_with_keys_and_live_servers() -> None:
    from aether.config import (
        AppConfig,
        MCPServerConfig,
        MessagingConfig,
        PlatformToggle,
        TransportConfig,
    )

    class _Resolver:
        async def known_names(self):
            return {"TELEGRAM_BOT_TOKEN"}

    class _Tools:
        def mcp_status(self):
            return {"mail": True}

        def server_action_count(self, name):
            return 7 if name == "mail" else 0

    config = AppConfig(
        messaging=MessagingConfig(telegram=PlatformToggle(enabled=True)),
        mcp_servers=[
            MCPServerConfig(name="mail", transport=TransportConfig(type="http", url="https://x")),
        ],
    )
    app = _app(config=config)
    app.state.resolver = _Resolver()
    app.state.tools = _Tools()
    async with _client(app) as client:
        response = await client.get("/api/settings/apps", params={"token": "secret"})
    data = response.json()
    assert {"name": "telegram", "enabled": True, "has_token": True} in data["connectors"]
    assert data["mcp_servers"] == [
        {"name": "mail", "transport": "http", "enabled": True, "connected": True, "actions": 7}
    ]


async def test_apps_feed_reports_screen_vision_unavailable_without_a_provider() -> None:
    async with _client(_app(None)) as client:  # no screen_vision on state
        response = await client.get("/api/settings/apps", params={"token": "secret"})
    assert response.json()["screen_vision_available"] is False


# ---------------------------------------------------------------------------
# /api/traces — decision traces for the web panel
# ---------------------------------------------------------------------------


async def test_traces_feed_is_token_guarded() -> None:
    async with _client(_app(None, traces=FakeTraces())) as client:
        assert (await client.get("/api/traces")).status_code == 401
        assert (await client.get("/api/traces/1")).status_code == 401


async def test_traces_feed_is_503_when_unwired() -> None:
    async with _client(_app(None)) as client:  # no traces on state
        response = await client.get("/api/traces", params={"token": "secret"})
        assert response.status_code == 503
        response = await client.get("/api/traces/1", params={"token": "secret"})
        assert response.status_code == 503


async def test_traces_feed_lists_summaries_newest_first() -> None:
    traces = FakeTraces()
    traces.add(kind="turn", label="telegram · vedant", payload={"calls": []})
    traces.add(kind="carry_out", label="mail__send_message", payload={"result": "sent"})
    async with _client(_app(None, traces=traces)) as client:
        response = await client.get("/api/traces", params={"token": "secret"})
    rows = response.json()["traces"]
    assert [r["id"] for r in rows] == [2, 1]  # newest first
    assert rows[0]["kind"] == "carry_out"
    assert rows[0]["label"] == "mail__send_message"
    assert rows[0]["at"].endswith("+00:00")
    assert "payload" not in rows[0]  # summaries only — the detail endpoint decrypts


async def test_trace_detail_serves_the_full_payload() -> None:
    traces = FakeTraces()
    trace = traces.add(
        kind="turn",
        label="telegram · vedant",
        payload={"trigger": {"messages": []}, "calls": [], "reply": "on it"},
    )
    async with _client(_app(None, traces=traces)) as client:
        response = await client.get(f"/api/traces/{trace.id}", params={"token": "secret"})
    data = response.json()
    assert data["id"] == trace.id
    assert data["kind"] == "turn"
    assert data["payload"]["reply"] == "on it"


async def test_trace_detail_is_404_for_an_unknown_trace() -> None:
    async with _client(_app(None, traces=FakeTraces())) as client:
        response = await client.get("/api/traces/99", params={"token": "secret"})
    assert response.status_code == 404


async def test_trace_detail_backfills_plain_for_legacy_calls() -> None:
    traces = FakeTraces()
    trace = traces.add(
        kind="turn",
        label="telegram · vedant",
        payload={
            "calls": [
                {"name": "mail__list_messages", "params": {}, "decision": "allow"},
                {"name": "mail__send_message", "params": {"to": "a@b.c"}, "plain": "Emailing Sam"},
            ]
        },
    )
    async with _client(_app(None, traces=traces)) as client:
        response = await client.get(f"/api/traces/{trace.id}", params={"token": "secret"})
    calls = response.json()["payload"]["calls"]
    assert calls[0]["plain"] == "Checking mail"  # backfilled from the backend vocabulary
    assert calls[1]["plain"] == "Emailing Sam"  # the recorded sentence wins
    assert "plain" not in traces.traces[0].payload["calls"][0]  # the store is never rewritten


async def test_events_feed_backfills_plain_for_legacy_agent_rows() -> None:
    store = FakeEventStore(
        {
            1: Event(
                id=1,
                source="agent",
                kind="action_done",
                occurred_at=datetime(2026, 9, 25, 10, 1, tzinfo=UTC),
                payload={
                    "tool": "composio__GMAIL_WHO_AM_I",
                    "params": {},
                    "detail": "ok",
                    "approval_id": 1,
                },
                salience_score=5.0,
                memorable=False,
                meta={},
            ),
            2: Event(
                id=2,
                source="agent",
                kind="action_done",
                occurred_at=datetime(2026, 9, 25, 10, 2, tzinfo=UTC),
                payload={
                    "tool": "mail__send_message",
                    "params": {"to": "a@b.c"},
                    "plain": "Emailing Sam",
                    "detail": "sent",
                    "approval_id": 2,
                },
                salience_score=5.0,
                memorable=False,
                meta={},
            ),
        }
    )
    async with _client(_app(None, events=store)) as client:
        response = await client.get("/api/events", params={"token": "secret"})
    by_id = {e["id"]: e for e in response.json()["events"]}
    assert by_id[1]["payload"]["plain"] == "Checked the connected gmail account"
    assert by_id[2]["payload"]["plain"] == "Emailing Sam"  # the recorded sentence wins


# ---------------------------------------------------------------------------
# /api/memory — resolved cross-platform identities + their latest note
# ---------------------------------------------------------------------------


class FakeEntitiesStore:
    def __init__(
        self, people: list[Entity] | None = None, notes: dict[int, list[EntityNote]] | None = None
    ):
        self.people = people or []
        self.notes = notes or {}

    async def list_people(self, limit: int = 50) -> list[Entity]:
        return self.people[:limit]

    async def notes_for(self, identity_id: int, limit: int = 10) -> list[EntityNote]:
        return self.notes.get(identity_id, [])[:limit]


async def test_memory_feed_is_token_guarded() -> None:
    async with _client(_app(None, entities=FakeEntitiesStore())) as client:
        assert (await client.get("/api/memory")).status_code == 401


async def test_memory_feed_is_503_without_a_store_wired() -> None:
    async with _client(_app(None)) as client:  # no entities on state
        response = await client.get("/api/memory", params={"token": "secret"})
    assert response.status_code == 503


async def test_memory_feed_reports_people_handles_and_latest_note() -> None:
    person = Entity(id=1, display_name="Sam", confidence=0.92, handles=[("telegram", "@sam")])
    note = EntityNote(
        id=9,
        identity_id=1,
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        payload={"kind": "relationship", "text": "coworker on the infra team"},
    )
    store = FakeEntitiesStore(people=[person], notes={1: [note]})
    async with _client(_app(None, entities=store)) as client:
        response = await client.get("/api/memory", params={"token": "secret"})
    data = response.json()
    assert data["people"] == [
        {
            "id": 1,
            "display_name": "Sam",
            "confidence": 0.92,
            "handles": [{"platform": "telegram", "handle": "@sam"}],
            "latest_note": {
                "at": "2026-09-20T00:00:00+00:00",
                "payload": {"kind": "relationship", "text": "coworker on the infra team"},
            },
        }
    ]


async def test_memory_feed_reports_null_note_for_a_person_with_none() -> None:
    person = Entity(id=2, display_name="Unnoted", confidence=1.0, handles=[])
    store = FakeEntitiesStore(people=[person])
    async with _client(_app(None, entities=store)) as client:
        response = await client.get("/api/memory", params={"token": "secret"})
    assert response.json()["people"][0]["latest_note"] is None


# ---------------------------------------------------------------------------
# /api/tasks — persisted scheduled actions
# ---------------------------------------------------------------------------


class FakeSchedulerStore:
    def __init__(self, actions: list[ScheduledAction] | None = None):
        self.actions = actions or []

    async def list(self, limit: int = 50) -> list[ScheduledAction]:
        return self.actions[:limit]


async def test_tasks_feed_is_token_guarded() -> None:
    async with _client(_app(None, scheduler=FakeSchedulerStore())) as client:
        assert (await client.get("/api/tasks")).status_code == 401


async def test_tasks_feed_is_503_without_a_scheduler_wired() -> None:
    async with _client(_app(None)) as client:  # no scheduler on state
        response = await client.get("/api/tasks", params={"token": "secret"})
    assert response.status_code == 503


async def test_tasks_feed_reports_scheduled_actions() -> None:
    action = ScheduledAction(
        id=5,
        label="follow up with sam",
        run_at=datetime(2026, 9, 27, 9, tzinfo=UTC),
        status="pending",
        payload={"type": "tool", "tool": "mail__send_message", "params": {}},
        created_at=datetime(2026, 9, 26, tzinfo=UTC),
    )
    store = FakeSchedulerStore(actions=[action])
    async with _client(_app(None, scheduler=store)) as client:
        response = await client.get("/api/tasks", params={"token": "secret"})
    assert response.json()["tasks"] == [
        {
            "id": 5,
            "label": "follow up with sam",
            "run_at": "2026-09-27T09:00:00+00:00",
            "status": "pending",
            "payload": {"type": "tool", "tool": "mail__send_message", "params": {}},
            "created_at": "2026-09-26T00:00:00+00:00",
        }
    ]


# ---------------------------------------------------------------------------
# /api/workspace* — the human-editable Markdown identity/memory layer
# ---------------------------------------------------------------------------


def _workspace(tmp_path) -> Workspace:
    ws = Workspace(tmp_path / "ws")
    ws.ensure_scaffold()
    return ws


async def test_workspace_feed_is_token_guarded(tmp_path) -> None:
    async with _client(_app(None, workspace=_workspace(tmp_path))) as client:
        assert (await client.get("/api/workspace")).status_code == 401
        assert (await client.get("/api/workspace/daily")).status_code == 401
        assert (await client.get("/api/workspace/search")).status_code == 401
        assert (await client.put("/api/workspace/soul", json={"text": "x"})).status_code == 401


async def test_workspace_feed_is_503_without_a_workspace_wired() -> None:
    async with _client(_app(None)) as client:  # no workspace on state
        response = await client.get("/api/workspace", params={"token": "secret"})
    assert response.status_code == 503


async def test_workspace_feed_reports_every_file_with_parsed_entries(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    workspace.remember("prefers concise notifications", "preference")
    async with _client(_app(None, workspace=workspace)) as client:
        response = await client.get("/api/workspace", params={"token": "secret"})
    body = response.json()
    assert body["needs_bootstrap"] is True
    assert set(body["files"]) == {"identity", "soul", "agents", "user", "memory"}
    user_entries = body["files"]["user"]["entries"]
    assert any(
        e["text"] == "prefers concise notifications" and e["source"] == "agent"
        for e in user_entries
    )
    assert "Name: Aether" in body["files"]["identity"]["raw"]


async def test_workspace_write_saves_a_human_edit(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    async with _client(_app(None, workspace=workspace)) as client:
        response = await client.put(
            "/api/workspace/soul",
            params={"token": "secret"},
            json={"text": "# SOUL.md\n\n- Be terse.\n"},
        )
    assert response.json() == {"ok": True}
    assert workspace.read("soul") == "# SOUL.md\n\n- Be terse.\n"


async def test_workspace_write_rejects_secrets_and_bad_files(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    async with _client(_app(None, workspace=workspace)) as client:
        secret = await client.put(
            "/api/workspace/memory",
            params={"token": "secret"},
            json={"text": "api_key: sk-abcdefghijklmnopqrstuvwxyz0123456789"},
        )
        assert secret.status_code == 422

        bad_file = await client.put(
            "/api/workspace/bootstrap",  # not in the editable set
            params={"token": "secret"},
            json={"text": "anything"},
        )
        assert bad_file.status_code == 400

        traversal = await client.put(
            "/api/workspace/..%2F..%2Fetc%2Fpasswd",
            params={"token": "secret"},
            json={"text": "anything"},
        )
        assert traversal.status_code in (400, 404)  # never a filesystem write outside the workspace


async def test_workspace_daily_feed_defaults_to_today(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    workspace.remember("deploy went out", "daily")
    async with _client(_app(None, workspace=workspace)) as client:
        response = await client.get("/api/workspace/daily", params={"token": "secret"})
    body = response.json()
    assert body["date"] == workspace.today()
    assert any(e["text"] == "deploy went out" for e in body["entries"])


async def test_workspace_daily_feed_rejects_a_malformed_date(tmp_path) -> None:
    async with _client(_app(None, workspace=_workspace(tmp_path))) as client:
        response = await client.get(
            "/api/workspace/daily", params={"token": "secret", "date": "not-a-date"}
        )
    assert response.status_code == 400


async def test_workspace_search_feed_finds_entries(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    workspace.remember("the user is working on Aether", "fact", category="Projects")
    async with _client(_app(None, workspace=workspace)) as client:
        response = await client.get(
            "/api/workspace/search", params={"token": "secret", "query": "working on Aether"}
        )
    hits = response.json()["hits"]
    assert any("working on Aether" in h["text"] for h in hits)


# /api/connections — the app hub, driven from the panel
# ---------------------------------------------------------------------------


async def test_connections_feed_is_token_guarded() -> None:
    async with _client(_app(None, composio=FakeComposioBridge())) as client:
        assert (await client.get("/api/connections")).status_code == 401
        assert (await client.post("/api/connections/gmail/connect")).status_code == 401
        assert (await client.post("/api/connections/acc-1/disconnect")).status_code == 401


async def test_connections_feed_is_503_without_a_bridge_wired() -> None:
    async with _client(_app(None)) as client:  # no composio on state
        response = await client.get("/api/connections", params={"token": "secret"})
        assert response.status_code == 503


async def test_connections_feed_answers_empty_and_unconfigured_without_a_key() -> None:
    app = _app(None, composio=FakeComposioBridge(available=False))
    async with _client(app) as client:
        response = await client.get("/api/connections", params={"token": "secret"})
    assert response.json() == {"configured": False, "connected": [], "toolkits": []}


async def test_connections_feed_lists_accounts_and_the_live_catalog() -> None:
    bridge = FakeComposioBridge(
        accounts=[ConnectedApp(id="acc-1", toolkit="gmail", status="ACTIVE", identity="me@x.test")]
    )
    async with _client(_app(None, composio=bridge)) as client:
        response = await client.get("/api/connections", params={"token": "secret"})
    data = response.json()
    assert data["configured"] is True
    assert data["connected"] == [
        {"id": "acc-1", "toolkit": "gmail", "status": "ACTIVE", "identity": "me@x.test"}
    ]
    assert data["toolkits"] == [
        {
            "slug": "gmail",
            "name": "Gmail",
            "logo": "https://logo.test/gmail.png",
            "description": "your mail",
        },
        {"slug": "github", "name": "GitHub", "logo": "", "description": "your code"},
    ]


async def test_connect_returns_the_link_and_the_loop_watches_it() -> None:
    bridge = FakeComposioBridge()
    agent = FakeAgent()
    async with _client(_app(None, composio=bridge, agent=agent)) as client:
        response = await client.post("/api/connections/gmail/connect", params={"token": "secret"})
    assert response.json() == {"redirect_url": "https://hub.example.test/connect/gmail"}
    assert bridge.authorized == ["gmail"]
    # a panel-driven connect lands in chat exactly like a chat-driven one
    assert agent.watched == [("gmail", "req-1")]


async def test_connect_without_a_key_is_an_honest_503() -> None:
    app = _app(None, composio=FakeComposioBridge(available=False))
    async with _client(app) as client:
        response = await client.post("/api/connections/gmail/connect", params={"token": "secret"})
    assert response.status_code == 503
    assert "COMPOSIO_API_KEY" in response.json()["detail"]


async def test_disconnect_removes_only_an_account_thats_ours() -> None:
    bridge = FakeComposioBridge(
        accounts=[ConnectedApp(id="acc-1", toolkit="gmail", status="ACTIVE", identity="me@x.test")]
    )
    async with _client(_app(None, composio=bridge)) as client:
        response = await client.post(
            "/api/connections/acc-1/disconnect", params={"token": "secret"}
        )
        missing = await client.post("/api/connections/acc-9/disconnect", params={"token": "secret"})
    assert response.json() == {"disconnected": "gmail", "id": "acc-1"}
    assert bridge.disconnected == ["acc-1"]
    assert missing.status_code == 404
