"""A real Aether web app wired to in-memory fakes instead of Postgres.

This is what the E2E tests drive with a real browser, and what a developer
can run by hand to click through the frontend without setting up a database:

    python tests/e2e/dev_server.py

Every fake here implements exactly the surface the real store classes do
(same methods, same return shapes) — routes.py and the frontend can't tell
the difference. Not a mock of the HTTP layer: the real FastAPI app, the
real routers, the real static files, just a fake persistence layer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Response, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from aether.agent.traces import Trace
from aether.api import router as api_router
from aether.authz.approvals import PENDING, Approval
from aether.chat.ws import router as chat_router
from aether.composio_bridge import ConnectedApp, ToolkitInfo
from aether.config import (
    AppConfig,
    AppContacts,
    AuthzConfig,
    AuthzRule,
    ContactRule,
    ContactsConfig,
    MCPServerConfig,
    MessagingConfig,
    PlatformToggle,
    Settings,
    TransportConfig,
)
from aether.connectors.base import InboundMessage
from aether.memory.entities import Entity, EntityNote
from aether.memory.events import Event
from aether.scheduler.jobs import ScheduledAction

WEB_DIR = Path(__file__).resolve().parents[2] / "src" / "aether" / "web"
API_TOKEN = "secret"

# the apps that carry their own contacts slice, matching the config model
_APPS_WITH_OWN_LISTS = ("gmail", "telegram", "discord", "slack")


class FakeApprovals:
    def __init__(self, now: datetime) -> None:
        self._rows = {
            1: Approval(
                id=1,
                tool_name="mail__send_message",
                params={
                    "to": "sam@example.com",
                    "subject": "Re: Thursday",
                    "body": "Works for me.",
                },
                status=PENDING,
                created_at=now - timedelta(minutes=6),
                expires_at=now + timedelta(hours=24),
                decided_at=None,
                decided_by=None,
                rules_matched="builtin:risky",
                note="reaches an external system or is hard to undo",
            ),
            2: Approval(
                id=2,
                tool_name="telegram__send_message",
                params={"chat_ref": "@sam", "text": "confirmed for 4pm"},
                status=PENDING,
                created_at=now - timedelta(minutes=2),
                expires_at=now + timedelta(hours=24),
                decided_at=None,
                decided_by=None,
                rules_matched="user:telegram__send_message",
                note="one-tap sends",
            ),
        }

    async def list_pending(self) -> list[Approval]:
        return [a for a in self._rows.values() if a.status == PENDING]

    async def decide(self, approval_id: int, decision: str) -> Approval | None:
        row = self._rows.get(approval_id)
        if row is None or row.status != PENDING:
            return None
        status = "approved" if decision == "approve" else "denied"
        updated = Approval(
            id=row.id,
            tool_name=row.tool_name,
            params=row.params,
            status=status,
            created_at=row.created_at,
            expires_at=row.expires_at,
            decided_at=datetime.now(UTC),
            decided_by="web",
            rules_matched=row.rules_matched,
            note=row.note,
        )
        self._rows[approval_id] = updated
        return updated


class FakeAgent:
    def __init__(self, approvals: FakeApprovals) -> None:
        self._approvals = approvals

    async def decide(self, approval_id: int, decision: str) -> Approval | None:
        return await self._approvals.decide(approval_id, decision)

    def notify(self) -> None:
        pass

    def submit_message(self, msg: InboundMessage) -> None:
        pass

    def watch_connect(self, toolkit: str, request_id: str) -> None:
        pass


class FakeAudit:
    def __init__(self, now: datetime) -> None:
        self.rows = [
            {
                "seq": 1,
                "actor": "agent",
                "tool_name": "mail__list_messages",
                "decision": "allow",
                "rules_matched": "builtin:read-only",
                "outcome": "read-only tool, no side effects",
                "created_at": now - timedelta(hours=2),
                "entry_hash": "7e4c19a04f8e2d1b91a0c3f5d8e7b2a6c4d9f1e8b3a7c2d5e9f0a1b4c8d6e3f7",
            },
            {
                "seq": 2,
                "actor": "agent",
                "tool_name": "telegram__send_message",
                "decision": "approve",
                "rules_matched": "builtin:risky",
                "outcome": "parked for approval",
                "created_at": now - timedelta(hours=1),
                "entry_hash": "b8e4d2f19a3c7e0b5d8f2a6c9e1b4d7f0a3c6e9b2d5f8a1c4e7b0d3f6a9c2e5b",
            },
            # decided rows — the attention page's "recently decided" strip reads these
            {
                "seq": 3,
                "actor": "user",
                "tool_name": "mail__send_message",
                "decision": "allow",
                "rules_matched": "",
                "outcome": "approved by user",
                "created_at": now - timedelta(minutes=40),
                "entry_hash": "c9f5e3a20b4d8f1c6e9a3d7f0b2e5c8a1d4f7b0e3c6a9d2f5b8e1c4a7d0f3b6e",
            },
            {
                "seq": 4,
                "actor": "agent",
                "tool_name": "mail__send_message",
                "decision": "info",
                "rules_matched": "",
                "outcome": "executed after approval",
                "created_at": now - timedelta(minutes=39),
                "entry_hash": "d0a6f4b31c5e9a2d7f0b4e8c1a6d3f9b2e5c8a1d4f7b0e3c6a9d2f5b8e1c4a7d",
            },
        ]

    async def recent(self, limit: int = 100) -> list[dict]:
        return list(reversed(self.rows))[:limit]  # newest first, matching the real AuditLog

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
        # the Contacts page's writes land here in the dev fixture — a real
        # hash would need the crypto module; a fixed one keeps the shape
        self.rows.append(
            {
                "seq": len(self.rows) + 1,
                "actor": actor,
                "tool_name": tool_name,
                "decision": decision,
                "rules_matched": rules_matched,
                "outcome": outcome,
                "created_at": datetime.now(UTC),
                "entry_hash": "f" * 64,
            }
        )
        return len(self.rows)

    async def verify_chain(self):
        from aether.authz.audit import ChainVerification

        return ChainVerification(ok=True, entries=len(self.rows), first_bad_seq=None, problem=None)


class FakeEventStore:
    def __init__(self, now: datetime) -> None:
        self.rows = [
            Event(
                id=3,
                source="agent",
                kind="action_done",
                occurred_at=now - timedelta(seconds=30),
                # the real agent-outcome shape (loop.py): tool is the raw
                # composio name and detail is str(result) — the raw JSON
                # envelope the tool returned, log_id and all
                payload={
                    "tool": "composio__GMAIL_WHO_AM_I",
                    "params": {},
                    # the model's own sentence, recorded by _remember_outcome
                    "plain": "Checking the connected Email account",
                    "detail": (
                        '{"successful":true,"data":{"data":{"email":"sam@example.com",'
                        '"name":"Sam"},"display_name":"sam@example.com"},'
                        '"error":null,"log_id":"log_abc123"}'
                    ),
                    "approval_id": 1,
                },
                salience_score=6.0,
                memorable=True,
                meta={},
            ),
            Event(
                id=4,
                source="agent",
                kind="action_done",
                occurred_at=now - timedelta(seconds=45),
                # no `plain` — a legacy agent row; the real events route
                # backfills it from the backend vocabulary
                payload={
                    "tool": "telegram__send_message",
                    "params": {"chat_ref": "@sam"},
                    "detail": "sent",
                    "approval_id": 2,
                },
                salience_score=5.0,
                memorable=True,
                meta={},
            ),
            Event(
                id=2,
                source="telegram",
                kind="chat_message",
                occurred_at=now - timedelta(minutes=1),
                # the real stored shape: the handle rides under _sender
                # (events.py SENDER_KEY), never as a top-level "handle"
                payload={
                    "text": "confirmed for 4pm",
                    "chat_ref": "@sam",
                    "_sender": {"platform": "telegram", "handle": "@sam"},
                },
                salience_score=8.0,
                memorable=True,
                meta={},
            ),
            Event(
                id=1,
                source="mail",
                kind="poll:unread",
                occurred_at=now - timedelta(minutes=10),
                payload={"from": "sam@example.com", "subject": "Re: Thursday"},
                salience_score=7.0,
                memorable=True,
                meta={},
            ),
        ]

    async def recent(self, limit: int = 50) -> list[Event]:
        return self.rows[:limit]


class FakeChatHistory:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def append(self, surface: str, direction: str, text: str) -> None:
        self.rows.append(
            {"direction": direction, "text": text, "at": datetime.now(UTC).isoformat()}
        )

    async def recent(self, limit: int = 50) -> list[dict]:
        return self.rows[-limit:]


class FakeChatHub:
    def __init__(self, history: FakeChatHistory) -> None:
        self._history = history
        self._clients: set[WebSocket] = set()

    @property
    def connected(self) -> int:
        return len(self._clients)

    def connect(self, ws: WebSocket) -> None:
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, text: str, *, persist: bool = True) -> None:
        if persist:
            await self._history.append("web", "out", text)
        for ws in list(self._clients):
            try:
                await ws.send_json({"type": "chat", "text": text})
            except Exception:
                self.disconnect(ws)


class FakeAgentSettings:
    def __init__(self) -> None:
        self.personality = ""

    async def get_personality(self) -> str:
        return self.personality

    async def set_personality(self, text: str) -> str:
        self.personality = text.strip()
        return self.personality


class FakeEntities:
    def __init__(self, now: datetime) -> None:
        self.people = [
            Entity(id=1, display_name="Sam", confidence=0.94, handles=[("telegram", "@sam")]),
        ]
        self.notes = {
            1: [
                EntityNote(
                    id=1,
                    identity_id=1,
                    created_at=now - timedelta(hours=2),
                    # the real note_entity tool payload: {kind, note, handle, platform}
                    payload={
                        "kind": "relationship",
                        "note": "coworker on the infra team",
                        "handle": "@sam",
                        "platform": "telegram",
                    },
                )
            ],
        }

    async def list_people(self, limit: int = 50) -> list[Entity]:
        return self.people[:limit]

    async def notes_for(self, identity_id: int, limit: int = 10) -> list[EntityNote]:
        return self.notes.get(identity_id, [])[:limit]


class FakeScheduler:
    def __init__(self, now: datetime) -> None:
        self.actions = [
            ScheduledAction(
                id=1,
                label="follow up with sam re: thursday",
                run_at=now + timedelta(hours=3),
                status="pending",
                payload={"type": "tool", "tool": "mail__send_message", "params": {}},
                created_at=now - timedelta(minutes=30),
            ),
        ]

    async def list(self, limit: int = 50) -> list[ScheduledAction]:
        return self.actions[:limit]


class FakeTraces:
    """One trace per kind, with realistic payloads — the Decision Traces page
    renders all four shapes from these."""

    def __init__(self, now: datetime) -> None:
        # the real AgentLoop._event_line format — the panel parses the payload
        # back out of this rather than ever showing the raw line
        mail_line = (
            f"{(now - timedelta(minutes=10)):%Y-%m-%d %H:%M} mail/poll:unread (salience 7): "
            '{"from": "sam@example.com", "subject": "Re: Thursday"}'
        )
        self.rows = [
            Trace(
                id=4,
                kind="carry_out",
                label="approval #1 — mail__send_message",  # the real label carries the raw tool name
                payload={
                    "approval_id": 1,
                    "tool": "mail__send_message",
                    "params": {"to": "sam@example.com", "subject": "Re: Thursday"},
                    "decided_by": "user",
                    "result": "sent",
                    "is_error": False,
                },
                created_at=now - timedelta(minutes=39),
            ),
            Trace(
                id=3,
                kind="scheduled",
                label="follow up with sam re: thursday",
                payload={
                    # real payload: run_at formatted "%Y-%m-%d %H:%M:%S%z", not ISO
                    "action": {
                        "id": 1,
                        "label": "follow up with sam re: thursday",
                        "run_at": f"{now + timedelta(hours=3):%Y-%m-%d %H:%M:%S%z}",
                    },
                    "calls": [
                        {
                            "name": "mail__list_messages",
                            "decision": "allow",
                            "matched_rule": "builtin:read-only",
                            "reason": "read-only tool, no side effects",
                            "result": "3 unread messages",
                            "is_error": False,
                        },
                        {
                            # composio actions arrive UPPER_SNAKE with the
                            # toolkit as the first word — the panel must read
                            # this as gmail, never as "composio"
                            "name": "composio__GMAIL_SEND_EMAIL",
                            "decision": "require_approval",
                            "matched_rule": "builtin:risky",
                            "reason": "reaches an external system or is hard to undo",
                            # the real parked-call result is an instruction to
                            # the model — the panel must not quote it
                            "result": (
                                "held for approval (#1) — the user has been asked on "
                                "their chat surfaces and the web panel"
                            ),
                            "is_error": False,
                            "approval_id": 1,
                        },
                    ],
                    "result": "no reply yet — nudge queued",
                    "is_error": False,
                },
                created_at=now - timedelta(minutes=30),
            ),
            Trace(
                id=2,
                kind="routine",
                label="morning mail triage",
                payload={
                    # the real trigger shape: a condition dict, not a string
                    "routine": {
                        "id": 1,
                        "label": "morning mail triage",
                        "trigger": {"source": "mail", "kind": "poll:unread"},
                    },
                    "event": {"id": 1, "source": "mail", "kind": "poll:unread", "line": mail_line},
                    "calls": [
                        {
                            "name": "note_entity",
                            "decision": "allow",
                            "matched_rule": "builtin:internal",
                            "reason": "internal tool, touches only Aether's own state",
                            "result": "noted",
                            "is_error": False,
                        }
                    ],
                },
                created_at=now - timedelta(minutes=20),
            ),
            Trace(
                id=1,
                kind="turn",
                # the real _trace_label: the text of the message that drove the turn
                label="can you confirm thursday at 4?",
                payload={
                    "trigger": {
                        "messages": [
                            {
                                "surface": "telegram",
                                "handle": "@sam",
                                "text": "can you confirm thursday at 4?",
                            }
                        ],
                        "observations": [
                            {"id": 1, "source": "mail", "kind": "poll:unread", "line": mail_line}
                        ],
                    },
                    "calls": [
                        {
                            "name": "mail__send_message",
                            # the model's own sentence, attached as `_plain`
                            # and recorded by _note_call — the new-row path
                            "plain": "Emailing Sam the Thursday confirmation",
                            "params": {"to": "sam@example.com", "subject": "Re: Thursday"},
                            "decision": "require_approval",
                            "matched_rule": "builtin:risky",
                            "reason": "reaches an external system or is hard to undo",
                            "approval_id": 1,
                        },
                        {
                            # no `plain` — a legacy row; the real route
                            # backfills it from the backend vocabulary
                            "name": "telegram__send_message",
                            "params": {"chat_ref": "@sam", "text": "confirmed for 4pm"},
                            "decision": "require_approval",
                            "matched_rule": "user:telegram__send_message",
                            "reason": "one-tap sends",
                            "approval_id": 2,
                        },
                    ],
                    # a line repeated across loop steps — the panel collapses
                    # consecutive duplicates rather than showing it twice
                    "reasoning": [
                        "Sam asked for a confirmation; the mail thread already has my draft reply.",
                        "Sam asked for a confirmation; the mail thread already has my draft reply.",
                        # models narrate raw tool names and paste raw JSON into
                        # their reasoning — the panel transcribes both
                        'I will run composio__GMAIL_SEND_EMAIL once approved — last check returned {"successful": true, "data": {"id": "msg-9"}}.',
                    ],
                    "reply": "I've drafted the reply to Sam — approve it and it goes out.",
                },
                created_at=now - timedelta(minutes=6),
            ),
        ]

    async def recent(self, limit: int = 30) -> list[Trace]:
        return self.rows[:limit]  # already newest first

    async def get(self, trace_id: int) -> Trace | None:
        return next((t for t in self.rows if t.id == trace_id), None)


class FakeConfigManager:
    """The read surface the panel uses (effective() like the real one) plus
    the small write surface the Contacts page needs — applied to the
    fixture's live AppConfig in memory, so the page round-trips by hand
    without Postgres."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    async def effective(self) -> dict:
        return {
            "sections": {
                "messaging": {
                    "telegram": {"enabled": True},
                    "discord": {"enabled": False},
                    "slack": {"enabled": False},
                },
                "mcp_servers": [{"name": "mail", "enabled": True}],
                "contacts": {"mode": "enforce"},
                "authz": {"approval_ttl_hours": 24},
                "agent": {"quiet_hours": ""},
                "llm": {"provider": "anthropic", "model": "claude-sonnet-5"},
            },
            "overrides": [
                {"path": "authz.approval_ttl_hours", "value": 24},
                {"path": "agent.personality", "value": "warm and a little informal"},
            ],
        }

    async def set_verdict(
        self, *, op: str, path: str, value: object = None, source: str = "panel"
    ) -> tuple[bool, str]:
        # the contacts vocabulary the panel writes: the default's mode and
        # list, plus one app's own contacts.<app>.mode / contacts.<app>.allowlist
        parts = path.split(".")
        default_section: ContactsConfig | AppContacts = self._config.contacts
        if len(parts) == 3 and parts[0] == "contacts" and parts[1] in _APPS_WITH_OWN_LISTS:
            default_section = getattr(self._config.contacts, parts[1])
            field = parts[2]
        elif len(parts) == 2 and parts[0] == "contacts":
            field = parts[1]
        else:
            return False, "the dev fixture only speaks the contacts settings"
        if field == "mode":
            if op == "reset":
                # only an app's own switch resets — back to following the default
                if not isinstance(default_section, AppContacts):
                    return False, "the default switch can't be reset from here"
                default_section.mode = None
            elif str(value) in ("enforce", "off"):
                default_section.mode = str(value)  # type: ignore[assignment]
            else:
                return False, "it must be one of: enforce, off, default"
        elif field == "allowlist":
            entries = default_section.allowlist
            if op == "add":
                entries.append(ContactRule.model_validate(value))
            else:
                gone = ContactRule.model_validate(value)
                if not any(
                    r.platform == gone.platform and r.handle == gone.handle for r in entries
                ):
                    return False, f"{gone.platform}/{gone.handle} isn't on the allowlist."
                entries[:] = [
                    r
                    for r in entries
                    if not (r.platform == gone.platform and r.handle == gone.handle)
                ]
        else:
            return False, "the dev fixture only speaks mode and allowlist"
        return True, "changed in the dev fixture — nothing persists across restarts"


class FakeTools:
    """The registry's read surface: which MCP servers answered and how many
    actions each exposes."""

    def mcp_status(self) -> dict:
        return {"mail": True}

    def server_action_count(self, name: str) -> int:
        return {"mail": 6}.get(name, 0)


class FakeResolver:
    """Secret-name surface only — the apps feed checks key presence by name."""

    async def known_names(self) -> set[str]:
        return {"TELEGRAM_BOT_TOKEN"}


class FakeHubBridge:
    """The Composio bridge double: one connected account and a small live
    catalog, so the apps page shows the hub section, the search, and the
    connect/disconnect round-trip. Credentials never appear — the connect
    is just a link."""

    def __init__(self) -> None:
        self.account_list = [
            ConnectedApp(
                id="acc-gmail", toolkit="gmail", status="ACTIVE", identity="me@example.com"
            ),
        ]
        self.toolkit_list = [
            ToolkitInfo(slug="gmail", name="Gmail", logo="", description="your mail"),
            ToolkitInfo(slug="github", name="GitHub", logo="", description="your code"),
            ToolkitInfo(slug="notion", name="Notion", logo="", description="your notes"),
        ]

    async def available(self) -> bool:
        return True

    async def accounts(self) -> list:
        return list(self.account_list)

    async def toolkits(self) -> list:
        return list(self.toolkit_list)

    async def authorize(self, toolkit: str) -> tuple[str, str]:
        return f"req-{toolkit}", f"https://hub.example.test/connect/{toolkit}"

    async def disconnect(self, account_id: str):
        for app in self.account_list:
            if app.id == account_id:
                self.account_list.remove(app)
                return app
        return None


def build_app() -> FastAPI:
    """A fresh app + fresh in-memory state — call once per test for isolation."""
    now = datetime.now(UTC)
    app = FastAPI(title="Aether (e2e dev fixture)")
    app.include_router(api_router)
    app.include_router(chat_router)
    # same unbuilt-panel guard as the real app: the panel is a Vite build
    # artifact, and a fresh checkout doesn't have it yet
    panel_built = (WEB_DIR / "index.html").is_file()
    if panel_built:
        app.mount("/assets", StaticFiles(directory=WEB_DIR), name="web-assets")

    # _env_file is load-bearing: without it pydantic-settings would read a
    # real .env sitting next to this fixture. Pyright's stubs miss the param.
    app.state.settings = Settings(_env_file=None, api_token=API_TOKEN)  # type: ignore[call-arg]
    config = AppConfig(
        authz=AuthzConfig(
            rules=[
                AuthzRule(
                    tool_pattern="telegram__send_message", decision="approve", note="one-tap sends"
                ),
            ],
            approval_ttl_hours=24,
        ),
        messaging=MessagingConfig(
            telegram=PlatformToggle(enabled=True),
            discord=PlatformToggle(enabled=False),
            slack=PlatformToggle(enabled=False),
        ),
        contacts=ContactsConfig(
            mode="enforce",
            allowlist=[
                # platform defaults to "*" — the page renders this as "Any platform"
                ContactRule(handle="sam@example.com"),
            ],
            # one app's own people: gmail's list also decides which inbox
            # mail the agent may read; telegram's holds its chat contacts
            gmail=AppContacts(allowlist=[ContactRule(platform="gmail", handle="abc123@xyz.com")]),
            telegram=AppContacts(allowlist=[ContactRule(platform="telegram", handle="@sam")]),
        ),
        mcp_servers=[
            MCPServerConfig(name="mail", transport=TransportConfig(type="http", url="https://x"))
        ],
    )
    app.state.config = config
    approvals = FakeApprovals(now)
    app.state.approvals = approvals
    app.state.audit = FakeAudit(now)
    app.state.events = FakeEventStore(now)
    app.state.agent = FakeAgent(approvals)
    chat_history = FakeChatHistory()
    app.state.chat_history = chat_history
    app.state.chat_hub = FakeChatHub(chat_history)
    app.state.agent_settings = FakeAgentSettings()
    app.state.entities = FakeEntities(now)
    app.state.scheduler = FakeScheduler(now)
    app.state.traces = FakeTraces(now)
    app.state.config_manager = FakeConfigManager(config)
    app.state.tools = FakeTools()
    app.state.resolver = FakeResolver()
    app.state.composio = FakeHubBridge()

    @app.get("/")
    async def index() -> Response:
        if not panel_built:
            return PlainTextResponse(
                "panel not built — run: cd webapp && npm install && npm run build",
                status_code=503,
            )
        return FileResponse(WEB_DIR / "index.html")

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(), host="127.0.0.1", port=8731, log_level="info")
