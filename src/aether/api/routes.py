"""REST endpoints. `settings` and component instances are read from
`request.app.state`, which the lifespan in main.py populates before the
server accepts traffic. Everything except /healthz is token-guarded —
the panel sends its token with every call.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel

from ..agent.tools import with_plain_calls, with_plain_event
from ..apps_catalog import CATALOG, recipe_for
from ..composio_bridge import ComposioNotConfigured
from ..workspace import SecretRejected, WorkspaceEntry, WorkspaceWriteError

log = logging.getLogger("aether.api")

router = APIRouter()

# a full-screen PNG is a few MB; anything past this is not a screenshot
MAX_CAPTURE_BYTES = 20 * 1024 * 1024


def _authorized(request: Request, token: str | None) -> bool:
    candidate = token or request.headers.get("x-aether-token", "")
    return bool(candidate) and hmac.compare_digest(candidate, request.app.state.settings.api_token)


class DecisionBody(BaseModel):
    decision: str  # "approve" | "deny"


class PersonalityBody(BaseModel):
    text: str


class WorkspaceTextBody(BaseModel):
    text: str


@router.post("/api/screen-capture", status_code=202)
async def screen_capture(
    request: Request,
    token: str | None = None,
    file: UploadFile = File(...),
    note: str = Form(""),
) -> dict:
    """The companion's upload point. Perception-only: the screenshot becomes
    a described, encrypted memory event — no action tools come from it."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    screen_vision = getattr(request.app.state, "screen_vision", None)
    if screen_vision is None:
        raise HTTPException(
            status_code=503,
            detail="screen vision is not configured (no vision-capable provider)",
        )
    png = await file.read(MAX_CAPTURE_BYTES + 1)
    if len(png) > MAX_CAPTURE_BYTES:
        raise HTTPException(status_code=413, detail="screenshot too large")
    if not png:
        raise HTTPException(status_code=400, detail="empty upload")
    result = await screen_vision.record(png, note)
    if result is None:  # defensive — record() only returns None pre-check
        raise HTTPException(status_code=503, detail="no vision provider")
    # fresh eyes for the agent: wake the loop so the capture is observed now
    agent = getattr(request.app.state, "agent", None)
    if agent is not None:
        agent.notify()
    return {"stored": result.stored, "reason": result.reason, "event_id": result.event_id}


@router.get("/api/screen-request")
async def take_screen_request(request: Request, token: str | None = None) -> dict:
    """The companion's poll: is the agent asking for a screenshot? Consumes
    the request (one capture per ask)."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    box = getattr(request.app.state, "capture_box", None)
    if box is None:
        raise HTTPException(status_code=503, detail="capture requests not wired")
    note = box.take()
    if note is None:
        return Response(status_code=204)  # nothing asked for — poll again later
    return {"note": note}


@router.get("/api/approvals")
async def list_approvals(request: Request, token: str | None = None) -> dict:
    """Pending approvals for the web panel — the same rows the chat
    surfaces show buttons for."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    approvals = request.app.state.approvals
    pending = await approvals.list_pending()
    return {
        "pending": [
            {
                "id": a.id,
                "tool_name": a.tool_name,
                "params": a.params,
                # why the gate parked it — the card's plain-language reason
                "rules_matched": a.rules_matched,
                "note": a.note,
                "created_at": a.created_at.isoformat(),
                "expires_at": a.expires_at.isoformat(),
            }
            for a in pending
        ]
    }


@router.post("/api/approvals/{approval_id}/decide")
async def decide_approval(
    approval_id: int, body: DecisionBody, request: Request, token: str | None = None
) -> dict:
    """The web panel's approve/deny buttons — the same path a chat-surface
    tap takes: decide, then run a freshly approved call."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="agent loop is not running")
    try:
        approval = await agent.decide(approval_id, body.decision)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if approval is None:
        return {"ok": False, "reason": "already decided or expired"}
    return {"ok": True, "status": approval.status}


@router.get("/api/events")
async def events_feed(request: Request, limit: int = 50, token: str | None = None) -> dict:
    """The newest events, decrypted for the owner's eyes only."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    events = await request.app.state.events.recent(limit=min(limit, 200))
    return {
        "events": [
            {
                "id": e.id,
                "source": e.source,
                "kind": e.kind,
                "at": e.occurred_at.isoformat(),
                "salience": e.salience_score,
                "memorable": e.memorable,
                # agent-outcome rows written before `plain` existed get the
                # backend's sentence at read time — old rows read like new
                "payload": (
                    with_plain_event(e.payload, kind=e.kind) if e.source == "agent" else e.payload
                ),
            }
            for e in events  # recent() is newest first — the feed's order
        ]
    }


_BUILTIN_RULES = [
    {
        "id": "builtin:config-tune",
        "decision": "allow",
        "description": "Personal tuning from chat: set_config on agent.*, salience.*, or "
        "llm.* — applies right away (llm.* also restarts the agent to load it).",
    },
    {
        "id": "builtin:config-security",
        "decision": "require_approval",
        "description": "set_config on the security sections (contacts, authz, messaging, "
        "mcp_servers) or on an unrecognized path — held for the user's one tap.",
    },
    {
        "id": "builtin:internal",
        "decision": "allow",
        "description": "Aether's own memory, scheduling, capture-request, and web-chat "
        "tools — they only ever touch Aether's own state.",
    },
    {
        "id": "builtin:risky",
        "decision": "require_approval",
        "description": "Verbs that reach an external system or are hard to undo: send, "
        "reply, forward, post, publish, delete, remove, cancel, pay, transfer, "
        "invite, book, create.",
    },
    {
        "id": "builtin:read-only",
        "decision": "allow",
        "description": "Observation verbs: list, search, get, read, fetch, find, query.",
    },
    {
        "id": "default:fail-safe",
        "decision": "require_approval",
        "description": "Anything that matches none of the above — an unknown tool can "
        "never run silently.",
    },
]


@router.get("/api/policy")
async def policy_feed(request: Request, token: str | None = None) -> dict:
    """Read-only view of the authorization policy: the user's own rules from
    config.yaml (evaluated first, in order) followed by the built-in ladder
    every call falls through to. No edit path — config.yaml is the source
    of truth."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    authz = request.app.state.config.authz
    return {
        "rules": [
            {
                "tool_pattern": r.tool_pattern,
                "param_pattern": r.param_pattern,
                "decision": r.decision,
                "note": r.note,
            }
            for r in authz.rules
        ],
        "builtin_rules": _BUILTIN_RULES,
        "approval_ttl_hours": authz.approval_ttl_hours,
    }


@router.get("/api/config")
async def config_feed(request: Request, token: str | None = None) -> dict:
    """Read-only: the live configuration (config.yaml merged with chat-made
    overrides) plus the override rows themselves. No edit path — changes
    come from chat (/config) through the same approval gate as every other
    tool call, never from the panel."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    manager = getattr(request.app.state, "config_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="config management is not wired")
    return await manager.effective()


@router.get("/api/settings/personality")
async def get_personality(request: Request, token: str | None = None) -> dict:
    """The owner's custom behavior instructions — appended to the real
    system prompt on every turn, not cosmetic."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    agent_settings = getattr(request.app.state, "agent_settings", None)
    if agent_settings is None:
        raise HTTPException(status_code=503, detail="agent settings store is not wired")
    return {"text": await agent_settings.get_personality()}


@router.put("/api/settings/personality")
async def set_personality(
    body: PersonalityBody, request: Request, token: str | None = None
) -> dict:
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    agent_settings = getattr(request.app.state, "agent_settings", None)
    if agent_settings is None:
        raise HTTPException(status_code=503, detail="agent settings store is not wired")
    saved = await agent_settings.set_personality(body.text)
    return {"ok": True, "text": saved}


@router.get("/api/settings/apps")
async def apps_feed(request: Request, token: str | None = None) -> dict:
    """Read-only view of what's actually connected: the native messaging
    toggles (and whether each one's keys are in place), the MCP servers
    with their live connection state, screen-vision availability, and the
    contact allowlist mode. The chat-surface recipes ride along so the
    panel can show what chat's /apps walk can add — hub apps come from
    /api/connections instead, fetched live. No edit path — connecting
    happens in chat, where keys are pasted and consumed."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    config = request.app.state.config

    # which key names exist right now — names only, never values
    resolver = getattr(request.app.state, "resolver", None)
    known = await resolver.known_names() if resolver is not None else set()

    def _has_token(platform: str) -> bool:
        recipe = recipe_for(platform)
        if recipe is None or not recipe.env_names:
            return False
        return all(name in known for name in recipe.env_names)

    # live MCP state from the tool registry — a server that's up counts
    # its actions; one that hasn't answered yet reads "still connecting"
    tools = getattr(request.app.state, "tools", None)
    status = tools.mcp_status() if tools is not None else {}

    def _actions(server_name: str) -> int:
        return tools.server_action_count(server_name) if tools is not None else 0

    return {
        "connectors": [
            {"name": name, "enabled": toggle.enabled, "has_token": _has_token(name)}
            for name, toggle in (
                ("telegram", config.messaging.telegram),
                ("discord", config.messaging.discord),
                ("slack", config.messaging.slack),
            )
        ],
        "mcp_servers": [
            {
                "name": s.name,
                "transport": s.transport.type,
                "enabled": s.enabled,
                "connected": bool(status.get(s.name)),
                "actions": _actions(s.name),
            }
            for s in config.mcp_servers
        ],
        "screen_vision_available": getattr(request.app.state, "screen_vision", None) is not None,
        "contacts_mode": config.contacts.mode,
        "catalog": [
            {"key": r.key, "name": r.name, "blurb": r.blurb, "kind": r.kind} for r in CATALOG
        ],
    }


# -- the app hub: Composio connections, driven from the panel -----------------------


@router.get("/api/connections")
async def connections_feed(request: Request, token: str | None = None) -> dict:
    """The app hub, live: every toolkit Composio can connect for this
    project (the panel searches and filters client-side) and this
    instance's connected accounts — toolkit, status, identity, never a
    credential. `configured: false` means no COMPOSIO_API_KEY yet; the
    shape still answers, empty."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    bridge = getattr(request.app.state, "composio", None)
    if bridge is None:
        raise HTTPException(status_code=503, detail="the app hub isn't wired on this instance")
    if not await bridge.available():
        return {"configured": False, "connected": [], "toolkits": []}
    accounts = await bridge.accounts()
    toolkits = await bridge.toolkits()
    return {
        "configured": True,
        "connected": [
            {"id": a.id, "toolkit": a.toolkit, "status": a.status, "identity": a.identity}
            for a in accounts
        ],
        "toolkits": [
            {"slug": t.slug, "name": t.name, "logo": t.logo, "description": t.description}
            for t in toolkits
        ],
    }


@router.post("/api/connections/{toolkit}/connect")
async def connect_app(request: Request, toolkit: str, token: str | None = None) -> dict:
    """Start one toolkit's Connect Link — the panel opens the returned URL
    in a new tab; credentials pass between the user and the hub only. The
    loop waits out of band and announces in chat, exactly like a
    chat-driven connect."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    bridge = getattr(request.app.state, "composio", None)
    if bridge is None:
        raise HTTPException(status_code=503, detail="the app hub isn't wired on this instance")
    try:
        request_id, url = await bridge.authorize(toolkit)
    except ComposioNotConfigured:
        raise HTTPException(
            status_code=503,
            detail="no COMPOSIO_API_KEY yet — say /apps add in chat and paste it once",
        ) from None
    except Exception:
        log.exception("connect start failed for %s", toolkit)
        raise HTTPException(status_code=502, detail="the hub couldn't start that connect") from None
    agent = getattr(request.app.state, "agent", None)
    if agent is not None:
        agent.watch_connect(toolkit, request_id)
    return {"redirect_url": url}


@router.post("/api/connections/{account_id}/disconnect")
async def disconnect_app(request: Request, account_id: str, token: str | None = None) -> dict:
    """Remove one connected account — the bridge verifies the id is this
    instance's own before anything is deleted; 404 when it isn't."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    bridge = getattr(request.app.state, "composio", None)
    if bridge is None:
        raise HTTPException(status_code=503, detail="the app hub isn't wired on this instance")
    try:
        removed = await bridge.disconnect(account_id)
    except ComposioNotConfigured:
        raise HTTPException(status_code=503, detail="no COMPOSIO_API_KEY set") from None
    if removed is None:
        raise HTTPException(status_code=404, detail="no such connected account")
    return {"disconnected": removed.toolkit, "id": removed.id}


@router.get("/api/audit")
async def audit_feed(request: Request, limit: int = 100, token: str | None = None) -> dict:
    """Recent audit entries plus the live chain verification — the panel
    shows both, so tampering is visible in the same view as the history."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    audit = request.app.state.audit
    rows = await audit.recent(limit=min(limit, 500))
    chain = await audit.verify_chain()
    return {
        "entries": [
            {
                "seq": r["seq"],
                "actor": r["actor"],
                "tool": r["tool_name"],
                "decision": r["decision"],
                "rules_matched": r["rules_matched"],
                "outcome": r["outcome"],
                "hash": r["entry_hash"],
                "at": r["created_at"].isoformat(),
            }
            for r in rows
        ],
        "chain": {
            "ok": chain.ok,
            "entries": chain.entries,
            "first_bad_seq": chain.first_bad_seq,
            "problem": chain.problem,
            # rows are newest-first, so the first row's hash is the current chain head
            "head_hash": rows[0]["entry_hash"] if rows else None,
        },
    }


@router.get("/api/traces")
async def traces_feed(request: Request, limit: int = 30, token: str | None = None) -> dict:
    """Recent decision traces (the "why" to the audit's "what") for the
    web panel — summaries; the full payload comes per trace."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    traces = getattr(request.app.state, "traces", None)
    if traces is None:
        raise HTTPException(status_code=503, detail="decision traces not wired")
    rows = await traces.recent(limit=min(limit, 200))
    return {
        "traces": [
            {
                "id": t.id,
                "kind": t.kind,
                "label": t.label,
                "at": t.created_at.isoformat(),
            }
            for t in rows  # recent() is newest first — the feed's order
        ]
    }


@router.get("/api/traces/{trace_id}")
async def trace_detail(trace_id: int, request: Request, token: str | None = None) -> dict:
    """One decision trace, decrypted for the owner's eyes: what triggered
    the act, the gate's ruling on every call, and what came back."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    traces = getattr(request.app.state, "traces", None)
    if traces is None:
        raise HTTPException(status_code=503, detail="decision traces not wired")
    trace = await traces.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"no trace {trace_id}")
    return {
        "id": trace.id,
        "kind": trace.kind,
        "label": trace.label,
        "at": trace.created_at.isoformat(),
        # calls recorded before `plain` existed get the backend's sentence
        # at read time — old traces read exactly like new ones
        "payload": with_plain_calls(trace.payload),
    }


@router.get("/api/memory")
async def memory_feed(request: Request, limit: int = 50, token: str | None = None) -> dict:
    """Everyone Aether has resolved a cross-platform identity for, with
    their linked handles and most recent relationship note."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    entities = getattr(request.app.state, "entities", None)
    if entities is None:
        raise HTTPException(status_code=503, detail="entity store is not wired")
    people = await entities.list_people(limit=min(limit, 200))
    out = []
    for person in people:
        notes = await entities.notes_for(person.id, limit=1)
        out.append(
            {
                "id": person.id,
                "display_name": person.display_name,
                "confidence": person.confidence,
                "handles": [{"platform": p, "handle": h} for p, h in person.handles],
                "latest_note": (
                    {"at": notes[0].created_at.isoformat(), "payload": notes[0].payload}
                    if notes
                    else None
                ),
            }
        )
    return {"people": out}


@router.get("/api/tasks")
async def tasks_feed(request: Request, limit: int = 50, token: str | None = None) -> dict:
    """Scheduled actions — persisted, survive restarts, run through the
    same authorization gate as everything else when they fire."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="scheduler is not wired")
    actions = await scheduler.list(limit=min(limit, 200))
    return {
        "tasks": [
            {
                "id": a.id,
                "label": a.label,
                "run_at": a.run_at.isoformat(),
                "status": a.status,
                "payload": a.payload,
                "created_at": a.created_at.isoformat(),
            }
            for a in actions
        ]
    }


_WORKSPACE_FILES = ("identity", "soul", "agents", "user", "memory")


def _entry_dict(entry: WorkspaceEntry) -> dict:
    return {
        "id": entry.id,
        "text": entry.text,
        "source": entry.source,  # "agent" | "user" | "human" (typed straight into the file)
        "at": entry.at,
        "status": entry.status,
        "confidence": entry.confidence,
    }


@router.get("/api/workspace")
async def workspace_feed(request: Request, token: str | None = None) -> dict:
    """Read-only view of the human-editable workspace: each curated file's
    parsed entries, so the panel can show generated vs human-authored
    content distinctly, plus whether first-run setup is still pending."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    workspace = getattr(request.app.state, "workspace", None)
    if workspace is None:
        raise HTTPException(status_code=503, detail="workspace is not enabled")
    return {
        "needs_bootstrap": workspace.needs_bootstrap(),
        "files": {
            key: {
                "raw": workspace.read(key),
                "entries": [_entry_dict(e) for e in workspace.entries(key)],
            }
            for key in _WORKSPACE_FILES
        },
    }


@router.put("/api/workspace/{file}")
async def workspace_write(
    file: str, body: WorkspaceTextBody, request: Request, token: str | None = None
) -> dict:
    """Save a direct edit to one of the curated workspace files — the
    control center's save path for human-authored content. Goes through
    the same secret guard as the agent's own writes."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    workspace = getattr(request.app.state, "workspace", None)
    if workspace is None:
        raise HTTPException(status_code=503, detail="workspace is not enabled")
    try:
        workspace.write_raw(file, body.text)
    except SecretRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except WorkspaceWriteError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True}


@router.get("/api/workspace/daily")
async def workspace_daily_feed(
    request: Request, date: str | None = None, token: str | None = None
) -> dict:
    """One day's working notes — defaults to today."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    workspace = getattr(request.app.state, "workspace", None)
    if workspace is None:
        raise HTTPException(status_code=503, detail="workspace is not enabled")
    resolved = date or workspace.today()
    try:
        entries = workspace.entries("daily", date_str=resolved)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"date": resolved, "entries": [_entry_dict(e) for e in entries]}


@router.get("/api/workspace/search")
async def workspace_search_feed(
    request: Request, query: str = "", limit: int = 10, token: str | None = None
) -> dict:
    """Deterministic keyword search across the whole workspace — identity,
    soul, agent instructions, user preferences, long-term memory, and
    recent daily notes."""
    if not _authorized(request, token):
        raise HTTPException(status_code=401, detail="bad or missing token")
    workspace = getattr(request.app.state, "workspace", None)
    if workspace is None:
        raise HTTPException(status_code=503, detail="workspace is not enabled")
    hits = workspace.search(query, limit=min(limit, 50))
    return {"hits": [{"file": h.file, "id": h.id, "text": h.text, "score": h.score} for h in hits]}
