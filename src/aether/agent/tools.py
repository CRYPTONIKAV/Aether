"""Aether's native tools — reading its own memory, keeping entity notes,
scheduling, requesting screen captures, talking to the user, managing
the user's routines, replaying why it acted, and proving the record
intact.

These are internal by construction: they touch only Aether's own state
(or the owner's own chat surfaces), which is why the authz classifier's
`builtin:internal` rule lets them run without approval. Everything that
leaves Aether still goes through the gate.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

from ..authz.audit import verification_text
from ..connectors.registry import ToolRegistry
from ..llm.types import ToolSpec, with_plain_param
from ..workspace import SecretRejected, WorkspaceWriteError

log = logging.getLogger("aether.agent.tools")

# what the tools need, expressed as protocols the loop actually owns
EventSearch = Callable[..., Awaitable[Any]]
NativeHandler = Callable[[dict[str, Any]], Awaitable[str]]


def _spec(name: str, description: str, properties: dict[str, Any], required: list[str]) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema=with_plain_param(
            {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        ),
    )


def _clip(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _parse_int_param(
    params: dict[str, Any], key: str, default: int, *, minimum: int | None = None
) -> int | None:
    """Parse an integer tool parameter, returning None (instead of raising)
    if the model passed something that isn't one — callers either report
    that in plain words or fall back to `default`, their choice."""
    try:
        value = int(params.get(key, default))
    except (TypeError, ValueError):
        return None
    return max(minimum, value) if minimum is not None else value


# how a tool call is said in plain words at each moment it reaches the
# user: the question when it's held for approval, the words when it's
# narrated, and the report after it ran. The record keeps the exact
# name; chat never shows one.
class ActionWords(NamedTuple):
    ask: str  # "Send the email"
    doing: str  # "sending an email"
    done: str  # "sent the email"


_ACTION_WORDS: dict[str, ActionWords] = {
    "mail": ActionWords("Send the email", "sending an email", "sent the email"),
    "gmail": ActionWords("Send the email", "sending an email", "sent the email"),
    "telegram": ActionWords("Send the message", "sending a message", "sent the message"),
    "discord": ActionWords("Send the message", "sending a message", "sent the message"),
    "slack": ActionWords("Send the message", "sending a message", "sent the message"),
    "whatsapp": ActionWords("Send the message", "sending a message", "sent the message"),
    "calendar": ActionWords(
        "Make the calendar change", "a calendar action", "made the calendar change"
    ),
    "googlecalendar": ActionWords(
        "Make the calendar change", "a calendar action", "made the calendar change"
    ),
    "github": ActionWords("Make the github change", "a github action", "made the github change"),
    "notion": ActionWords("Update notion", "a notion update", "updated notion"),
    "linear": ActionWords("Make the linear change", "a linear action", "made the linear change"),
    "payments": ActionWords("Send the payment", "a payment", "sent the payment"),
}

_NATIVE_WORDS: dict[str, ActionWords] = {
    "send_chat_message": ActionWords("Send the message", "messaging you", "sent you the message"),
    "memory_search": ActionWords("Search my memory", "searching my memory", "searched my memory"),
    "note_entity": ActionWords("Keep the note", "keeping a note", "kept the note"),
    "get_pending_approvals": ActionWords(
        "Check pending approvals", "checking pending approvals", "checked pending approvals"
    ),
    "schedule_action": ActionWords(
        "Schedule the action", "scheduling an action", "scheduled the action"
    ),
    "request_screen_capture": ActionWords(
        "Take the screenshot", "asking for a screenshot", "asked for the screenshot"
    ),
    "create_routine": ActionWords("Arm the routine", "arming a routine", "armed the routine"),
    "list_routines": ActionWords("List the routines", "listing routines", "listed the routines"),
    "set_routine_enabled": ActionWords(
        "Arm or pause the routine",
        "pausing or re-arming a routine",
        "changed the routine's armed state",
    ),
    "delete_routine": ActionWords(
        "Delete the routine", "deleting a routine", "deleted the routine"
    ),
    "explain_decision": ActionWords(
        "Replay the decision", "replaying a decision", "replayed the decision"
    ),
    "verify_integrity": ActionWords(
        "Verify my records", "verifying my records", "verified my records"
    ),
    "get_config": ActionWords(
        "Read my configuration", "reading my configuration", "read my configuration"
    ),
    "set_config": ActionWords(
        "Change my configuration", "changing my configuration", "changed my configuration"
    ),
    "workspace_read": ActionWords(
        "Read my workspace notes", "checking my workspace notes", "read my workspace notes"
    ),
    "workspace_search": ActionWords(
        "Search my workspace notes", "searching my workspace notes", "searched my workspace notes"
    ),
    "workspace_remember": ActionWords(
        "Write the workspace note", "writing a workspace note", "wrote the workspace note"
    ),
    "workspace_promote": ActionWords(
        "Promote the workspace note", "promoting a workspace note", "promoted the workspace note"
    ),
    "workspace_rewrite": ActionWords(
        "Update my identity file",
        "updating my identity or personality file",
        "updated my identity file",
    ),
    "workspace_finish_bootstrap": ActionWords(
        "Finish first-run setup", "finishing first-run setup", "finished first-run setup"
    ),
    "search_tools": ActionWords(
        "Look for the right tool", "looking for the right tool", "looked for the right tool"
    ),
}

# the few parameters worth putting in front of the user, keyed to the
# label they read under — everything else stays in the decision record
_SALIENT_PARAMS: tuple[tuple[str, str], ...] = (
    ("to", "to"),
    ("recipient", "to"),
    ("recipients", "to"),
    ("email", "to"),
    ("channel", "channel"),
    ("channel_id", "channel"),
    ("chat_id", "chat"),
    ("chat_ref", "to"),
    ("username", "username"),
    ("title", "title"),
    ("subject", "subject"),
    ("label", "label"),
    ("name", "name"),
    ("query", "query"),
    ("description", "description"),
    ("file_name", "file"),
    ("path", "path"),
)


def _action_words(name: str) -> ActionWords:
    """(ask, doing, done) for one tool call, the way the user would say it."""
    server, sep, tool = name.partition("__")
    if sep:
        lowered = tool.lower()
        # a composio tool carries its toolkit as the name's prefix —
        # composio__GMAIL_SEND_EMAIL reads as gmail, never as "composio";
        # the verb checks below run on the action that follows the toolkit
        if server == "composio":
            app, _, action = lowered.partition("_")
        else:
            app, action = server, lowered
        # an action idiom the verb tables can't phrase
        if action == "who_am_i":
            account = f"the connected {app} account"
            return ActionWords(f"Check {account}", f"checking {account}", f"checked {account}")
        # a read-like tool must never read as a send — "I went ahead with
        # sending an email" about listing mail would be a lie
        read_like = ("list", "search", "get", "read", "find", "check", "query", "fetch")
        if any(action.startswith(verb) for verb in read_like):
            return ActionWords(f"Check {app}", f"checking {app}", f"checked {app}")
        return _ACTION_WORDS.get(
            app,
            ActionWords(f"Run the {app} action", f"an action in {app}", f"made the {app} change"),
        )
    return _NATIVE_WORDS.get(
        name, ActionWords("Run the internal step", "an internal step", "ran the internal step")
    )


def _plain_action(name: str) -> str:
    """Plain words for a tool call, the way the user would say it."""
    return _action_words(name).doing


def _salient_context(params: Any) -> str:
    """The one or two details that make an action concrete — "to
    rahul@gmail.com, subject hello" — never a full argument dump. The
    decision record keeps everything."""
    if not isinstance(params, dict):
        return ""
    bits: list[str] = []
    for key, label in _SALIENT_PARAMS:
        if len(bits) >= 3:
            break
        value = params.get(key)
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        if not isinstance(value, (str, int, float)):
            continue
        text = str(value).strip()
        if not text or len(text) > 80:
            continue
        shown = f"'{text}'" if " " in text else text
        bits.append(f"{label} {shown}")
    return ", ".join(bits)


def describe_ask(name: str, params: Any = None) -> str:
    """The approval question exactly as the user reads it — "Send the
    email (to rahul@gmail.com, subject hello)?" Never the raw tool name,
    never a JSON dump; the decision record keeps the exact call."""
    ask = _action_words(name).ask
    ctx = _salient_context(params)
    return f"{ask} ({ctx})?" if ctx else f"{ask}?"


def describe_outcome(name: str, params: Any = None) -> str:
    """The after-run chat message — "Sent the email (to rahul@gmail.com)."
    The result stated plainly, with no status ceremony around it; the exact
    tool output stays in the decision record."""
    done = _action_words(name).done
    ctx = _salient_context(params)
    text = f"{done} ({ctx})" if ctx else done
    return text[0].upper() + text[1:] + "."


def describe_denied(name: str, params: Any = None) -> str:
    """The not-run record line — "Not run — you denied sending an email (to
    rahul@gmail.com)." For the activity log; the chat acknowledgment of a
    denial is describe_dismissed."""
    doing = _action_words(name).doing
    ctx = _salient_context(params)
    return f"Not run — you denied {doing} ({ctx})." if ctx else f"Not run — you denied {doing}."


def describe_dismissed(name: str, params: Any = None) -> str:
    """The chat acknowledgment when the user taps Deny — "Okay — not
    sending an email (to rahul@gmail.com)." A plain okay, not a report."""
    doing = _action_words(name).doing
    ctx = _salient_context(params)
    return f"Okay — not {doing} ({ctx})." if ctx else f"Okay — not {doing}."


def describe_call(name: str, params: Any = None) -> str:
    """One call as a plain sentence — "Checking mail (from sam@example.com)".
    Recorded on every traced call as `plain`, and computed at read time for
    rows that predate the field. Never the raw tool name, never a dump."""
    doing = _action_words(name).doing
    ctx = _salient_context(params)
    text = f"{doing} ({ctx})" if ctx else doing
    return text[0].upper() + text[1:]


def plain_outcome(name: str, params: Any, kind: str) -> str:
    """The `plain` line for a stored agent-outcome event, by kind — the
    past-tense sentence for a completed run, framed for a failure or a
    denial. A finished act must never read as one still in flight, or a
    later turn re-drafts it as something left to do."""
    if kind == "action_failed":
        doing = describe_call(name, params)
        return f"Failed — {doing[0].lower()}{doing[1:]}"
    if kind == "action_denied":
        text = describe_denied(name, params)
        return text[:-1] if text.endswith(".") else text
    text = describe_outcome(name, params)
    return text[:-1] if text.endswith(".") else text


def with_plain_calls(payload: Any) -> Any:
    """A trace payload with `plain` on every recorded call that lacks it —
    rows written before the field existed get the backend's sentence at read
    time, so old traces read exactly like new ones. Returns a copy; the
    store is never rewritten."""
    if not isinstance(payload, dict) or not isinstance(payload.get("calls"), list):
        return payload
    return {
        **payload,
        "calls": [
            {**c, "plain": describe_call(c.get("name"), c.get("params"))}
            if isinstance(c, dict) and c.get("name") and not c.get("plain")
            else c
            for c in payload["calls"]
        ],
    }


def with_plain_event(payload: Any, *, kind: str = "") -> Any:
    """An agent-outcome event payload with `plain` filled in if missing —
    the same read-time backfill as with_plain_calls, for the activity feed."""
    if not isinstance(payload, dict) or payload.get("plain") or not payload.get("tool"):
        return payload
    return {**payload, "plain": plain_outcome(payload["tool"], payload.get("params"), kind)}


def format_trace(t: Any) -> str:
    """Render one recorded trace for the model to answer from: the
    trigger, every gated call with its ruling, and what came back."""
    p = t.payload or {}
    lines = [f"Trace #{t.id} ({t.kind}) — {t.label or '(no label)'}"]
    trigger = p.get("trigger") or {}
    for m in trigger.get("messages") or []:
        lines.append(
            f"  asked: {m.get('handle', '?')} via {m.get('surface', '?')}: {m.get('text', '')}"
        )
    observations = trigger.get("observations") or []
    if observations:
        lines.append(f"  seen: {len(observations)} new event(s)")
        for o in observations:
            lines.append(f"    [{o.get('source')}/{o.get('kind')}] {o.get('line', '')}")
    recalled = trigger.get("recalled") or []
    if recalled:
        lines.append(f"  kept in mind: {len(recalled)} related memory(ies)")
        for r in recalled:
            lines.append(f"    [{r.get('source')}/{r.get('kind')}] {r.get('line', '')}")
    routine = p.get("routine")
    if routine:
        lines.append(
            f"  routine: #{routine.get('id')} '{routine.get('label')}' "
            f"when {json.dumps(routine.get('trigger') or {}, ensure_ascii=False)}"
        )
    event = p.get("event")
    if event:
        lines.append(
            f"  fired on: [{event.get('source')}/{event.get('kind')}] {event.get('line', '')}"
        )
    action = p.get("action")
    if action:
        lines.append(
            f"  scheduled: #{action.get('id')} '{action.get('label')}' at {action.get('run_at')}"
        )
    if p.get("approval_id") is not None:
        lines.append(
            f"  approval #{p.get('approval_id')}, decided by {p.get('decided_by') or 'the user'}"
        )
    for c in p.get("calls") or []:
        gate = f"gate: {c.get('decision', '?')}"
        if c.get("matched_rule"):
            gate += f" ({c.get('matched_rule')})"
        if c.get("audit_seq") is not None:
            gate += f" [audit #{c.get('audit_seq')}]"
        if c.get("approval_id") is not None:
            gate += f" [held as approval #{c.get('approval_id')}]"
        lines.append(
            f"  → {c.get('name', '?')} "
            f"{json.dumps(c.get('params') or {}, ensure_ascii=False, default=str)} "
            f"— {gate} → {_clip(str(c.get('result', '')), 200)}"
        )
    if p.get("audit_seq") is not None and not p.get("calls"):
        lines.append(f"  audit: #{p.get('audit_seq')}")
    for r in p.get("reasoning") or []:
        lines.append(f"  reasoning: {r}")
    if p.get("reply"):
        lines.append(f"  reply: {p['reply']}")
    # a top-level result only exists on acts that are not a call list
    # (an approved call carried out); a scheduled fire's result already
    # shows on its call line
    if p.get("result") is not None and not p.get("calls"):
        suffix = " (error)" if p.get("is_error") else ""
        lines.append(f"  result: {_clip(str(p.get('result')), 200)}{suffix}")
    if p.get("error"):
        lines.append(f"  error: the turn failed — {p['error']}")
    return "\n".join(lines)


def plain_replay(trace: Any) -> str:
    """The recorded trace as the user reads it in chat: what happened, in
    plain words, with the trace id for everything exact. Internal call
    names and raw error text stay in the record — this never echoes them."""
    p = trace.payload or {}
    lines = [f"that message, from the record (trace #{trace.id}):"]

    trigger = p.get("trigger") or {}
    messages = trigger.get("messages") or []
    if messages:
        first = _clip(str(messages[0].get("text", "")), 200)
        more = f" (+{len(messages) - 1} more)" if len(messages) > 1 else ""
        lines.append(f'You asked: "{first}"{more}')
    elif p.get("routine"):
        lines.append(f"Your routine '{p['routine'].get('label')}' fired.")
    elif p.get("action"):
        lines.append(f"The scheduled action '{p['action'].get('label')}' came due.")
    elif p.get("approval_id") is not None:
        lines.append(
            f"Approval #{p.get('approval_id')}, decided by {p.get('decided_by') or 'you'}."
        )
    elif trigger.get("observations"):
        count = len(trigger["observations"])
        lines.append(f"I noticed {count} new event{'s' if count != 1 else ''}.")

    for c in p.get("calls") or []:
        action = _plain_action(str(c.get("name", "")))
        decision = str(c.get("decision", ""))
        if c.get("is_error") and decision != "deny":
            # the exact failure stays in the trace — chat gets the shape
            lines.append(f"I tried {action} — it didn't work out.")
        elif decision == "allow":
            rule = f" ({c.get('matched_rule')})" if c.get("matched_rule") else ""
            lines.append(f"I went ahead with {action} — the gate let it through{rule}.")
        elif decision == "require_approval":
            held = f" (#{c.get('approval_id')})" if c.get("approval_id") is not None else ""
            lines.append(f"I proposed {action}; the gate held it for your one-tap approval{held}.")
        elif decision == "deny":
            reason = f" ({c.get('reason')})" if c.get("reason") else ""
            lines.append(f"I proposed {action}; the gate blocked it{reason}.")
        else:
            lines.append(f"I proposed {action}; it didn't run.")

    # a top-level result only exists on acts that are not a call list
    # (an approved call carried out); a scheduled fire's result already
    # shows on its call line
    if p.get("result") is not None and not p.get("calls"):
        if p.get("is_error"):
            lines.append("It couldn't run — I said so at the time.")
        else:
            lines.append("It ran as approved.")

    if p.get("reply"):
        lines.append(f'I replied: "{_clip(str(p["reply"]), 200)}"')

    lines.append(
        "The full recorded detail — the exact call, the parameters, the "
        f"gate's ruling — is trace #{trace.id} in the panel."
    )
    return "\n".join(lines)


def register_native_tools(
    *,
    registry: ToolRegistry,
    events: Any,
    entities: Any,
    approvals: Any,
    scheduler: Any,
    surfaces: Any,
    capture_box: Any,
    routines: Any = None,
    traces: Any = None,
    audit: Any = None,
    config_manager: Any = None,
    workspace: Any = None,
    vectors: Any = None,
) -> int:
    """Add Aether's own tools to the flat namespace. Returns how many.

    `routines` wires the standing-trigger tools, `traces` the
    decision-replay tool, `audit` the integrity check, `config_manager` the
    configuration tools, and `workspace` the Markdown workspace tools;
    without the store behind one it is not offered at all (same shape as a
    loop without an MCP host: no feature, no dead tool spec)."""

    async def memory_search(params: dict[str, Any]) -> str:
        query = str(params.get("query", "")).strip()
        limit = _parse_int_param(params, "limit", 5)
        if limit is None:
            return "memory_search needs limit to be a number."
        # a bracketed id from an earlier hit re-reads one whole stored
        # document, untruncated, page by page — no query needed. An absent
        # id must stay absent: the parse helper clamps to `minimum`, which
        # would turn every plain query into a read of event 1
        if params.get("event_id") not in (None, ""):
            event_id = _parse_int_param(params, "event_id", 1, minimum=1)
            if event_id is None:
                return "memory_search needs event_id to be a number."
            if vectors is not None:
                text = await vectors.read_document(event_id)
                if text:
                    return text
            event = await events.get(event_id)
            if event is None:
                return f"No stored memory with id {event_id}."
            p = event.payload or {}
            if p.get("text") and p.get("filename"):
                # the pre-vector read-back: the payload's text preview
                return f"{p.get('filename')}:\n" + str(p.get("text", ""))
            return _clip(json.dumps(event.payload, default=str, ensure_ascii=False), 2000)
        if not query:
            return "memory_search needs a query."
        if vectors is not None:
            try:
                hits = await vectors.search(query, limit=limit)
            except Exception:
                hits = []
            if hits:
                lines = []
                for hit in hits:
                    where = ", ".join(part for part in (hit.filename, hit.location) if part)
                    lines.append(
                        f"[{hit.event_id}] {hit.occurred_at:%Y-%m-%d %H:%M} "
                        f"{hit.source}/{hit.kind}"
                        + (f" — {where}:\n" if where else ":\n")
                        + hit.text
                    )
                return "\n".join(lines)
        found = await events.search(query, limit=limit)
        if not found:
            return "No matching memories."
        lines = []
        for e in found:
            p = e.payload or {}
            if p.get("text") and p.get("filename"):
                # a shared document — the one channel where the model can
                # re-read what the file said; the JSON dump would clip the
                # text away to nothing
                lines.append(
                    f"[{e.id}] {e.occurred_at:%Y-%m-%d %H:%M} {e.source}/{e.kind} "
                    f"— {p.get('filename')}:\n" + _clip(str(p.get("text", "")), 2000)
                )
                continue
            lines.append(
                f"[{e.id}] {e.occurred_at:%Y-%m-%d %H:%M} {e.source}/{e.kind}: "
                + _clip(json.dumps(e.payload, default=str, ensure_ascii=False))
            )
        return "\n".join(lines)

    async def note_entity(params: dict[str, Any]) -> str:
        handle = str(params.get("handle", "")).strip()
        note = str(params.get("note", "")).strip()
        if not handle or not note:
            return "note_entity needs a handle and a note."
        platform = str(params.get("platform", "*")).strip() or "*"
        entity = await entities.resolve(platform, handle, str(params.get("display_name", "")))
        await entities.note(
            entity.id,
            {
                "kind": str(params.get("kind", "general")),
                "note": note,
                "handle": handle,
                "platform": platform,
            },
        )
        return f"Noted on {entity.display_name or handle} (identity {entity.id})."

    async def get_pending_approvals(params: dict[str, Any]) -> str:
        pending = await approvals.list_pending()
        if not pending:
            return "No pending approvals."
        return "\n".join(
            f"#{a.id} {a.tool_name} ({a.created_at:%Y-%m-%d %H:%M}, "
            f"expires {a.expires_at:%Y-%m-%d %H:%M}): "
            + _clip(json.dumps(a.params, default=str, ensure_ascii=False), 200)
            for a in pending
        )

    async def schedule_action(params: dict[str, Any]) -> str:
        label = str(params.get("label", "")).strip()
        tool_name = str(params.get("tool_name", "")).strip()
        if not label or not tool_name:
            return "schedule_action needs a label and a tool_name."
        raw_when = str(params.get("run_at", "")).strip()
        try:
            run_at = datetime.fromisoformat(raw_when)
        except ValueError:
            return f"run_at {raw_when!r} is not an ISO datetime (e.g. 2026-09-25T18:30:00+05:30)."
        if run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=UTC)
        if run_at <= datetime.now(UTC) + timedelta(seconds=5):
            return "run_at must be in the future."
        action = await scheduler.create(
            label=label,
            run_at=run_at,
            payload={
                "type": "tool",
                "tool": tool_name,
                "params": dict(params.get("params") or {}),
            },
        )
        return (
            f"Scheduled as action {action.id} for {run_at.isoformat()}. It will "
            "still pass the authorization gate when it fires — a risky tool "
            "will ask the user then."
        )

    async def request_screen_capture(params: dict[str, Any]) -> str:
        reason = str(params.get("reason", "")).strip()
        capture_box.set(reason)
        return (
            "Screen capture requested — the companion (`aether_snap --poll`) "
            "picks the request up and uploads a screenshot; it will show up "
            "as a screen_capture memory as soon as it lands."
        )

    async def send_chat_message(params: dict[str, Any]) -> str:
        text = str(params.get("text", "")).strip()
        if not text:
            return "send_chat_message needs text."
        await surfaces.send_to_user(text)
        return "Sent to the user's chat surfaces."

    async def verify_integrity(params: dict[str, Any]) -> str:
        verification = await audit.verify_chain()
        newest = None
        if verification.entries:
            rows = await audit.recent(1)
            newest = rows[0]["created_at"] if rows else None
        return verification_text(verification, newest)

    async def get_config(params: dict[str, Any]) -> str:
        path = str(params.get("path", "")).strip() or None
        return await config_manager.show(path)

    async def set_config(params: dict[str, Any]) -> str:
        op = str(params.get("op", "")).strip().lower()
        path = str(params.get("path", "")).strip()
        if op not in ("set", "add", "remove", "reset"):
            return "set_config needs an op: set, add, remove, or reset."
        if not path:
            return "set_config needs a config path — e.g. agent.tick_seconds."
        if op == "set" and "value" not in params:
            return 'set needs a value — pass "value": null to clear one.'
        return await config_manager.set(op=op, path=path, value=params.get("value"), source="tool")

    async def workspace_read(params: dict[str, Any]) -> str:
        file = str(params.get("file", "")).strip().lower()
        if not file:
            return "workspace_read needs a file: identity, soul, agents, user, memory, bootstrap, or daily."
        date_str = str(params.get("date", "")).strip() or None
        try:
            text = workspace.read(file, date_str=date_str)
        except ValueError as exc:
            return str(exc)
        return text or f"{file} is empty."

    async def workspace_search(params: dict[str, Any]) -> str:
        query = str(params.get("query", "")).strip()
        if not query:
            return "workspace_search needs a query."
        limit = _parse_int_param(params, "limit", 5)
        if limit is None:
            return "workspace_search needs limit to be a number."
        hits = workspace.search(query, limit=limit)
        if not hits:
            return "No matching workspace memory."
        return "\n".join(f"[{h.file}#{h.id or '-'}] {h.text}" for h in hits)

    async def workspace_remember(params: dict[str, Any]) -> str:
        text = str(params.get("text", "")).strip()
        kind = str(params.get("kind", "")).strip().lower()
        if not text or kind not in ("preference", "fact", "daily"):
            return "workspace_remember needs text and a kind: preference, fact, or daily."
        source = str(params.get("source") or "agent").strip().lower()
        try:
            entry_id = workspace.remember(
                text,
                kind,
                source=source,
                category=str(params.get("category") or "").strip() or None,
                confidence=params.get("confidence"),
                supersedes=[str(s) for s in (params.get("supersedes") or [])] or None,
            )
        except (SecretRejected, ValueError) as exc:
            return f"not saved: {exc}"
        except WorkspaceWriteError as exc:
            return f"not saved — {exc}"
        return f"Saved to the workspace as entry {entry_id}."

    async def workspace_promote(params: dict[str, Any]) -> str:
        date_str = str(params.get("date", "")).strip()
        entry_id = str(params.get("entry_id", "")).strip()
        if not date_str or not entry_id:
            return "workspace_promote needs a date (YYYY-MM-DD) and an entry_id."
        try:
            ok = workspace.promote(
                date_str, entry_id, category=str(params.get("category") or "").strip() or None
            )
        except ValueError as exc:
            return str(exc)
        except WorkspaceWriteError as exc:
            return f"not promoted — {exc}"
        return (
            f"Promoted {entry_id} into MEMORY.md."
            if ok
            else f"No active entry {entry_id} on {date_str}."
        )

    async def workspace_rewrite(params: dict[str, Any]) -> str:
        file = str(params.get("file", "")).strip().lower()
        text = str(params.get("text", "")).strip()
        if file not in ("identity", "soul"):
            return (
                "workspace_rewrite can only replace identity or soul whole — "
                "use workspace_remember for a preference or fact, and ask the "
                "user to edit agents/user/memory directly for anything else."
            )
        if not text:
            return "workspace_rewrite needs text."
        try:
            workspace.write_raw(file, text)
        except SecretRejected as exc:
            return f"not saved: {exc}"
        except WorkspaceWriteError as exc:
            return f"not saved — {exc}"
        return f"{file}.md replaced."

    async def workspace_finish_bootstrap(params: dict[str, Any]) -> str:
        try:
            done = workspace.finish_bootstrap()
        except WorkspaceWriteError as exc:
            return f"not finished — {exc}"
        return "First-run setup marked complete." if done else "No first-run setup was pending."

    def _describe_trigger(trigger: dict[str, Any]) -> str:
        parts: list[str] = []
        if trigger.get("from"):
            parts.append(f"from {trigger['from']}")
        if trigger.get("source"):
            parts.append(f"source {trigger['source']}")
        if trigger.get("kind"):
            parts.append(f"kind {trigger['kind']}")
        if trigger.get("contains"):
            parts.append(f"containing {trigger['contains']!r}")
        return " and ".join(parts) or "any event"

    def _routine_id(params: dict[str, Any]) -> int | None:
        raw = str(params.get("routine_id", "")).strip()
        try:
            return int(raw)
        except ValueError:
            return None

    async def create_routine(params: dict[str, Any]) -> str:
        label = str(params.get("label", "")).strip()
        tool_name = str(params.get("tool_name", "")).strip()
        trigger = {
            key: str(value).strip()
            for key, value in (
                ("source", params.get("when_source")),
                ("kind", params.get("when_kind")),
                ("from", params.get("when_from")),
                ("contains", params.get("when_contains")),
            )
            if str(value or "").strip()
        }
        if not label:
            return "create_routine needs a label."
        if not tool_name:
            return "create_routine needs a tool_name — what to run when it triggers."
        if not trigger:
            return (
                "create_routine needs at least one condition: when_source, "
                "when_kind, when_from, or when_contains. A reaction to "
                "'every event' is not something the user would want."
            )
        cooldown = _parse_int_param(params, "cooldown_seconds", 300)
        if cooldown is None:
            return "cooldown_seconds must be an integer (seconds between fires)."
        routine = await routines.create(
            label=label,
            trigger=trigger,
            action={
                "type": "tool",
                "tool": tool_name,
                "params": dict(params.get("params") or {}),
            },
            cooldown_seconds=cooldown,
        )
        return (
            f"Routine {routine.id} '{label}' armed: when {_describe_trigger(trigger)}, "
            f"run {tool_name}. It still passes the authorization gate every "
            "time it fires — a risky tool will ask the user then."
        )

    async def list_routines(params: dict[str, Any]) -> str:
        rows = await routines.list()
        if not rows:
            return "No routines armed."
        lines = []
        for r in rows:
            fired = (
                f"fired {r.fire_count}x"
                + (f", last {r.last_fired_at:%Y-%m-%d %H:%M}" if r.last_fired_at else "")
                if r.fire_count
                else "never fired"
            )
            state = "paused" if not r.enabled else fired
            lines.append(
                f"#{r.id} '{r.label}' — when {_describe_trigger(r.trigger)} → "
                f"run {r.action.get('tool', '?')} ({state})"
            )
        return "\n".join(lines)

    async def set_routine_enabled(params: dict[str, Any]) -> str:
        routine_id = _routine_id(params)
        if routine_id is None:
            return "set_routine_enabled needs a routine_id."
        enabled = bool(params.get("enabled", True))
        routine = await routines.set_enabled(routine_id, enabled)
        if routine is None:
            return f"No routine {routine_id}."
        verb = "re-armed" if enabled else "paused"
        return f"Routine {routine_id} '{routine.label}' {verb}."

    async def delete_routine(params: dict[str, Any]) -> str:
        routine_id = _routine_id(params)
        if routine_id is None:
            return "delete_routine needs a routine_id."
        if await routines.delete(routine_id):
            return f"Routine {routine_id} deleted."
        return f"No routine {routine_id}."

    async def explain_decision(params: dict[str, Any]) -> str:
        raw_trace = str(params.get("trace_id", "")).strip()
        raw_approval = str(params.get("approval_id", "")).strip()
        about = str(params.get("about", "")).strip()
        limit = _parse_int_param(params, "limit", 3, minimum=1)
        if limit is None:
            limit = 3

        if raw_trace:
            try:
                trace_id = int(raw_trace)
            except ValueError:
                return f"trace_id {raw_trace!r} is not a number."
            trace = await traces.get(trace_id)
            if trace is None:
                return f"No recorded trace {trace_id}."
            return format_trace(trace)

        if raw_approval:
            try:
                approval_id = int(raw_approval)
            except ValueError:
                return f"approval_id {raw_approval!r} is not a number."
            origins: list[Any] = []  # the act that proposed the held call
            carries: list[Any] = []  # the approved call running
            for t in await traces.recent(50):
                p = t.payload or {}
                if p.get("approval_id") == approval_id:
                    carries.append(t)
                elif any(c.get("approval_id") == approval_id for c in p.get("calls") or []):
                    origins.append(t)
            if not origins and not carries:
                return f"No recorded trace involving approval #{approval_id}."
            return "\n\n".join(format_trace(t) for t in (origins + carries)[:4])

        if about:
            needle = about.lower()
            found = [
                t
                for t in await traces.recent(50)
                if needle
                in " ".join(
                    [
                        t.kind,
                        t.label or "",
                        json.dumps(t.payload or {}, default=str, ensure_ascii=False),
                    ]
                ).lower()
            ]
            if not found:
                return f"No recorded trace mentioning {about!r}."
            return "\n\n".join(format_trace(t) for t in found[:limit])

        rows = await traces.recent(limit)
        if not rows:
            return "No decision traces recorded yet."
        return "\n\n".join(format_trace(t) for t in rows)

    async def search_tools(params: dict[str, Any]) -> str:
        query = str(params.get("query", "")).strip()
        if not query:
            return "search_tools needs a query — e.g. an app or action like 'calendar' or 'create issue'."
        found = registry.search(query)
        if not found:
            return f"No tools match {query!r}. The app may not be connected — try the apps page."
        lines = [f"{s.name} — {_clip(s.description, 120)}" for s in found]
        return "Matching tools (now available to call directly this turn):\n" + "\n".join(lines)

    natives: list[tuple[ToolSpec, NativeHandler]] = [
        (
            _spec(
                "search_tools",
                "Find tools for an app or action that isn't in your current "
                "tool list — e.g. 'calendar', 'send message', 'create issue'. "
                "Matches become callable immediately after this returns.",
                {
                    "query": {
                        "type": "string",
                        "description": "app name or action to find tools for",
                    },
                },
                ["query"],
            ),
            search_tools,
        ),
        (
            _spec(
                "memory_search",
                "Search Aether's cross-app memory for past events, messages, "
                "screen captures, shared documents, and notes. Semantic — "
                "describe the topic, don't just repeat keywords. A hit from a "
                "document carries its bracketed id; pass that id as event_id "
                "to re-read the whole stored document, page by page.",
                {
                    "query": {"type": "string", "description": "what to look for"},
                    "limit": {"type": "integer", "description": "max results (default 5)"},
                    "event_id": {
                        "type": "integer",
                        "description": "re-read one whole stored document by its id",
                    },
                },
                [],
            ),
            memory_search,
        ),
        (
            _spec(
                "note_entity",
                "Keep a relationship note about a person in Aether's memory "
                "(e.g. 'owes me a reply', 'book club Fridays').",
                {
                    "handle": {"type": "string", "description": "email, @handle, or phone"},
                    "note": {"type": "string", "description": "what to remember"},
                    "platform": {"type": "string", "description": "platform hint, * for any"},
                    "display_name": {"type": "string", "description": "name, if known"},
                    "kind": {"type": "string", "description": "note category"},
                },
                ["handle", "note"],
            ),
            note_entity,
        ),
        (
            _spec(
                "get_pending_approvals",
                "List tool calls parked for the user's one-tap approval.",
                {},
                [],
            ),
            get_pending_approvals,
        ),
        (
            _spec(
                "schedule_action",
                "Run a tool call at a future time. Persisted — it survives "
                "restarts. The call still passes the authorization gate at "
                "fire time.",
                {
                    "label": {"type": "string", "description": "short human label"},
                    "run_at": {"type": "string", "description": "ISO datetime with timezone"},
                    "tool_name": {"type": "string", "description": "tool to run"},
                    "params": {"type": "object", "description": "arguments for the tool"},
                },
                ["label", "run_at", "tool_name"],
            ),
            schedule_action,
        ),
        (
            _spec(
                "request_screen_capture",
                "Ask the user's companion for a fresh screenshot (it becomes "
                "a screen_capture memory — perception only).",
                {"reason": {"type": "string", "description": "why you want a look"}},
                [],
            ),
            request_screen_capture,
        ),
        (
            _spec(
                "send_chat_message",
                "Push a message to the user's chat surfaces mid-turn "
                "(Telegram, Discord, Slack, web chat — whichever are "
                "enabled). Your end-of-turn reply is delivered "
                "automatically — never use this to answer or to repeat it.",
                {"text": {"type": "string", "description": "the message"}},
                ["text"],
            ),
            send_chat_message,
        ),
    ]
    if routines is not None:
        natives.extend(
            [
                (
                    _spec(
                        "create_routine",
                        "Arm a standing reaction: 'when X happens, run Y'. "
                        "Conditions (when_source, when_kind, when_from, "
                        "when_contains) are AND-combined; at least one is "
                        "required. Every fire passes the authorization gate.",
                        {
                            "label": {"type": "string", "description": "short human label"},
                            "when_source": {
                                "type": "string",
                                "description": "event source to match, e.g. mail",
                            },
                            "when_kind": {
                                "type": "string",
                                "description": "event kind, e.g. chat_message",
                            },
                            "when_from": {
                                "type": "string",
                                "description": "sender handle to match",
                            },
                            "when_contains": {
                                "type": "string",
                                "description": "text the event must contain",
                            },
                            "tool_name": {
                                "type": "string",
                                "description": "tool to run on a match",
                            },
                            "params": {"type": "object", "description": "arguments for that tool"},
                            "cooldown_seconds": {
                                "type": "integer",
                                "description": "minimum seconds between fires (default 300)",
                            },
                        },
                        ["label", "tool_name"],
                    ),
                    create_routine,
                ),
                (
                    _spec(
                        "list_routines",
                        "List the user's armed routines — trigger, action, "
                        "and how often each has fired.",
                        {},
                        [],
                    ),
                    list_routines,
                ),
                (
                    _spec(
                        "set_routine_enabled",
                        "Pause or re-arm a routine without deleting it.",
                        {
                            "routine_id": {"type": "integer", "description": "routine to toggle"},
                            "enabled": {"type": "boolean", "description": "false pauses it"},
                        },
                        ["routine_id"],
                    ),
                    set_routine_enabled,
                ),
                (
                    _spec(
                        "delete_routine",
                        "Delete one of the user's routines.",
                        {"routine_id": {"type": "integer", "description": "routine to delete"}},
                        ["routine_id"],
                    ),
                    delete_routine,
                ),
            ]
        )
    if traces is not None:
        natives.extend(
            [
                (
                    _spec(
                        "explain_decision",
                        "Replay why an action happened: the recorded trace of "
                        "what was seen, what was proposed, how the "
                        "authorization gate ruled, and what came back. Look it "
                        "up by trace_id, by the approval_id a call was held as, "
                        "or by a topic (about); with nothing given, the most "
                        "recent traces. Answer 'why did you do that?' from the "
                        "record, not from memory.",
                        {
                            "trace_id": {"type": "integer", "description": "a specific trace"},
                            "approval_id": {
                                "type": "integer",
                                "description": "the approval a call was held as",
                            },
                            "about": {
                                "type": "string",
                                "description": "topic to search recent traces for",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "max traces when listing (default 3)",
                            },
                        },
                        [],
                    ),
                    explain_decision,
                ),
            ]
        )
    if audit is not None:
        natives.extend(
            [
                (
                    _spec(
                        "verify_integrity",
                        "Verify the tamper-evident hash chain of the decision "
                        "record and answer in plain words. Use it whenever the "
                        "user asks whether your records can be trusted — 'can "
                        "anyone tamper with your logs?', 'prove nothing was "
                        "edited'.",
                        {},
                        [],
                    ),
                    verify_integrity,
                ),
            ]
        )
    if config_manager is not None:
        natives.extend(
            [
                (
                    _spec(
                        "get_config",
                        "Read Aether's current configuration in plain words: "
                        "the overview (with everything changed from "
                        "config.yaml marked), one section ('llm', 'agent', "
                        "'salience', 'messaging', 'contacts', 'authz', "
                        "'mcp_servers'), or one setting by its dotted path. "
                        "Use it whenever the user asks what's set, what a "
                        "setting is, or what they've changed from chat.",
                        {
                            "path": {
                                "type": "string",
                                "description": "optional section or dotted path, e.g. agent.tick_seconds",
                            },
                        },
                        [],
                    ),
                    get_config,
                ),
                (
                    _spec(
                        "set_config",
                        "Change a config.yaml setting from chat. op: set (a "
                        "value, or null to clear), add/remove (list entries), "
                        "reset (back to config.yaml). Paths: "
                        "agent.tick_seconds, agent.max_tool_iterations, "
                        "agent.daily_surface_cap, agent.quiet_hours, "
                        "agent.quiet_urgent_salience, salience.threshold, "
                        "salience.rate_cap_per_hour, llm.provider, llm.model, "
                        "llm.vision_model, llm.salience_model, llm.max_tokens, "
                        "llm.ollama_vision, messaging.<platform>.enabled, "
                        "contacts.mode, contacts.allowlist, authz.rules, "
                        "authz.approval_ttl_hours, mcp_servers, "
                        "mcp_servers.<name>.enabled. Tuning (agent.*, "
                        "salience.*) applies immediately; llm.* persists and "
                        "Aether restarts itself to load it; security sections "
                        "(contacts, authz, messaging, mcp_servers) are held "
                        "for the user's one-tap approval — when a call is "
                        "held, say so plainly and do not propose it again. "
                        "Secrets (.env) are not part of this vocabulary.",
                        {
                            "op": {
                                "type": "string",
                                "description": "set, add, remove, or reset",
                            },
                            "path": {
                                "type": "string",
                                "description": "the dotted setting path",
                            },
                            "value": {
                                "description": "the new value, or null to clear — "
                                "for add/remove the list entry, e.g. "
                                '{"platform": "telegram", "handle": "@friend"} '
                                "for contacts.allowlist",
                            },
                        },
                        ["op", "path"],
                    ),
                    set_config,
                ),
            ]
        )
    if workspace is not None:
        natives.extend(
            [
                (
                    _spec(
                        "workspace_read",
                        "Read one file from Aether's human-editable Markdown "
                        "workspace: identity, soul, agents, user, memory, "
                        "bootstrap (only present during first-run setup), or "
                        "daily (today's working notes, or an older date).",
                        {
                            "file": {
                                "type": "string",
                                "description": "identity | soul | agents | user | memory | bootstrap | daily",
                            },
                            "date": {
                                "type": "string",
                                "description": "YYYY-MM-DD, for file: daily — defaults to today",
                            },
                        },
                        ["file"],
                    ),
                    workspace_read,
                ),
                (
                    _spec(
                        "workspace_search",
                        "Search the whole workspace (identity, soul, agent "
                        "instructions, user preferences, long-term memory, "
                        "and recent daily notes) for a keyword or phrase.",
                        {
                            "query": {"type": "string", "description": "what to look for"},
                            "limit": {"type": "integer", "description": "max results (default 5)"},
                        },
                        ["query"],
                    ),
                    workspace_search,
                ),
                (
                    _spec(
                        "workspace_remember",
                        "Write a durable note to the workspace — sparingly; "
                        "most things belong in ordinary memory_search memory, "
                        "not here. kind=preference (USER.md, something the "
                        "user consistently wants — pass supersedes with any "
                        "entry ids it replaces), kind=fact (MEMORY.md, a "
                        "durable fact or decision; category groups related "
                        "facts, default 'Notes'), or kind=daily (today's "
                        "working notes, for context worth keeping only for "
                        "this session). Refuses anything that looks like a "
                        "secret, password, or API key.",
                        {
                            "text": {"type": "string", "description": "what to remember"},
                            "kind": {"type": "string", "description": "preference | fact | daily"},
                            "source": {
                                "type": "string",
                                "description": "user (they stated this directly) or agent "
                                "(you inferred it) — default agent",
                            },
                            "category": {
                                "type": "string",
                                "description": "fact only: section heading in MEMORY.md",
                            },
                            "confidence": {
                                "type": "number",
                                "description": "0-1, if this is an inference rather than something stated",
                            },
                            "supersedes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "preference only: entry ids this one replaces",
                            },
                        },
                        ["text", "kind"],
                    ),
                    workspace_remember,
                ),
                (
                    _spec(
                        "workspace_rewrite",
                        "Replace IDENTITY.md or SOUL.md wholesale — only "
                        "when the user explicitly asks you to change who "
                        "you are or how you sound. For anything else "
                        "(preferences, facts, daily notes) use "
                        "workspace_remember instead.",
                        {
                            "file": {"type": "string", "description": "identity | soul"},
                            "text": {
                                "type": "string",
                                "description": "the file's full new content",
                            },
                        },
                        ["file", "text"],
                    ),
                    workspace_rewrite,
                ),
                (
                    _spec(
                        "workspace_promote",
                        "Promote an entry from a daily working-notes file "
                        "into MEMORY.md, because it turned out to matter "
                        "beyond that day. The original stays on record, "
                        "marked promoted.",
                        {
                            "date": {
                                "type": "string",
                                "description": "YYYY-MM-DD the entry was written on",
                            },
                            "entry_id": {
                                "type": "string",
                                "description": "the entry's id, from workspace_search or workspace_read",
                            },
                            "category": {
                                "type": "string",
                                "description": "section heading in MEMORY.md, default 'Notes'",
                            },
                        },
                        ["date", "entry_id"],
                    ),
                    workspace_promote,
                ),
                (
                    _spec(
                        "workspace_finish_bootstrap",
                        "Mark first-run workspace setup complete, once "
                        "you've introduced yourself and learned the user's "
                        "basic preferences. Removes BOOTSTRAP.md.",
                        {},
                        [],
                    ),
                    workspace_finish_bootstrap,
                ),
            ]
        )
    for spec, handler in natives:
        registry.add_native(spec, handler)
    return len(natives)
