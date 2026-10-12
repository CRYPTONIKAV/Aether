"""The continuous loop: watch feeds, remember, reason, act — with a human
in the path for anything risky.

One background task. Each tick it (a) fires due MCP source polls (the
user's config.yaml poll_tools — a standing instruction, so these run
directly and are audited), (b) turns inbound chat into memory, (c) scores
everything new for salience, (d) checks the user's standing routines
against the new events — deterministic matching, with the taught action
passing the gate at fire time — and (e) if anything actually happened,
runs one authz-gated LLM turn and fans the reply out to every enabled
surface.

Between ticks it sleeps `agent.tick_seconds`, waking early the moment work
arrives (a message, a screen capture, a wake() from the API).

The executor is the single choke point: every ToolCall the model proposes,
every scheduled action when it fires, and every routine when it triggers,
goes through `classify` → run / park / deny, and lands in the hash-chained
audit log either way. And every act leaves a trace — what triggered it,
what the gate ruled, what came back — so "why did you do that?" always
has a record to answer from.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..authz.approvals import APPROVED, DENIED, Approval, Approvals
from ..authz.audit import AuditLog, verification_text
from ..authz.policy import Decision, Policy
from ..composio_bridge import SERVER_ENTRY, SERVER_NAME, ComposioBridge
from ..config import AppConfig, PollTool
from ..config_store import ConfigManager
from ..config_store import actions_word as _actions_word
from ..connectors.attachments import AttachmentProcessor, attachments_note
from ..connectors.base import (
    ConnectorUnavailableError,
    InboundMessage,
    MessagingConnector,
    UnknownToolError,
)
from ..connectors.mcp_host import MCPHost
from ..connectors.registry import ToolRegistry
from ..llm.agent import run_tool_loop
from ..llm.registry import ProviderRegistry
from ..llm.types import PLAIN_PARAM, Message, ToolCall, ToolResult, ToolSpec
from ..memory.context import ContextBuilder
from ..memory.entities import Sender
from ..memory.events import Event, EventStore
from ..memory.salience import Salience
from ..memory.vectors import MemoryHit, VectorMemory
from ..routines import Routines, trigger_matches
from ..scheduler.jobs import ScheduledAction, Scheduler
from ..secret_env import EnvResolver
from ..secret_store import SecretStore
from .apps_wizard import AppsWizard
from .config_wizard import ConfigWizard
from .config_wizard import coerce_config_value as _coerce_config_value
from .prompts import SYSTEM_PROMPT
from .settings import AgentSettings
from .tools import (
    FAILURE_REPLY,
    describe_ask,
    describe_call,
    describe_dismissed,
    describe_outcome,
    plain_outcome,
    plain_replay,
)
from .traces import CARRY_OUT, ROUTINE, SCHEDULED, TURN, Traces

log = logging.getLogger("aether.agent")

_MAX_OBSERVATIONS = 20

# Hub watchdog: a connection hub (composio) whose session exists but whose
# endpoint has stopped answering tools/list flaps in a reconnect loop
# forever — sessions.use still succeeds for it, so the bridge's reuse path
# never escapes. After this much continuous unreadiness, the loop cuts one
# fresh session (the host's next reconnect re-resolves the $NAME refs onto
# it), then leaves it alone for the cooldown; a few fruitless resets per
# boot mean the outage is Composio's, not the session's, and the normal
# reconnect loop is what rides it out.
_HUB_RESET_AFTER_SECONDS = 90.0  # boot grace + a few backoff cycles
_HUB_RESET_COOLDOWN_SECONDS = 600.0
_HUB_MAX_RESETS_PER_BOOT = 3

# History windowing (token control): a turn's prompt is capped at this many
# messages — the context block and the newest triggers always stay, the
# oldest middle entries fall off first. The tool loop's own back-and-forth
# appends after this, so it isn't truncated mid-flight.
_MAX_HISTORY_MESSAGES = 12
# ~4 chars/token; 12k chars ≈ 3k tokens of trigger text per call
_MAX_HISTORY_CHARS = 12_000

# how much of the cross-surface transcript (chat_messages, both directions)
# is offered to each turn as conversation memory. The window above trims it
# oldest-first under pressure — events and the fresh inbound always survive.
_TRANSCRIPT_MESSAGES = 20


def _window_history(messages: list[Message], keep_head: int = 1) -> list[Message]:
    """Bound a turn's input history so repeated LLM calls stay cheap.

    Pins the first `keep_head` messages (the context block, when one was
    prepended — without one, nothing is pinned, or a stale event would
    survive every window) plus the newest messages; drops oldest middle
    entries until both the count and the rough char budget fit."""
    if len(messages) <= _MAX_HISTORY_MESSAGES:
        windowed = list(messages)
    else:
        windowed = messages[:keep_head] + messages[-(_MAX_HISTORY_MESSAGES - keep_head) :]

    def size(ms: list[Message]) -> int:
        return sum(len(m.text) for m in ms)

    while len(windowed) > keep_head + 1 and size(windowed) > _MAX_HISTORY_CHARS:
        del windowed[keep_head]  # oldest after the pinned head
    return windowed


# the words that make a reply a question about the message it answers —
# anything else replies normally and the model handles it
WHY_REPLY_WORDS = frozenset({"why", "explain"})

# consecutive failed polls before a connector is announced as down. The user
# hears about the state once, and once again when it recovers — never per
# failure. A server with no configured poll has no heartbeat to watch.
POLL_FAILURES_BEFORE_DOWN = 2

# which .env names each messaging surface needs — /apps says "no token in
# .env yet" until every one of them is there
_PLATFORM_TOKENS = {
    "telegram": ("TELEGRAM_BOT_TOKEN",),
    "discord": ("DISCORD_BOT_TOKEN",),
    "slack": ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"),
}


def _parse_quiet_hours(raw: str | None) -> tuple[int, int] | None:
    """Parse "HH:MM-HH:MM" (local time) into minutes since midnight. The
    window may wrap midnight ("23:00-08:00"); a malformed value is logged
    and ignored — a budget the config file can't explain is worse than none."""
    if not raw:
        return None
    start_text, sep, end_text = raw.partition("-")
    try:
        if not sep:
            raise ValueError(f"quiet_hours {raw!r} must look like 'HH:MM-HH:MM'")
        start_h, start_m = (int(part) for part in start_text.strip().split(":"))
        end_h, end_m = (int(part) for part in end_text.strip().split(":"))
        window = (start_h * 60 + start_m, end_h * 60 + end_m)
        if not all(0 <= m < 24 * 60 for m in window):
            raise ValueError(f"quiet_hours {raw!r} is outside a day")
        return window
    except ValueError as exc:
        log.warning("%s — quiet hours disabled", exc)
        return None


class CaptureRequestBox:
    """A one-slot mailbox: the agent asks for a screenshot, the companion
    (polling GET /api/screen-request) takes the request and captures."""

    def __init__(self) -> None:
        self._note: str | None = None

    def set(self, note: str = "") -> None:
        self._note = note.strip()

    def take(self) -> str | None:
        note = self._note
        self._note = None
        return note


class SurfaceFanout:
    """Everything the agent says reaches every enabled surface. One dying
    surface never takes the loop down with it."""

    def __init__(
        self,
        connectors: list[MessagingConnector] | None = None,
        hub: Any = None,
        history: Any = None,
    ) -> None:
        self.connectors = list(connectors or [])
        self.hub = hub
        self.history = history

    def add_connectors(self, connectors: list[MessagingConnector]) -> None:
        self.connectors.extend(connectors)

    def remove_connector(self, name: str) -> MessagingConnector | None:
        """Take one connector out of the fanout by name — a surface the
        user just switched off. Stopping it is the caller's job; this stays
        sync like the rest of the fanout."""
        for i, connector in enumerate(self.connectors):
            if connector.name == name:
                del self.connectors[i]
                return connector
        return None

    async def send_to_user(self, text: str) -> dict[str, str]:
        """Fan out to every enabled surface, collecting where the words
        landed: {connector name: platform message id}. Those ids are what
        link a user's later "why?" back to the trace of this send. The hub
        has no id, and a dying surface never takes the loop down — its
        slot is just missing from the map."""
        refs: dict[str, str] = {}
        if self.hub is not None:
            try:
                await self.hub.broadcast(text)
            except Exception:
                log.exception("web chat broadcast failed")
        for connector in self.connectors:
            try:
                ref = await connector.send_to_user(text)
            except Exception:
                log.exception("%s send failed — continuing", connector.name)
                continue
            if ref:
                refs[connector.name] = ref
        # everything that actually landed joins the transcript so the next
        # turn can see its own words; the hub persists the web line itself
        # (broadcast), so only connector surfaces are written here
        if self.history is not None:
            for name in refs:
                try:
                    await self.history.append(name, "out", text)
                except Exception:
                    log.exception("transcript write failed for %s — continuing", name)
        return refs

    async def present_approval(self, approval_id: int, text: str) -> None:
        """One plain-language question to every surface — "Send the email
        (to rahul@gmail.com)?" Raw tool names and argument dumps never
        reach the user; the decision record keeps the exact call."""
        for connector in self.connectors:
            try:
                await connector.present_approval(approval_id, text)
            except Exception:
                log.exception("%s approval presentation failed", connector.name)
        if not self.connectors and self.hub is None:
            log.warning("approval #%d has no surface to appear on", approval_id)


def _delivered_by_tool(calls: list[dict[str, Any]]) -> bool:
    """True when the turn already put words in front of the user through
    send_chat_message — a send that actually delivered (allowed, no error,
    non-empty text), not one that was denied, parked, or refused."""
    for call in calls:
        if call.get("name") != "send_chat_message":
            continue
        if call.get("decision") != "allow" or call.get("is_error"):
            continue
        if not str(call.get("params", {}).get("text", "")).strip():
            continue  # the handler refuses empty text; nothing went out
        return True
    return False


# the value sentinel: "not on the line" — distinct from None, which is an
# explicit clear (llm.vision_model, agent.quiet_hours)
_MISSING = object()

_CONFIG_USAGE = (
    "Aether /config — the guided walk: send just /config and answer the questions.\n"
    "Everything below is the one-line way to do the same.\n\n"
    "/config show — the overview, with everything changed from chat marked\n"
    "/config show <section|path> — one section's settings, or one value\n"
    "/config set <path> <value> — e.g. /config set agent.tick_seconds 10\n"
    "/config add <path> <value…> — contacts.allowlist <platform> <handle> · "
    "authz.rules <pattern> <decision> [note] · mcp_servers {json object}\n"
    "/config remove <path> <value…> — allowlist [platform] <handle> · "
    "authz.rules <pattern> · mcp_servers <name>\n"
    "/config reset <path> — back to config.yaml's value\n\n"
    "Personal tuning (agent.*, salience.*, llm.*) applies right away; llm.* "
    "restarts me to load it. Security sections (contacts, authz, messaging, "
    "mcp_servers) wait for your one-tap approval. Keys paste in /apps — "
    "/config never sets one."
)


def _parse_config_command(text: str) -> tuple[str, str | None, Any]:
    """(op, path, value) from a /config line; op "usage" for anything the
    grammar doesn't recognize. Deliberately dumb — every meaning lives in
    ConfigManager, so the command path and the tool path share one
    interpretation and can never drift."""
    parts = text.split(maxsplit=2)
    if len(parts) == 1:
        # bare /config — the guided walk when the manager is wired; the
        # parser still reads it as the overview for the unwired answer
        return "show", None, _MISSING
    sub = parts[1].lower()
    if sub == "show":
        shown = parts[2].strip() or None if len(parts) > 2 else None
        return "show", shown, _MISSING
    if sub not in ("set", "add", "remove", "reset"):
        return "usage", None, _MISSING
    if len(parts) < 3:
        return sub, None, _MISSING
    rest = parts[2].strip()
    path, _, tail = rest.partition(" ")
    if not path:
        return "usage", None, _MISSING
    if sub == "reset":
        return "reset", path, _MISSING
    tail = tail.strip()
    if not tail:
        return sub, path, _MISSING
    return sub, path, _coerce_config_value(tail)


class AgentLoop:
    def __init__(
        self,
        *,
        providers: ProviderRegistry | None,
        tools: ToolRegistry,
        policy: Policy,
        approvals: Approvals,
        audit: AuditLog,
        events: EventStore,
        salience: Salience,
        context: ContextBuilder,
        scheduler: Scheduler,
        surfaces: SurfaceFanout,
        capture_box: CaptureRequestBox,
        config: AppConfig,
        host: MCPHost | None = None,
        routines: Routines | None = None,
        traces: Traces | None = None,
        agent_settings: AgentSettings | None = None,
        config_manager: ConfigManager | None = None,
        workspace: Any = None,
        resolver: EnvResolver | None = None,
        secret_store: SecretStore | None = None,
        settings: Any = None,
        composio: ComposioBridge | None = None,
        transcript: Any = None,
        attachments: AttachmentProcessor | None = None,
        vectors: VectorMemory | None = None,
    ) -> None:
        self._providers = providers
        self._tools = tools
        self._policy = policy
        self._approvals = approvals
        self._audit = audit
        self._events = events
        self._salience = salience
        self._context = context
        self._scheduler = scheduler
        self._surfaces = surfaces
        self._capture_box = capture_box
        self._config = config
        self._host = host
        self._routines = routines
        self._traces = traces
        self._agent_settings = agent_settings
        self._config_manager = config_manager
        self._workspace = workspace
        self._resolver = resolver
        self._secret_store = secret_store
        self._settings = settings
        self._composio = composio
        # the cross-surface transcript (chat_messages) — both directions,
        # read back each turn as conversation memory
        self._transcript = transcript
        # shared files: images described, documents read — past the
        # allowlist, in the drain, so a filtered sender's files are never seen
        self._attachments = attachments
        # the vector half of memory: chunked, embedded, sourced — retrieval
        # is semantic and automatic; a missing embedder means no recall,
        # and memory_search says the plain failure message
        self._vectors = vectors
        # the guided /config walk in progress, if any — memory only
        self._wizard: ConfigWizard | None = None
        # the guided /apps walk in progress, if any — same rules, memory only
        self._apps_walk: AppsWizard | None = None
        # mcp servers whose "I'll tell you the moment it's up" was already
        # said this boot — a flapping server never announces twice
        self._announced_ready: set[str] = set()
        # the tool_version each server's namespace was last synced at — a
        # connection hub grows as the user approves apps, and this is how a
        # tick notices without diffing full tool lists every time
        self._synced_tool_versions: dict[str, int] = {}
        # in-flight Connect Link waits — tracked so a completed task is
        # never garbage-collected mid-flight
        self._connect_waits: set[asyncio.Task[None]] = set()
        # hub watchdog state: when the composio connection first went
        # unready (monotonic), when a session reset last ran, and how many
        # this boot — see _watch_hub
        self._hub_unready_since: float | None = None
        self._hub_last_reset: float | None = None
        self._hub_resets = 0

        self._queue: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self._woken = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        # the vector backfill loop's task — None until start() arms it
        self._backfill_task: asyncio.Task[None] | None = None
        self._running = False
        self._last_event_id = 0
        self._poll_targets: dict[str, tuple[str, PollTool]] = {}
        self._next_poll: dict[str, datetime] = {}
        self._poll_failures: dict[str, int] = {}
        self._down: set[str] = set()
        self._quiet = _parse_quiet_hours(config.agent.quiet_hours)
        self._held: list[str] = []  # unprompted notes waiting out the window
        self.rebuild_poll_targets()

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._running = True
            self._task = asyncio.create_task(self.run(), name="agent-loop")
            # catch-up indexing runs beside the answering loop, never on
            # it — a turn never waits on history draining
            if self._vectors is not None and self._vectors.enabled:
                self._backfill_task = asyncio.create_task(
                    self._backfill_loop(), name="vector-backfill"
                )

    async def stop(self) -> None:
        self._running = False
        task, self._task = self._task, None
        backfill, self._backfill_task = self._backfill_task, None
        self._woken.set()
        if backfill is not None:
            backfill.cancel()
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(backfill, timeout=10)
        if task is not None:
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=10)

    async def _backfill_loop(self) -> None:
        """Index the events that predate vector memory (or lost a race with
        a rate limit), 50 oldest at a time: tight passes while history is
        draining, a minute's rest once caught up. A failed pass just
        retries — indexed_at is the ledger, so nothing is ever lost."""
        assert self._vectors is not None
        while self._running:
            try:
                handled = await self._vectors.backfill(50)
            except Exception:
                log.exception("vector backfill pass failed — pausing before the retry")
                handled = None
            await asyncio.sleep(1 if handled else 60)

    def notify(self) -> None:
        """Wake the loop early — used after screen captures, poll results,
        anything the API layer knows the agent will want to see now."""
        self._woken.set()

    def submit_message(self, message: InboundMessage) -> None:
        """Queue an inbound chat message — sync, for the WebSocket path and
        tests. Connectors get the awaitable handle_inbound instead, because
        the InboundHandler contract they call through is async."""
        self._queue.put_nowait(message)
        self._woken.set()

    async def handle_inbound(self, message: InboundMessage) -> None:
        """Connector-facing inbound entry — satisfies the async InboundHandler
        contract (connectors await their handler; a sync return is the
        "'NoneType' object can't be awaited" crash)."""
        self.submit_message(message)

    # -- the interruption budget ------------------------------------------------

    def _quiet_now(self) -> bool:
        """True when local time is inside the quiet window — the user means
        *their* night, so this reads the clock naively."""
        if self._quiet is None:
            return False
        now = datetime.now()
        minutes = now.hour * 60 + now.minute
        start, end = self._quiet
        if start == end:
            return True  # a degenerate window is a 24h hold — config's choice
        if start < end:
            return start <= minutes < end
        return minutes >= start or minutes < end  # wraps midnight

    def reparse_quiet_hours(self) -> None:
        """A chat-made change to agent.quiet_hours, applied now — the
        window is parsed once at boot, so the parse has to be redone on
        change (a malformed value refuses at the config layer, never
        here)."""
        self._quiet = _parse_quiet_hours(self._config.agent.quiet_hours)

    async def _notify(self, text: str, *, urgent: float = 0.0) -> dict[str, str] | None:
        """Send an unprompted note under the interruption budget: during
        quiet hours it waits (batched, shipped when the window ends) unless
        it is urgent enough on the 0-10 salience scale. Solicited words —
        replies to the user, approvals, scheduled deliveries — never come
        through here. Returns where the words landed, or None when they
        are being held (or empty) — held words haven't reached a surface,
        so there is nothing to link yet."""
        if not text.strip():
            return None
        if self._quiet_now() and urgent < self._config.agent.quiet_urgent_salience:
            self._held.append(text)
            log.info("held for quiet hours: %s", text[:80])
            return None
        return await self._surfaces.send_to_user(text)

    # -- the loop ---------------------------------------------------------------

    async def run(self) -> None:
        self._last_event_id = await self._events.max_id()
        log.info(
            "agent loop started (tick=%ss, last event id %d)",
            self._config.agent.tick_seconds,
            self._last_event_id,
        )
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("agent tick failed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._woken.wait(), timeout=self._config.agent.tick_seconds)
            self._woken.clear()

    async def _tick(self) -> None:
        anchor = self._last_event_id

        # (a) due source polls first, so their results are observed this tick
        await self._run_due_polls()

        # (a2) a server that answered late joins the namespace here —
        # keeping the "the moment it's up" promise is the tick's own job
        await self._reconcile_mcp()

        # (a3) a hub whose endpoint has gone stale never gets ready on its
        # own — the watchdog cuts it a fresh session
        await self._watch_hub()

        # (a4) pending approvals past their TTL flip to expired and are
        # audited — without this they'd sit as 'pending' forever, and every
        # pending-approvals reader would keep offering dead decisions
        if self._approvals is not None:
            try:
                await self._approvals.expire_overdue()
            except Exception:
                log.exception("expiring overdue approvals failed — continuing")

        # (b) inbound chat becomes memory (allowlist applies at ingest; a
        # dropped sender is never seen by the model at all)
        drained: list[InboundMessage] = []
        while not self._queue.empty():
            drained.append(self._queue.get_nowait())
        messages: list[InboundMessage] = []
        for message in drained:
            # the deterministic vocabulary answers here, before anything is
            # ingested or any model is woken: a /verify badge and a why?
            # replay cost no tokens and store no memory — they are reads,
            # answered from the record
            if await self._try_chat_command(message):
                continue
            if await self._try_explain_reply(message):
                continue
            result = await self._events.ingest(
                source=message.surface,
                kind="chat_message",
                payload={"text": message.text, "chat_ref": message.chat_ref},
                sender=Sender(platform=message.surface, handle=message.handle),
            )
            if result.stored and result.event_id is not None:
                await self._salience.score_event(result.event_id)
                self._last_event_id = max(self._last_event_id, result.event_id)
            if result.reason == "filtered:contacts":
                log.info(
                    "inbound from %s on %s filtered by the allowlist",
                    message.handle,
                    message.surface,
                )
                continue
            # (b2) files the user shared: described (images) or extracted
            # (documents) into memory here — past the allowlist, before the
            # observation list is built, so they reach this same tick's turn
            if self._attachments is not None and message.attachments:
                try:
                    results = await self._attachments.process(message)
                except Exception:
                    log.exception("attachment processing failed — continuing")
                    results = []
                for one in results:
                    if (
                        one.ingested is not None
                        and one.ingested.stored
                        and one.ingested.event_id is not None
                    ):
                        self._last_event_id = max(self._last_event_id, one.ingested.event_id)
                note = attachments_note(results)
                if note:
                    # solicited — the user just shared something, so it never
                    # waits out quiet hours
                    await self._surfaces.send_to_user(note)
            # transcript write happens only past the allowlist — a filtered
            # sender must never reach conversation memory, or their words
            # would leak into a later prompt. Web inbound is persisted by
            # the websocket handler itself, so it's skipped here.
            if self._transcript is not None and message.surface != "web":
                try:
                    await self._transcript.append(message.surface, "in", message.text)
                except Exception:
                    log.exception("transcript write failed for inbound %s", message.surface)
            messages.append(message)

        # (c) everything else ingested since last tick becomes observations
        observations = await self._events.list_since(anchor, limit=_MAX_OBSERVATIONS)
        for event in observations:
            self._last_event_id = max(self._last_event_id, event.id)
            await self._salience.score_event(event.id)

        # (c2) standing triggers the user taught — deterministic matching
        # over the same observations, before any LLM is involved
        if observations and self._routines is not None:
            await self._run_routines(observations)

        # held unprompted notes from the quiet window ship as one batched
        # message the moment the window ends — the budget pays out, in one
        # interruption instead of a drip
        if self._held and not self._quiet_now():
            held, self._held = self._held, []
            await self._surfaces.send_to_user("While it was quiet:\n\n" + "\n\n".join(held))

        if not messages and not observations:
            return  # a quiet tick: no LLM turn, no tokens spent

        await self._turn(messages, observations)

    # -- the deterministic chat vocabulary ---------------------------------------

    async def _try_chat_command(self, message: InboundMessage) -> bool:
        """Answer the deterministic vocabulary straight from the record or
        the live state — no LLM turn, no tokens, nothing ingested, no
        trace. A /config walk in progress answers here too: its questions
        and replies are part of the same vocabulary. Returns True when the
        message was a handled command."""
        text = message.text.strip()
        first = text.split(maxsplit=1)[0].lower() if text else ""
        if first == "/config":
            # bare /config is the friendly front door: the guided walk. A
            # mid-walk /config starts it over, and an expert line replaces
            # any walk in progress — the user spoke expert, honor it.
            if len(text.split()) == 1 and self._config_manager is not None:
                self._apps_walk = None  # one walk at a time
                self._wizard = ConfigWizard(
                    self._config,
                    apply=self._apply_config,
                    show=self._config_manager.show,
                    yaml_value=self._config_manager.yaml_value,
                )
                await self._surfaces.send_to_user(self._wizard.start())
                return True
            self._wizard = None
            return await self._config_command(text)
        if self._wizard is not None:
            if text.startswith("/"):
                self._wizard = None  # a command ends the walk quietly
            else:
                wizard = self._wizard
                reply = await wizard.handle(text)
                if reply is None:
                    # the walk let go of the message — it's normal chat
                    self._wizard = None
                    return False
                if not wizard.alive:
                    self._wizard = None  # goodbye — no menu is waiting
                await self._surfaces.send_to_user(reply)
                return True
        if first == "/apps":
            # the app front door: bare /apps is the status, /apps add <name>
            # starts the connect walk. Either replaces a live config walk,
            # as a bare /config replaces an apps one.
            self._wizard = None
            self._apps_walk = self._new_apps_wizard()
            parts = text.split(maxsplit=2)
            if len(parts) >= 2 and parts[1].lower() == "add":
                name = parts[2] if len(parts) > 2 else ""
                reply = await self._apps_walk.start_add(name)
            else:
                reply = await self._apps_walk.start()
            await self._surfaces.send_to_user(reply)
            self._maybe_spawn_connect_waiter()
            if not self._apps_walk.alive:
                self._apps_walk = None
            return True
        if self._apps_walk is not None:
            if text.startswith("/"):
                self._apps_walk = None  # a command ends the walk quietly
            else:
                walk = self._apps_walk
                reply = await walk.handle(text)
                if reply is None:
                    # the walk let go of the message — it's normal chat
                    self._apps_walk = None
                    return False
                self._maybe_spawn_connect_waiter()
                if not walk.alive:
                    self._apps_walk = None  # done, or goodbye — nothing is waiting
                await self._surfaces.send_to_user(reply)
                return True
        if text.lower() != "/verify":
            return False
        try:
            verification = await self._audit.verify_chain()
            newest = None
            if verification.entries:
                rows = await self._audit.recent(1)
                newest = rows[0]["created_at"] if rows else None
            # the render sits inside the same guard: a bug in the badge
            # wording must degrade to the fallback note, not kill the tick
            # and take every other batched message down with it
            text = verification_text(verification, newest)
        except Exception:
            log.exception("/verify could not read the decision record")
            await self._surfaces.send_to_user(
                "I couldn't verify the record just now — the decision "
                "log isn't answering. Nothing else is affected."
            )
            return True
        # solicited — the user asked for the check, so it never waits out
        # quiet hours
        await self._surfaces.send_to_user(text)
        return True

    async def _config_command(self, text: str) -> bool:
        """The /config command, deterministic end to end: no ingest, no
        model, no tokens. `show` is a read answered from the live config;
        every write goes through the executor as a synthetic set_config
        call — the same choke point as the model's proposals, so a
        security path parks for one-tap approval exactly like any other
        gated action."""
        try:
            reply = await self._run_config_command(text)
        except Exception:
            log.exception("/config failed")
            reply = "that didn't work — the configuration is untouched. Nothing else is affected."
        # solicited — the user asked, so it never waits out quiet hours
        await self._surfaces.send_to_user(reply)
        return True

    async def _run_config_command(self, text: str) -> str:
        if self._config_manager is None:
            return "config management isn't wired on this instance."
        op, path, value = _parse_config_command(text)
        if op == "usage":
            return _CONFIG_USAGE
        if op == "show":
            return await self._config_manager.show(path)
        if value is _MISSING and op in ("set", "add", "remove"):
            if op == "set":
                return (
                    "set needs a value — e.g. /config set agent.tick_seconds 10 "
                    "(or the word none to clear a setting)."
                )
            return (
                f"{op} needs a value after the path — e.g. "
                f"/config {op} contacts.allowlist telegram @friend."
            )
        return await self._apply_config(op, path, value)

    async def _apply_config(
        self, op: str, path: str, value: Any = _MISSING, origin: str | None = None
    ) -> str:
        """One config change through the executor — the same synthetic
        set_config call for both /config front doors (the guided walk and
        the one-line grammar), so a write is classified, audited, parked or
        applied exactly like a model proposal. `value` absent means reset,
        which carries none. `origin` marks a connect the user just drove in
        chat ("walk") — the /apps pick-and-paste applies directly, where
        /config lines and model proposals still park."""
        params: dict[str, Any] = {"op": op, "path": path}
        if value is not _MISSING:
            params["value"] = value
        calls: list[dict[str, Any]] = []
        result = await self._execute(
            ToolCall(id="cmd-config", name="set_config", arguments=params),
            trace=calls,
            origin=origin,
        )
        held = calls[0].get("approval_id") if calls else None
        if held is not None:
            return (
                f"that one's security-shaped — held for your one-tap "
                f"approval (#{held}). Tap approve and it's done."
            )
        return result.content

    # -- /apps: the status render and the connect plumbing ---------------------------

    async def _env_names(self) -> set[str]:
        """Which secret names exist right now — names only, never values:
        pastes, .env, the real environment. An unwired resolver (a bare
        loop, tests) reads as "nothing set yet", the honest answer for a
        loop that can't check."""
        if self._resolver is None:
            return set()
        return await self._resolver.known_names()

    async def _store_secret(self, name: str, value: str, app: str) -> bool:
        """One pasted key into the encrypted store. Called from the
        deterministic command path — before ingest — so the value is
        consumed here and never reaches the model, memory, or the audit
        record, which carries the name and the app only. False when this
        loop has no store wired, and the walk answers honestly below."""
        if self._secret_store is None:
            return False
        await self._secret_store.set(name, value)
        await self._audit.append(
            actor="owner",
            tool_name="store_app_secret",
            decision="allow",
            rules_matched="builtin:internal",
            params={"name": name, "app": app},
            outcome="stored encrypted — value never recorded",
        )
        return True

    async def _apps_status(self) -> str:
        """What's connected, honestly — the same render at bare /apps and
        after a connect. A messaging surface that's on without its token
        says so; a hub app names its identity (or its non-active status);
        an mcp server counts its actions when it has answered and makes
        the promise when it hasn't. The footer is the gate, in one line —
        connection status and what Aether may do are separate truths."""
        names = await self._env_names()
        lines: list[str] = []
        messaging = self._config.messaging
        for platform in ("telegram", "discord", "slack"):
            if not getattr(messaging, platform).enabled:
                continue
            if all(name in names for name in _PLATFORM_TOKENS[platform]):
                lines.append(f"· {platform} — on")
            else:
                lines.append(f"· {platform} — on — no token yet")
        if self._composio is not None:
            try:
                apps = await self._composio.accounts()
            except Exception:
                log.exception("reading the hub's connected apps failed")
                apps = []
            for app in apps:
                if app.status == "ACTIVE":
                    who = f" — {app.identity}" if app.identity else ""
                    lines.append(f"· {app.toolkit}{who} — connected")
                else:
                    lines.append(
                        f"· {app.toolkit} — {app.status.lower()} — "
                        f"/apps add {app.toolkit} to reconnect"
                    )
        status = self._tools.mcp_status()
        for server in self._config.mcp_servers:
            if not server.enabled:
                continue
            if status.get(server.name):
                count = self._tools.server_action_count(server.name)
                lines.append(f"· {server.name} — connected · {_actions_word(count)}")
            else:
                lines.append(
                    f"· {server.name} — still connecting — I'll tell you the moment it's up"
                )
        head = "your apps:\n" + "\n".join(lines) if lines else "your apps: nothing connected yet."
        return f"{head}\nthe gate: reads run free — sends, deletes and anything new ask first."

    def _new_apps_wizard(self) -> AppsWizard:
        """Every walk wires the same loop functions — the connect command
        path and the background waiter's permissions walk alike."""
        return AppsWizard(
            apply=self._apply_config,
            status=self._apps_status,
            env_names=self._env_names,
            save_secret=self._store_secret,
            find_toolkits=self._find_toolkits,
            authorize=self._authorize_composio,
        )

    async def _find_toolkits(self, query: str) -> list[tuple[str, str]]:
        """The hub's live catalog, fuzzy-matched — [] when there's no hub
        wired, which the walk reads as 'nothing by that name'."""
        if self._composio is None:
            return []
        try:
            found = await self._composio.find_toolkits(query)
        except Exception:
            log.exception("toolkit lookup failed")
            return []
        return [(t.slug, t.name) for t in found]

    async def _authorize_composio(self, toolkit: str) -> tuple[str, str] | None:
        """Start one app's Connect Link. The first connect also pins the
        composio mcp_servers entry — the walk's pick is the approval, so
        it rides origin "walk" through the same executor; a user rule that
        parks it anyway gets its card relayed and the link still served
        (the tools arrive when the one-tap lands)."""
        bridge = self._composio
        if bridge is None:
            return None
        try:
            entry = next((s for s in self._config.mcp_servers if s.name == SERVER_NAME), None)
            reply = ""
            if entry is None:
                reply = await self._apply_config(
                    "add", "mcp_servers", dict(SERVER_ENTRY), origin="walk"
                )
            elif not entry.enabled:
                reply = await self._apply_config(
                    "set", f"mcp_servers.{SERVER_NAME}.enabled", True, origin="walk"
                )
            # the held-for-approval reply must reach the user directly —
            # checked by its stable words, never by a symbol
            if "held for your one-tap approval" in reply:
                await self._surfaces.send_to_user(reply)
            return await bridge.authorize(toolkit)
        except Exception:
            log.exception("starting the %s connect failed", toolkit)
            return None

    def _maybe_spawn_connect_waiter(self) -> None:
        """The walk just served a Connect Link — wait for the click out of
        band so the user isn't stuck watching. The walk hands over
        (toolkit, request id) exactly once."""
        walk = self._apps_walk
        if walk is None or walk.pending_connect is None:
            return
        toolkit, request_id = walk.pending_connect
        walk.pending_connect = None
        self.watch_connect(toolkit, request_id)

    def watch_connect(self, toolkit: str, request_id: str) -> None:
        """A Connect Link waiting on its click — from the chat walk or the
        web panel alike; the same out-of-band wait, the same announcement."""
        if self._composio is None:
            return
        task = asyncio.create_task(
            self._wait_for_connect(toolkit, request_id), name="composio-connect"
        )
        self._connect_waits.add(task)
        task.add_done_callback(self._connect_waits.discard)

    async def _wait_for_connect(self, toolkit: str, request_id: str) -> None:
        """The out-of-band half of a Connect Link: block until the click
        lands (or the link dies), then ask the gate's question — what may
        Aether do with the new app. Solicited by the user's own click, so
        the answer never waits out quiet hours."""
        bridge = self._composio
        assert bridge is not None
        try:
            app = await bridge.wait_and_enable(toolkit, request_id)
        except Exception:
            log.exception("the %s connect never completed", toolkit)
            await self._surfaces.send_to_user(
                f"the {toolkit} connection didn't finish — the link may "
                f"have expired. /apps add {toolkit} to try again."
            )
            return
        walk = self._apps_walk
        if walk is None or not walk.alive:
            walk = self._new_apps_wizard()
            self._apps_walk = walk
        await self._surfaces.send_to_user(walk.enter_permissions(app.toolkit, app.identity))

    async def _try_explain_reply(self, message: InboundMessage) -> bool:
        """A reply that is just 'why?' (or 'explain') about one of my own
        messages: the recorded trace of that message answers,
        deterministically — no LLM turn. Anything else — other words, or a
        message nobody recorded — falls through to the normal path."""
        if not message.reply_to_id:
            return False
        if message.text.strip().rstrip("?!").lower() not in WHY_REPLY_WORDS:
            return False
        trace = await self._trace_for_ref(message.surface, message.reply_to_id)
        if trace is None:
            return False  # nothing links that message — the model answers
        # solicited by the reply, so the replay is sent directly
        await self._surfaces.send_to_user(plain_replay(trace))
        return True

    async def _trace_for_ref(self, surface: str, ref: str) -> Any | None:
        """The newest trace whose recorded delivery includes this platform
        message id on this surface — the why? lookup. Scans recent traces;
        a ref older than that window falls through to the model, which can
        still answer with explain_decision."""
        if self._traces is None:
            return None
        for trace in await self._traces.recent(50):
            if (trace.payload or {}).get("chat_refs", {}).get(surface) == ref:
                return trace
        return None

    # -- source polling -----------------------------------------------------------

    def rebuild_poll_targets(self) -> None:
        """Rebuild the poll schedule from the host's current servers — the
        boot block, factored out so a chat-made mcp_servers change can
        re-run it against the new live set. Servers that left take their
        stale watchdog state with them."""
        self._poll_targets.clear()
        self._next_poll.clear()
        if self._host is None:
            return
        now = datetime.now(UTC)
        for conn in self._host.connections:
            for i, poll in enumerate(conn.poll_tools):
                key = f"{conn.name}:{poll.tool}:{i}"
                self._poll_targets[key] = (conn.name, poll)
                self._next_poll[key] = now  # the first tick runs every poll once
        live = {conn.name for conn in self._host.connections}
        self._poll_failures = {k: v for k, v in self._poll_failures.items() if k in live}
        self._down = {s for s in self._down if s in live}

    async def _run_due_polls(self) -> None:
        """Run every configured poll that is due. The polls double as the
        connector watchdog's heartbeat: consecutive failures mark a server
        down (announced once), and the next good poll announces the
        recovery."""
        if not self._poll_targets:
            return
        now = datetime.now(UTC)
        for key, (server, poll) in list(self._poll_targets.items()):
            if now < self._next_poll.get(key, now):
                continue
            self._next_poll[key] = now + timedelta(minutes=poll.every_minutes)
            qualified = f"{server}__{poll.tool}"
            try:
                result = await self._host.call(qualified, poll.args)  # type: ignore[union-attr]
            except (UnknownToolError, ConnectorUnavailableError) as exc:
                log.warning("source poll %s failed: %s", qualified, exc)
                failures = self._poll_failures.get(server, 0) + 1
                self._poll_failures[server] = failures
                if failures >= POLL_FAILURES_BEFORE_DOWN and server not in self._down:
                    self._down.add(server)
                    await self._notify(
                        f"{server} hasn't been responding — I've paused "
                        "relying on it and will say when it's back."
                    )
                continue
            # a good poll is the heartbeat: the failure count resets, and a
            # server that was announced down gets its recovery announced
            self._poll_failures.pop(server, None)
            if server in self._down:
                self._down.discard(server)
                await self._notify(f"{server} is responding again.")
            await self._audit.append(
                actor="scheduler",
                tool_name=qualified,
                decision="allow",
                rules_matched="config:poll_tools",
                params=poll.args,
                outcome="source poll",
            )
            ingest = await self._events.ingest(
                source=server,
                kind=f"poll:{poll.tool}",
                payload={"args": poll.args, "result": result[:4000]},
            )
            if ingest.stored:
                self.notify()

    # -- the late-ready reconcile ---------------------------------------------------

    async def _reconcile_mcp(self) -> None:
        """Keep the tool namespace honest about servers that answered
        late: the sync runs at boot and at apply time, so a server that
        comes ready after its moment would otherwise never enter the
        namespace at all — its actions stay uncallable while the config
        says it's linked. Every tick, a server that's ready at the host
        but missing from the namespace is synced in, and its coming-up is
        said out loud — once per boot, so a flapping server never
        announces twice. A ready server whose tool list changed since the
        last sync — a connection hub the user approved new apps on — is
        re-synced and its new actions announced the same way.

        One exception: the composio hub never announces its ups. Its tools
        arriving or growing is housekeeping, not news — it speaks only
        when it's down, which the watchdog below already does."""
        if self._host is None:
            return
        versions: dict[str, int] = {}
        late: set[str] = set()
        changed: set[str] = set()
        for conn in self._host.connections:
            if not conn.ready:
                continue
            versions[conn.name] = getattr(conn, "tool_version", 0)
            if not self._tools.has_server(conn.name):
                late.add(conn.name)
            elif conn.name in self._synced_tool_versions:
                if self._synced_tool_versions[conn.name] != versions[conn.name]:
                    changed.add(conn.name)
            else:
                # ready when its sync ran at boot or apply time — never
                # late, never announced from here; just record
                self._synced_tool_versions[conn.name] = versions[conn.name]
        if not late and not changed:
            return
        before = {name: self._tools.server_action_count(name) for name in changed}
        self._tools.sync_mcp_tools()
        for name in sorted(late):
            self._synced_tool_versions[name] = versions[name]
            if name == SERVER_NAME:
                continue  # the hub joins silently — see the docstring
            if name in self._announced_ready:
                continue
            self._announced_ready.add(name)
            note = f"{name} is up"
            count = self._tools.server_action_count(name)
            if count:
                note += f" — {_actions_word(count)} in my vocabulary"
            # it completes a flow the user started in chat, so like a
            # /config reply it never waits out quiet hours
            await self._surfaces.send_to_user(note)
        for name in sorted(changed):
            self._synced_tool_versions[name] = versions[name]
            if name == SERVER_NAME:
                continue  # hub growth is said by the connect flow, not here
            gained = self._tools.server_action_count(name) - before[name]
            if gained > 0:
                word = "action" if gained == 1 else "actions"
                await self._surfaces.send_to_user(f"{name} — {gained} new {word} in my vocabulary")

    # -- the hub watchdog ---------------------------------------------------------

    async def _watch_hub(self) -> None:
        """A hub session can go stale in a way the reuse path can't see:
        the router still answers initialize and sessions.use, but tools/list
        comes back as a dead stream, so the connection flaps in a reconnect
        loop on the same URL forever. Sustained unreadiness → the bridge
        cuts one fresh session; the host re-resolves the $NAME refs onto it
        at its next reconnect, and the reconcile above announces the tools
        when they land. Cooldown + per-boot cap: if resets don't fix it,
        the outage is Composio's and only its own recovery ends it."""
        if self._composio is None or self._host is None:
            return
        conn = next((c for c in self._host.connections if c.name == SERVER_NAME), None)
        if conn is None or conn.ready:
            self._hub_unready_since = None
            return
        now = time.monotonic()
        if self._hub_unready_since is None:
            self._hub_unready_since = now
            return
        if now - self._hub_unready_since < _HUB_RESET_AFTER_SECONDS:
            return
        if (
            self._hub_last_reset is not None
            and now - self._hub_last_reset < _HUB_RESET_COOLDOWN_SECONDS
        ):
            return
        if self._hub_resets >= _HUB_MAX_RESETS_PER_BOOT:
            return
        self._hub_resets += 1
        self._hub_last_reset = now
        self._hub_unready_since = None
        try:
            reset = await self._composio.reset_session()
        except Exception:
            log.warning("hub session reset failed", exc_info=True)
            return
        if reset:
            log.info("composio endpoint stale — cut a fresh session")
            await self._surfaces.send_to_user(
                "composio's endpoint stopped answering, so I've cut a fresh "
                "session — its tools rejoin in a few seconds."
            )

    # -- routines ---------------------------------------------------------------

    async def _run_routines(self, observations: list[Event]) -> None:
        """Evaluate the user's standing triggers against this tick's new
        events. Matching is pure code — no LLM in the decision of *whether*
        to react — and each fire goes through `_execute`, so the gate
        applies at fire time, every time."""
        routines = await self._routines.list_enabled()
        if not routines:
            return
        now = datetime.now(UTC)
        for routine in routines:
            if (
                routine.last_fired_at is not None
                and routine.cooldown_seconds > 0
                and now < routine.last_fired_at + timedelta(seconds=routine.cooldown_seconds)
            ):
                continue  # inside its cooldown window — a burst of similar events fires it once
            event = next(
                (e for e in observations if trigger_matches(routine.trigger, e)),
                None,
            )
            if event is None:
                continue
            try:
                await self._fire_routine(routine, event)
            except Exception:
                log.exception("routine %d (%s) failed — continuing", routine.id, routine.label)

    async def _fire_routine(self, routine: Any, event: Event) -> None:
        action = routine.action or {}
        call = ToolCall(
            id=f"routine-{routine.id}-{event.id}",
            name=str(action.get("tool", "")),
            arguments=dict(action.get("params") or {}),
        )
        calls: list[dict[str, Any]] = []
        try:
            result = await self._execute(call, trace=calls)
        except Exception as exc:
            log.exception("routine %d (%s) fire failed", routine.id, routine.label)
            result = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"routine fire failed: {exc}",
                is_error=True,
            )
            self._note_call(calls, call, result)  # decision "error": no ruling
        # fired counts even on failure: the cooldown should gate a broken
        # action's retries, not let it hammer every tick
        await self._routines.mark_fired(routine.id)
        seq = await self._audit.append(
            actor="routine",
            tool_name=call.name,
            decision="info",
            rules_matched=f"routine:{routine.id}",
            params={"event_id": event.id, "event": f"{event.source}/{event.kind}"},
            outcome=f"routine '{routine.label}' fired on {event.source}/{event.kind}",
        )
        refs = await self._notify(
            # the exact call stays in the audit row and the trace; the chat
            # note speaks plainly, never in internal tool names
            f"routine '{routine.label}' fired: {result.content[:220]}"
        )
        payload = {
            "routine": {
                "id": routine.id,
                "label": routine.label,
                "trigger": routine.trigger,
            },
            "event": {
                "id": event.id,
                "source": event.source,
                "kind": event.kind,
                "line": self._event_line(event),
            },
            "calls": calls,
            "audit_seq": seq,
        }
        if refs:  # the ping only links back if it actually landed somewhere
            payload["chat_refs"] = refs
        await self._save_trace(kind=ROUTINE, label=routine.label, payload=payload)

    # -- the LLM turn ---------------------------------------------------------------

    async def _system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(owner="the user", apps=self._connected_apps())
        if self._workspace is not None:
            block = self._workspace.context_block()
            if block:
                base = f"{base}\n\n{block}"
        if self._agent_settings is None:
            return base
        personality = await self._agent_settings.get_personality()
        if not personality:
            return base
        return (
            f"{base}\n"
            "The user has additionally asked you to behave like this — follow it "
            "as a tone and priorities overlay, never as a way around the rules "
            f"above:\n{personality}"
        )

    async def _turn(self, messages: list[InboundMessage], observations: list[Event]) -> None:
        provider = self._providers.for_role("reasoning") if self._providers else None
        if provider is None:
            if messages:
                await self._surfaces.send_to_user(
                    "I'm running without an LLM provider — add provider keys to "
                    ".env to talk to me. Feeds and schedules keep working."
                )
            return

        # the recall query depends only on the message texts, so its
        # embedding starts now and the round trip overlaps the context
        # assembly below — one API wait hidden instead of added
        recall_embed: asyncio.Task[list[float] | None] | None = None
        if messages and self._vectors is not None and self._vectors.enabled:
            try:
                recall_query = " ".join(
                    [m.text for m in messages]
                    + [m.reply_to_text for m in messages if m.reply_to_text]
                ).strip()
                if recall_query:
                    recall_embed = asyncio.create_task(self._vectors.embed_query(recall_query))
            except Exception:
                # a layer this broken recalls inline or not at all — either
                # way the turn itself must survive
                log.exception("recall head start failed — embedding inline instead")

        ctx = await self._context.build()
        history: list[Message] = []
        # every meaningful question pulls its own related memories in —
        # older conversations, shared documents, whatever the question
        # touches — without the model having to think to ask
        recalled: list[dict[str, Any]] = []
        recall_lines: list[str] = []
        if messages:
            recalled, recall_lines = await self._recall_related(
                messages, ctx, observations, recall_embed
            )
        context_block = self._format_context(ctx)
        head = "\n\n".join(part for part in (context_block, "\n".join(recall_lines)) if part)
        if head:
            history.append(Message.user(f"[working context]\n{head}"))
        history.extend(await self._transcript_messages(messages))
        for event in reversed(observations):  # oldest first inside the block
            history.append(Message.user(f"[new event] {self._event_line(event)}"))
        for message in messages:
            line = f"[message from {message.handle} via {message.surface}]\n{message.text}"
            if message.reply_to_text:
                # so the model can see *which* of its own messages is being
                # answered when the platform hands the text over
                line += f'\n(replying to my earlier message: "{message.reply_to_text[:200]}")'
            history.append(Message.user(line))
        history = _window_history(history, keep_head=1 if context_block else 0)

        calls: list[dict[str, Any]] = []  # filled by _execute as the turn runs
        trace: dict[str, Any] = {
            "trigger": {
                "messages": [
                    {
                        "surface": m.surface,
                        "handle": m.handle,
                        "text": m.text,
                        "reply_to_id": m.reply_to_id,
                        "reply_to_text": m.reply_to_text,
                    }
                    for m in messages
                ],
                "observations": [
                    {
                        "id": e.id,
                        "source": e.source,
                        "kind": e.kind,
                        "line": self._event_line(e),
                    }
                    for e in observations
                ],
                # what the turn kept in mind beyond its own window — same
                # shape as observations, so the panel's plain-language
                # rendering parses it unchanged
                "recalled": recalled,
            },
            "calls": calls,
        }
        full_history: list[Message] = []
        reply = ""
        # tool gating: natives always; MCP tools whose names match the
        # trigger text or anything the model has said/called this turn, plus
        # whatever search_tools found — recorded by name in `unlocked`,
        # since tool results never reach the gate's text (Message.text
        # excludes them)
        trigger_text = " ".join(
            [m.text for m in messages]
            + [
                f"{e.source} {e.kind} {json.dumps(e.payload, default=str)[:200]}"
                for e in observations
            ]
        )
        unlocked: set[str] = set()

        # the agent's own acts this turn is being told about: a tool whose
        # outcome just arrived. On a turn nobody asked for, re-proposing
        # one is a loop, not a new request — the executor refuses to park
        # it again (see _execute), so one carried-out ask never repeats.
        # A turn with a message in it never builds this map: the user
        # typing "send it again" is a fresh request and must be heard.
        settled: dict[str, str] = {}
        if not messages:
            for event in observations:
                if event.source != "agent" or not isinstance(event.payload, dict):
                    continue
                tool = event.payload.get("tool")
                if isinstance(tool, str) and tool:
                    settled[tool] = str(
                        event.payload.get("plain")
                        or describe_call(tool, event.payload.get("params"))
                    )

        def _gate(active: set[str], so_far: list[Message]) -> list[ToolSpec]:
            said = (
                trigger_text
                + " "
                + " ".join(m.text + " " + " ".join(c.name for c in m.tool_calls) for m in so_far)
            )
            return self._tools.gated_specs(said, unlocked=active | unlocked)

        async def _execute_unlocking(call: ToolCall) -> ToolResult:
            result = await self._execute(call, trace=calls, settled=settled)
            if call.name == "search_tools":
                # re-run the same search the handler did and offer its
                # matches on the next iteration — the result text itself
                # can't carry them through the gate
                unlocked.update(
                    s.name for s in self._tools.search(str(call.arguments.get("query", "")))
                )
            return result

        try:
            final, full_history = await run_tool_loop(
                provider,
                await self._system_prompt(),
                history,
                _gate,
                _execute_unlocking,
                max_iterations=self._config.agent.max_tool_iterations,
            )
            reply = final.text.strip()
            if reply and not _delivered_by_tool(calls):
                if messages:
                    # the user spoke first — a reply is solicited, it never
                    # waits; record where it landed so a later "why?" can
                    # find this trace
                    refs = await self._surfaces.send_to_user(reply)
                    if refs:
                        trace["chat_refs"] = refs
                else:
                    # unprompted commentary: the interruption budget decides,
                    # urgent = the loudest thing this turn saw (0-10 salience)
                    refs = await self._notify(
                        reply,
                        urgent=max((e.salience_score for e in observations), default=0.0),
                    )
                    if refs:
                        trace["chat_refs"] = refs
            elif reply:
                # the model already sent its words this turn via send_chat_message;
                # delivering the final reply too is what reads as a duplicate —
                # the same answer twice, differently worded. The reply stays in
                # the trace, so replay still shows how the turn ended.
                log.info("reply not sent — send_chat_message already reached the user this turn")
        except Exception as exc:
            trace["error"] = f"{type(exc).__name__}: {exc}"
            if messages:
                # a solicited turn dying mid-flight must not leave the user
                # staring at silence — plain words, and send_to_user already
                # swallows per-surface failures so this can't double-fault.
                # Unprompted ticks stay quiet: a background hiccup is a log
                # line, never an interruption.
                await self._surfaces.send_to_user(FAILURE_REPLY)
            raise
        finally:
            # persist even when the turn died mid-flight: the calls that did
            # run are already in `calls` (filled by reference), so a partial
            # record still answers "what were you doing?"
            reasoning = [
                m.text.strip() for m in full_history if m.role == "assistant" and m.text.strip()
            ]
            if reasoning and reply and reasoning[-1] == reply:
                reasoning.pop()  # the reply is stored as its own field
            if reasoning:
                trace["reasoning"] = reasoning
            if reply:
                trace["reply"] = reply
            await self._save_trace(
                kind=TURN,
                label=self._trace_label(messages, observations),
                payload=trace,
            )

    def _connected_apps(self) -> str:
        """The live MCP server list for the system prompt — the model sees
        exactly what's reachable, so "can you send emails?" gets an honest
        "mail isn't connected" instead of an invented capability."""
        servers = self._tools.mcp_servers()
        return ", ".join(servers) if servers else "none"

    async def _recall_related(
        self,
        messages: list[InboundMessage],
        ctx: Any,
        observations: list[Event],
        query_embed: asyncio.Task[list[float] | None] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """The memories a solicited turn pulls in automatically: chunks of
        older conversations and shared documents that relate to what was
        just asked — semantic, so a missing or failing embedding layer
        simply recalls nothing. `query_embed` is the head start _turn gave
        the query embedding so its round trip overlaps the context build;
        without one it is embedded here. Excludes what the turn already
        sees (working context events, this tick's chat observations), but a
        document shared this same tick stays retrievable — that is what
        makes "what's in slide 3" answerable in one turn. Failure costs
        the turn nothing: an empty recall."""
        if self._vectors is None or not self._vectors.enabled:
            return [], []
        query = " ".join(
            [m.text for m in messages] + [m.reply_to_text for m in messages if m.reply_to_text]
        ).strip()
        if not query:
            return [], []
        excluded = {e.id for e in ctx.events}
        excluded |= {e.id for e in observations if e.kind == "chat_message"}
        try:
            q = (
                await query_embed
                if query_embed is not None
                else await self._vectors.embed_query(query)
            )
            if q is None:
                return [], []
            hits = await self._vectors.search_vector(q, limit=12)
            # a document the user just named goes to the front, ahead
            # of the similarity ranking alone — "explain deadlock
            # prevention from OSY.pdf" must land in OSY.pdf
            lowered = query.lower()
            hits.sort(
                key=lambda h: (
                    0 if h.filename and Path(h.filename).stem.lower() in lowered else 1,
                    -h.score,
                )
            )
            candidates: list[tuple[int, str, str, str, str]] = []
            for hit in hits:
                candidates.append(
                    (hit.event_id, hit.source, hit.kind, await self._hit_header(hit), hit.text)
                )
            return self._render_recall(candidates, excluded)
        except Exception:
            log.exception("memory recall failed — continuing without it")
            return [], []

    async def _hit_header(self, hit: MemoryHit) -> str:
        """The citation half of a recalled chunk: which document and page,
        or who said it and when — never a bare blob of text."""
        when = f"{hit.occurred_at:%Y-%m-%d}"
        if hit.kind == "shared_doc":
            where = ", ".join(part for part in (hit.filename, hit.location) if part)
            return where or hit.source
        if hit.kind == "chat_message":
            handle = None
            try:
                event = await self._events.get(hit.event_id)
                if event and isinstance(event.payload, dict):
                    sender = event.payload.get("_sender")
                    if isinstance(sender, dict):
                        handle = sender.get("handle")
            except Exception:
                pass
            if handle:
                return f"@{handle} on {hit.source}, {when}"
        return f"{hit.source}, {when}"

    @staticmethod
    def _render_recall(
        candidates: list[tuple[int, str, str, str, str]], excluded: set[int]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Candidates in, capped prompt lines + trace entries out. The caps
        rank by relevance, they don't window the store: at most 3 chunks
        per source event, 6 lines, ~4,000 chars — so six documents can
        contribute without any one of them drowning the rest."""
        lines: list[str] = []
        trace: list[dict[str, Any]] = []
        per_event: dict[int, int] = {}
        chars = 0
        for event_id, source, kind, header, text in candidates:
            if event_id in excluded or per_event.get(event_id, 0) >= 3:
                continue
            line = f"[related memory — {header}] {text[:600]}".strip()
            if not line:
                continue
            if len(lines) >= 6 or chars + len(line) > 4000:
                break
            per_event[event_id] = per_event.get(event_id, 0) + 1
            chars += len(line)
            lines.append(line)
            trace.append({"id": event_id, "source": source, "kind": kind, "line": line[:200]})
        return trace, lines

    async def _transcript_messages(self, current: list[InboundMessage]) -> list[Message]:
        """The recent cross-surface dialogue as user/assistant messages —
        the turn's conversation memory. Rows echoing this turn's own inbound
        are dropped: they get the richer [message from …] line below. A
        missing or broken transcript costs the turn nothing."""
        if self._transcript is None:
            return []
        try:
            rows = await self._transcript.recent(limit=_TRANSCRIPT_MESSAGES)
        except Exception:
            log.exception("transcript read failed — continuing without it")
            return []
        fresh = {m.text for m in current}
        out: list[Message] = []
        for row in rows:
            text = row.get("text", "").strip()
            if not text:
                continue
            if row["direction"] == "out":
                out.append(Message.assistant(text))
            elif text not in fresh:
                out.append(Message.user(f"[{row.get('surface', '?')}] {text}"))
        return out

    def _format_context(self, ctx: Any) -> str:
        lines: list[str] = []
        for event in ctx.events[:15]:
            lines.append(f"- {self._event_line(event)}")
        for note in ctx.notes:
            who = note.payload.get("handle") or "someone"
            lines.append(f"- note on {who}: {note.payload.get('note', '')}")
        if ctx.pending_approvals:
            lines.append(f"- {len(ctx.pending_approvals)} approval(s) still waiting for the user")
        if ctx.today_memorable:
            lines.append(f"- {ctx.today_memorable} memorable event(s) so far today")
        return "\n".join(lines)

    @staticmethod
    def _event_line(event: Event) -> str:
        text = json.dumps(event.payload, ensure_ascii=False, default=str)
        return (
            f"{event.occurred_at:%Y-%m-%d %H:%M} {event.source}/{event.kind}"
            f" (salience {event.salience_score:.0f}): {text[:400]}"
        )

    # -- the authorization-gated executor ---------------------------------------------

    @staticmethod
    def _note_call(
        trace: list[dict[str, Any]] | None,
        call: ToolCall,
        result: ToolResult,
        *,
        ruling: Any = None,
        audit_seq: int | None = None,
        approval_id: int | None = None,
    ) -> None:
        """Append one call's full record to the trace being built, if any —
        the gate's own view: the ruling, the audit row (or approval) it
        produced, and what came back. `trace is None` means the call isn't
        part of a traced act; tracing stays out of the decision path."""
        if trace is None:
            return
        if ruling is None:  # the call failed before the gate produced an outcome
            decision, matched_rule, reason = "error", "", ""
        else:
            decision = ruling.decision.value
            matched_rule = ruling.matched_rule
            reason = ruling.reason
        trace.append(
            {
                "id": call.id,
                "name": call.name,
                # the model's own sentence when it offered one, the backend's
                # vocabulary otherwise — the panel never needs the raw name
                "plain": call.plain or describe_call(call.name, call.arguments),
                "params": dict(call.arguments),
                "decision": decision,
                "matched_rule": matched_rule,
                "reason": reason,
                "audit_seq": audit_seq,
                "approval_id": approval_id,
                "result": result.content,
                "is_error": result.is_error,
            }
        )

    def _plain_unavailable(self, name: str) -> str:
        """Plain words for a name the namespace can't run — the model relays
        these to the user verbatim, so no internal names, no error jargon.
        The trace and audit keep the exact call for replay."""
        server, _, tool_name = name.partition("__")
        if not tool_name:
            return "that action isn't available right now"
        if self._tools.has_server(server):
            return f"that action isn't available in {server} right now"
        return f"{server} isn't connected right now"

    @staticmethod
    def _plain_unresponsive(name: str) -> str:
        """Plain words for a linked app that failed to answer a call."""
        server, sep, _ = name.partition("__")
        if sep:
            return f"{server} isn't responding right now"
        return "that app isn't responding right now"

    async def _execute(
        self,
        call: ToolCall,
        trace: list[dict[str, Any]] | None = None,
        origin: str | None = None,
        settled: dict[str, str] | None = None,
    ) -> ToolResult:
        """The single choke point between a proposal and the world. `trace`,
        when given, is the call-record list of the decision trace being
        built — passed explicitly because the scheduler worker and the agent
        loop run as separate tasks and must never write to a shared one.
        `origin`, when given, is the loop-internal marker for a change the
        user just drove in chat — it never rides a model proposal, so the
        policy can let a walk-driven connect through without widening
        anything else. `settled`, when given, maps a tool name to the plain
        sentence of the outcome this turn was just told about — an
        unprompted turn passes it so a re-proposal of an act that already
        ran is refused instead of parked again.

        A call the namespace cannot run never parks: the user is not asked
        to consent to certain failure — it comes back in the same plain
        words as any other missing tool."""
        # the model's one-sentence account of the call rides in as `_plain`
        # (see PLAIN_PARAM): lift it onto the record and hand the policy,
        # the audit, and the tool a call that never carries it
        plain = call.arguments.get(PLAIN_PARAM)
        if plain is not None:
            call = ToolCall(
                call.id,
                call.name,
                {k: v for k, v in call.arguments.items() if k != PLAIN_PARAM},
                plain=str(plain),
            )
        ruling = self._policy.classify(call.name, call.arguments, origin=origin)

        if ruling.decision is Decision.DENY:
            seq = await self._audit.append(
                actor="agent",
                tool_name=call.name,
                decision="deny",
                rules_matched=ruling.matched_rule,
                params=call.arguments,
                outcome=ruling.reason,
            )
            result = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"denied by policy: {ruling.reason}",
                is_error=True,
            )
            self._note_call(trace, call, result, ruling=ruling, audit_seq=seq)
            return result

        if ruling.decision is Decision.ALLOW:
            try:
                content = await self._tools.execute(call.name, call.arguments)
            except UnknownToolError:
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=self._plain_unavailable(call.name),
                    is_error=True,
                )
                self._note_call(trace, call, result, ruling=ruling)
                return result
            except ConnectorUnavailableError:
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=self._plain_unresponsive(call.name),
                    is_error=True,
                )
                self._note_call(trace, call, result, ruling=ruling)
                return result
            seq = await self._audit.append(
                actor="agent",
                tool_name=call.name,
                decision="allow",
                rules_matched=ruling.matched_rule,
                params=call.arguments,
                outcome=ruling.reason,
            )
            result = ToolResult(tool_call_id=call.id, name=call.name, content=content)
            self._note_call(trace, call, result, ruling=ruling, audit_seq=seq)
            return result

        # a risky call the namespace can't run must never ask for consent —
        # approving certain failure is not a decision worth interrupting the
        # user for. Answer with the same plain words the ALLOW path uses;
        # the trace keeps the exact call.
        if self._tools.get(call.name) is None:
            result = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=self._plain_unavailable(call.name),
                is_error=True,
            )
            self._note_call(trace, call, result, ruling=ruling)
            return result

        # a call this unprompted turn was just told the outcome of: it
        # already ran (or was already answered) and the user heard the
        # result. Parking it again is how one approved send becomes a
        # loop of identical asks — the guard is code, because the parked
        # result's "do not propose this call again" only lives in the
        # turn that parked it, which a later turn never sees again.
        if settled and call.name in settled:
            result = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=(f"already settled — {settled[call.name]}. Do not propose it again."),
            )
            self._note_call(trace, call, result, ruling=ruling)
            return result

        # the same call parked earlier this turn: the model repeating
        # itself must not put the same question in front of the user
        # twice — the second proposal gets the first's held answer
        if trace:
            for entry in trace:
                if (
                    entry.get("approval_id") is not None
                    and entry.get("name") == call.name
                    and entry.get("params") == call.arguments
                ):
                    result = ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        content=str(entry.get("result", "")),
                    )
                    self._note_call(trace, call, result, ruling=ruling)
                    return result

        # REQUIRE_APPROVAL: park it, ask, and let the turn go on
        approval = await self._approvals.create(
            tool_name=call.name,
            params=call.arguments,
            rules_matched=ruling.matched_rule,
            note=ruling.reason,
        )
        await self._surfaces.present_approval(approval.id, describe_ask(call.name, call.arguments))
        result = ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=(
                f"held for approval (#{approval.id}) — the user has been asked on "
                "their chat surfaces and the web panel; it will run if they "
                "approve. Do not propose this call again. The request is already "
                "in front of the user with one-tap buttons — do not announce or "
                "narrate the wait in your reply."
            ),
        )
        self._note_call(trace, call, result, ruling=ruling, approval_id=approval.id)
        return result

    # -- decisions ------------------------------------------------------------------

    async def decide(self, approval_id: int, decision: str) -> Approval | None:
        """The web panel's path: decide, then carry out a fresh approval."""
        decision = {"approve": APPROVED, "deny": DENIED}.get(decision, decision)
        approval = await self._approvals.decide(approval_id, decision)
        if approval is not None:
            await self._carry_out(approval)
        return approval

    async def execute_decision(self, approval_id: int, decision: str) -> None:
        """The chat surfaces' path: they already persisted the decision via
        Approvals.decide; this only carries a fresh one out."""
        approval = await self._approvals.get(approval_id)
        if approval is None or approval.status not in (APPROVED, DENIED):
            return
        await self._carry_out(approval)

    async def _carry_out(self, approval: Approval) -> None:
        if approval.status == DENIED:
            await self._surfaces.send_to_user(
                describe_dismissed(approval.tool_name, approval.params)
            )
            await self._remember_outcome(approval, "action_denied", "denied by the user")
            return
        trace: dict[str, Any] = {
            "approval_id": approval.id,
            "tool": approval.tool_name,
            "params": dict(approval.params),
            "decided_by": approval.decided_by,
        }
        try:
            result = await self._tools.execute(approval.tool_name, approval.params)
        except UnknownToolError as exc:
            # rows parked before the up-front check — or an app unlinked
            # while one sat pending — still land here. The user hears plain
            # words; the exact name and error stay in the trace and the log.
            log.warning("approved call %s can't run: %s", approval.tool_name, exc)
            await self._approvals.mark_failed(approval.id)
            trace["result"] = str(exc)
            trace["is_error"] = True
            trace["chat_refs"] = await self._surfaces.send_to_user(
                f"the approved action couldn't run — {self._plain_unavailable(approval.tool_name)}"
            )
        except ConnectorUnavailableError as exc:
            log.warning("approved call %s can't run: %s", approval.tool_name, exc)
            await self._approvals.mark_failed(approval.id)
            trace["result"] = str(exc)
            trace["is_error"] = True
            trace["chat_refs"] = await self._surfaces.send_to_user(
                f"the approved action couldn't run — {self._plain_unresponsive(approval.tool_name)}"
            )
        except Exception as exc:
            log.exception("approved call %s failed to run", approval.tool_name)
            await self._approvals.mark_failed(approval.id)
            trace["result"] = str(exc)
            trace["is_error"] = True
            trace["chat_refs"] = await self._surfaces.send_to_user(
                "the approved action couldn't run — it failed unexpectedly. "
                "The decision record has exactly what happened."
            )
        else:
            await self._approvals.mark_executed(approval.id)
            trace["result"] = result
            trace["is_error"] = False
            # plain words only — the exact result stays in the decision
            # record and the trace; a raw dump never reaches the user
            trace["chat_refs"] = await self._surfaces.send_to_user(
                describe_outcome(approval.tool_name, approval.params)
            )
        await self._remember_outcome(
            approval,
            "action_failed" if trace["is_error"] else "action_done",
            str(trace["result"]),
        )
        await self._save_trace(
            kind=CARRY_OUT,
            label=f"approval #{approval.id} — {approval.tool_name}",
            payload=trace,
        )

    async def _remember_outcome(self, approval: Approval, kind: str, detail: str) -> None:
        """Close the loop in memory. The model's only picture of past turns
        is the event store — every turn rebuilds its history from memorable
        events — and nothing else records the agent's *own* acts: an
        approved action's outcome used to reach only the chat surface and
        the trace store, neither of which a later turn ever reads, so the
        request kept looking open ("which mail did you mean?") hours after
        it ran. Recording must never break the act itself, so a failed
        write is a log line, not a raise."""
        try:
            ingest = await self._events.ingest(
                source="agent",
                kind=kind,
                payload={
                    "tool": approval.tool_name,
                    "params": dict(approval.params),
                    "plain": plain_outcome(approval.tool_name, approval.params, kind),
                    "detail": detail[:1000],
                    # unique per approval, so two identical sends both record
                    "approval_id": approval.id,
                },
            )
        except Exception:
            log.warning("couldn't record the %s outcome", kind, exc_info=True)
            return
        if ingest.stored and ingest.event_id is not None:
            await self._salience.score_event(ingest.event_id)

    # -- scheduled actions ------------------------------------------------------------

    async def execute_scheduled(self, action: ScheduledAction) -> str:
        """What the scheduler worker runs for each due row. The payload is a
        tool call — and it passes the same gate as the model's proposals."""
        payload = action.payload or {}
        if payload.get("type") != "tool":
            raise ValueError(f"unknown scheduled payload type: {payload.get('type')!r}")
        call = ToolCall(
            id=f"sched-{action.id}",
            name=str(payload.get("tool", "")),
            arguments=dict(payload.get("params") or {}),
        )
        calls: list[dict[str, Any]] = []
        result = await self._execute(call, trace=calls)
        # recorded before any raise, so a failed fire still has its trace
        await self._save_trace(
            kind=SCHEDULED,
            label=action.label,
            payload={
                "action": {
                    "id": action.id,
                    "label": action.label,
                    "run_at": f"{action.run_at:%Y-%m-%d %H:%M:%S%z}",
                },
                "calls": calls,
                "result": result.content,
                "is_error": result.is_error,
            },
        )
        if result.is_error:
            raise RuntimeError(result.content)
        return result.content

    # -- traces -------------------------------------------------------------------

    async def _save_trace(self, *, kind: str, label: str, payload: dict[str, Any]) -> None:
        """Persist one trace. Traces are bookkeeping, never a participant:
        if the store isn't wired or the write fails, the act the trace
        records has already happened and stays untouched."""
        if self._traces is None:
            return
        try:
            await self._traces.create(kind=kind, label=label, payload=payload)
        except Exception:
            log.exception("failed to persist %s trace — the act stands", kind)

    @staticmethod
    def _trace_label(messages: list[InboundMessage], observations: list[Event]) -> str:
        """One human line for the trace list: the message that drove the
        turn, or what was seen if the turn was observation-only."""
        if messages:
            return messages[0].text[:48]
        return f"observed {len(observations)} new event(s)"
