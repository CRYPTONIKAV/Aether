"""Connector contracts shared by every surface Aether talks through."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ..authz.approvals import EXPIRED, PENDING


class UnknownToolError(LookupError):
    """A tool name that isn't in the flat namespace."""


class ConnectorUnavailableError(RuntimeError):
    """A connector (typically an MCP server) is down or never connected."""


@dataclass(frozen=True)
class Attachment:
    """One file the user sent with a chat message. The connector that owns
    the platform credentials downloads the bytes before emitting; the loop's
    attachment processor turns them into memory."""

    kind: str  # "image" | "doc"
    filename: str
    media_type: str  # best-effort MIME type, "" when the platform doesn't say
    data: bytes


_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic"})


def classify_attachment(filename: str, media_type: str = "") -> str:
    """ "image" when the platform or the filename says so, "doc" otherwise."""
    if (media_type or "").startswith("image/"):
        return "image"
    name = (filename or "").lower()
    return "image" if any(name.endswith(ext) for ext in _IMAGE_EXTENSIONS) else "doc"


@dataclass(frozen=True)
class InboundMessage:
    """One message from the user on some surface.

    `handle` is the sender's handle as the platform wrote it — normalization
    and the contact allowlist happen at ingest, not here.

    `reply_to_id`/`reply_to_text` carry what the message answers, when the
    platform says so — the id is what links a later "why?" back to the trace
    of the message it's about.

    `attachments` are the files the message carried, bytes already
    downloaded by the connector.
    """

    surface: str  # "telegram" | "discord" | "slack" | "web"
    handle: str
    text: str
    chat_ref: str  # opaque, surface-specific reply target
    reply_to_id: str = ""  # platform id of the message this one answers, if any
    reply_to_text: str = ""  # its text, when the platform hands it over
    attachments: tuple[Attachment, ...] = ()


InboundHandler = Callable[[InboundMessage], Awaitable[None]]
# (approval_id, APPROVED | DENIED) — the Phase-6 executor's cue
DecisionCallback = Callable[[int, str], Awaitable[None]]


class ApprovalDecider(Protocol):
    """What messaging connectors need from the approvals store."""

    async def decide(self, approval_id: int, decision: str) -> Any: ...
    async def get(self, approval_id: int) -> Any: ...


class MessagingConnector:
    """Shared approve/deny + inbound plumbing; SDK glue lives in subclasses.

    Every surface presents the same buttons: a pending approval goes out
    with Approve/Deny attached, the press lands in one place (`_decide`),
    and only a *fresh* decision (not an expired or double-pressed one)
    fires `on_decision`.
    """

    name = ""

    def __init__(
        self,
        *,
        enabled: bool,
        approvals: ApprovalDecider | None = None,
        on_decision: DecisionCallback | None = None,
        on_inbound: InboundHandler | None = None,
    ) -> None:
        self._enabled = enabled
        self._approvals = approvals
        self._on_decision = on_decision
        self._on_inbound = on_inbound

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def start(self) -> None:
        """Lifecycle contract every surface shares; a surface with nothing to
        start up (or one stopped for config) overrides this."""

    async def stop(self) -> None:
        """The shutdown counterpart — a surface that never started is a
        no-op, so the fanout can stop everything unconditionally."""

    async def _decide(self, approval_id: int, decision: str) -> Any:
        """The one shared path behind every platform's approve/deny buttons."""
        result = None
        if self._approvals is not None:
            result = await self._approvals.decide(approval_id, decision)
        if result is not None and self._on_decision is not None:
            await self._on_decision(approval_id, decision)
        return result

    async def _decide_and_explain(self, approval_id: int, decision: str) -> str | None:
        """Runs the shared decide path; returns None on a fresh decision (no
        feedback needed — the carry-out message is the acknowledgment), or
        a short, platform-neutral phrase when the tap didn't apply, so a
        stale or double tap is never met with silence."""
        if await self._decide(approval_id, decision) is not None:
            return None
        if self._approvals is None:
            return "that request could not be found."
        approval = await self._approvals.get(approval_id)
        if approval is None:
            return "that request could not be found."
        if approval.status in (PENDING, EXPIRED):
            # still PENDING but decide() couldn't apply it — the only way
            # that happens is the expiry clock running out between the
            # button rendering and this tap
            return "that approval has expired."
        return "that request was already decided."

    async def _emit_inbound(self, message: InboundMessage) -> None:
        if self._on_inbound is not None:
            await self._on_inbound(message)
