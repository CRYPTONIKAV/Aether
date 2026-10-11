"""Slack surface: Socket Mode — an outbound WebSocket, so no public URL and
no signing secrets. DMs in, replies and approval buttons out.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import httpx

from ..authz.approvals import APPROVED, DENIED
from .attachments import MAX_ATTACHMENT_BYTES, reported_size_exceeds_cap
from .base import (
    ApprovalDecider,
    Attachment,
    DecisionCallback,
    InboundHandler,
    InboundMessage,
    MessagingConnector,
    classify_attachment,
)

log = logging.getLogger("aether.connectors.slack")

APPROVE_ACTION = "aether_approve"
DENY_ACTION = "aether_deny"


def event_to_inbound(
    event: dict[str, Any], downloaded: dict[str, bytes] | None = None
) -> InboundMessage | None:
    """A Slack message event as an InboundMessage, or None when it's not a
    human DM (bot chatter and channel noise are ignored). A file share
    (`subtype: "file_share"`) is a human DM too — `downloaded` carries the
    bytes per file id, fetched by the connector which owns the token."""
    if not event or event.get("channel_type") != "im":
        return None
    subtype = event.get("subtype") or ""
    files = event.get("files") or []
    if event.get("bot_id") or (subtype and subtype != "file_share"):
        return None
    if subtype == "file_share" and not files:
        return None
    text = event.get("text") or ""
    user = event.get("user") or ""
    if (not text and not files) or not user:
        return None
    attachments = tuple(
        Attachment(
            classify_attachment(str(f.get("name") or ""), str(f.get("mimetype") or "")),
            str(f.get("name") or "file"),
            str(f.get("mimetype") or ""),
            blob,
        )
        for f in files
        if (blob := (downloaded or {}).get(str(f.get("id")), b""))
    )
    return InboundMessage(
        surface="slack",
        handle=user,
        text=text,
        chat_ref=event.get("channel", ""),
        # a thread reply carries its parent's ts — enough to link a later
        # "why?" back to the trace of the message it answers
        reply_to_id=str(event.get("thread_ts") or ""),
        attachments=attachments,
    )


def approval_blocks(approval_id: int, text: str) -> list[dict[str, Any]]:
    """Block Kit payload for a pending approval: the plain-language
    question and two buttons. The raw tool name never appears — the
    decision record keeps it."""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text,
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "action_id": APPROVE_ACTION,
                    "value": str(approval_id),
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Deny"},
                    "action_id": DENY_ACTION,
                    "value": str(approval_id),
                    "style": "danger",
                },
            ],
        },
    ]


class SlackConnector(MessagingConnector):
    name = "slack"

    def __init__(
        self,
        *,
        bot_token: str,
        app_token: str,
        enabled: bool = True,
        approvals: ApprovalDecider | None = None,
        on_decision: DecisionCallback | None = None,
        on_inbound: InboundHandler | None = None,
        app_factory: Any = None,
        socket_factory: Any = None,
    ) -> None:
        super().__init__(
            enabled=enabled,
            approvals=approvals,
            on_decision=on_decision,
            on_inbound=on_inbound,
        )
        self._bot_token = bot_token
        self._app_token = app_token
        self._app_factory = app_factory or self._default_app_factory
        self._socket_factory = socket_factory or self._default_socket_factory
        self._app: Any = None
        self._socket: Any = None
        self._chat_ref: str | None = None

    @staticmethod
    def _default_app_factory(bot_token: str) -> Any:
        from slack_bolt.async_app import AsyncApp

        return AsyncApp(token=bot_token)

    @staticmethod
    def _default_socket_factory(app: Any, app_token: str) -> Any:
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

        return AsyncSocketModeHandler(app, app_token)

    # -- lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        if not self._enabled or not self._bot_token or not self._app_token:
            log.info(
                "slack connector not started (enabled=%s, bot token=%s, app token=%s)",
                self._enabled,
                bool(self._bot_token),
                bool(self._app_token),
            )
            return
        app = self._app_factory(self._bot_token)
        app.event("message")(self.on_message)
        app.action(APPROVE_ACTION)(self.on_approve)
        app.action(DENY_ACTION)(self.on_deny)
        socket = self._socket_factory(app, self._app_token)
        await socket.connect_async()
        self._app, self._socket = app, socket
        log.info("slack socket mode connected")

    async def stop(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            try:
                await socket.disconnect_async()
            except Exception:
                log.exception("slack socket disconnect failed")
        self._app = None

    # -- inbound -----------------------------------------------------------------

    async def on_message(self, event: dict[str, Any], say: Any, client: Any) -> None:
        event = dict(event)
        downloaded = await self._download_files(list(event.get("files") or []))
        inbound = event_to_inbound(event, downloaded)
        if inbound is None:
            return
        self._chat_ref = inbound.chat_ref
        await self._emit_inbound(inbound)

    async def _download_files(self, files: list[dict[str, Any]]) -> dict[str, bytes]:
        """Bytes per file id — the connector is the one holding the bot
        token. A file that can't be downloaded (oversize, network) is
        logged and skipped; it must never take the surface down."""
        out: dict[str, bytes] = {}
        for f in files:
            url = f.get("url_private_download") or ""
            name = f.get("name")
            if not url:
                continue
            if reported_size_exceeds_cap(f.get("size"), MAX_ATTACHMENT_BYTES):
                log.warning("slack file %s is over the size cap — skipped", name)
                continue
            try:
                data = await self._stream_download(url, name)
            except Exception:
                log.warning("slack file %s download failed — continuing", name, exc_info=True)
                continue
            if data is not None:
                out[str(f.get("id"))] = data
        return out

    async def _stream_download(self, url: str, name: str) -> bytes | None:
        """Downloaded in chunks, stopping the instant the running total
        passes the cap — a reported size (missing, zero, or just wrong)
        never gets the last word on how much we actually pull into memory."""
        chunks = bytearray()
        headers = {"Authorization": f"Bearer {self._bot_token}"}
        async with (
            httpx.AsyncClient() as client,
            client.stream("GET", url, headers=headers) as response,
        ):
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                chunks += chunk
                if len(chunks) > MAX_ATTACHMENT_BYTES:
                    log.warning(
                        "slack file %s exceeded the size cap mid-download — discarded", name
                    )
                    return None
        return bytes(chunks)

    # -- buttons -----------------------------------------------------------------

    async def on_approve(self, ack: Any, body: dict[str, Any], client: Any) -> None:
        await self._handle_button(ack, body, client, APPROVED)

    async def on_deny(self, ack: Any, body: dict[str, Any], client: Any) -> None:
        await self._handle_button(ack, body, client, DENIED)

    async def _handle_button(
        self, ack: Any, body: dict[str, Any], client: Any, decision: str
    ) -> None:
        await ack()
        try:
            approval_id = int(body["actions"][0]["value"])
        except (KeyError, IndexError, TypeError, ValueError):
            log.warning("odd slack action payload: %r", body)
            return
        note = await self._decide_and_explain(approval_id, decision)
        with contextlib.suppress(
            Exception
        ):  # message may be old or locked — the store has the truth
            # a fresh decision needs no extra note — the outcome message
            # from _carry_out is the acknowledgment; a stale or double tap
            # gets a small note instead of landing with no feedback at all
            message = body.get("message", {})
            blocks = [
                block for block in message.get("blocks", []) if block.get("type") != "actions"
            ]
            if note:
                blocks.append(
                    {"type": "context", "elements": [{"type": "mrkdwn", "text": f"_{note}_"}]}
                )
            await client.chat_update(
                channel=body["channel"]["id"],
                ts=message["ts"],
                text=message.get("text", ""),
                blocks=blocks,
            )

    # -- outbound ------------------------------------------------------------------

    async def send_to_user(self, text: str) -> str | None:
        """Send, and answer with the message ts — what links a later "why?"
        thread reply back to the trace of this send."""
        if self._app is None or not self._chat_ref:
            log.info("slack: nowhere to send yet")
            return None
        response = await self._app.client.chat_postMessage(channel=self._chat_ref, text=text)
        try:
            return str(response["ts"] or "") or None
        except (KeyError, TypeError, IndexError):
            # a shape without the ts — the send happened, it just can't be
            # linked back to
            return None

    async def present_approval(self, approval_id: int, text: str) -> None:
        """One plain-language question plus one-tap Approve/Deny. The raw
        tool name never reaches the user — the decision record keeps it."""
        if self._app is None or not self._chat_ref:
            log.info("slack: nowhere to send yet — approval visible in the web panel")
            return
        await self._app.client.chat_postMessage(
            channel=self._chat_ref,
            text=text,
            blocks=approval_blocks(approval_id, text),
        )
