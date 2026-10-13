"""Discord surface: gateway over WebSocket (no public URL), DMs in, replies
and approval buttons out. DMs only — the bot ignores servers and channels.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import discord

from ..authz.approvals import APPROVED, DENIED
from .attachments import MAX_ATTACHMENT_BYTES, actual_size_exceeds_cap, reported_size_exceeds_cap
from .base import (
    ApprovalDecider,
    Attachment,
    DecisionCallback,
    InboundHandler,
    InboundMessage,
    MessagingConnector,
    classify_attachment,
)

log = logging.getLogger("aether.connectors.discord")


class ApprovalView(discord.ui.View):
    """In-memory approve/deny buttons; they live exactly as long as the bot
    process. Approvals that outlive a restart surface in the web panel."""

    def __init__(self, decide_and_explain: Any, approval_id: int) -> None:
        super().__init__(timeout=None)
        self._decide_and_explain = decide_and_explain
        self._approval_id = approval_id

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.green)
    async def approve(self, button: discord.ui.Button, interaction: discord.Interaction) -> None:
        await self._finish(interaction, APPROVED)

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.red)
    async def deny(self, button: discord.ui.Button, interaction: discord.Interaction) -> None:
        await self._finish(interaction, DENIED)

    async def _finish(self, interaction: Any, decision: str) -> None:
        note = await self._decide_and_explain(self._approval_id, decision)
        for child in self.children:
            child.disabled = True
        # a fresh decision needs no extra note — `content` is left out
        # entirely so the question text stays untouched (py-cord treats an
        # explicit content=None as "clear it", not "leave it alone"); a
        # stale or double tap gets a short note instead of landing with no
        # feedback at all
        if note:
            await interaction.response.edit_message(content=note, view=self)
        else:
            await interaction.response.edit_message(view=self)


def _is_dm(channel: Any) -> bool:
    """Duck-typed so unit tests can pass lightweight fakes."""
    return getattr(channel, "type", None) == discord.ChannelType.private


class DiscordConnector(MessagingConnector):
    name = "discord"

    def __init__(
        self,
        *,
        token: str,
        enabled: bool = True,
        approvals: ApprovalDecider | None = None,
        on_decision: DecisionCallback | None = None,
        on_inbound: InboundHandler | None = None,
        bot_factory: Any = None,
    ) -> None:
        super().__init__(
            enabled=enabled,
            approvals=approvals,
            on_decision=on_decision,
            on_inbound=on_inbound,
        )
        self._token = token
        self._bot_factory = bot_factory or self._default_bot_factory
        self._bot: Any = None
        self._task: asyncio.Task[None] | None = None
        self._channel: Any = None

    @staticmethod
    def _default_bot_factory() -> Any:
        # message content is a privileged intent — the user must flip it on
        # in the Discord developer portal too (noted in the README config table)
        intents = discord.Intents.default()
        intents.message_content = True
        return discord.Bot(intents=intents)

    # -- lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        if not self._enabled or not self._token:
            log.info(
                "discord connector not started (enabled=%s, token set=%s)",
                self._enabled,
                bool(self._token),
            )
            return
        bot = self._bot_factory()
        bot.event(self.on_message)
        self._bot = bot
        self._task = asyncio.create_task(self._run_bot(), name="discord-gateway")
        log.info("discord connector starting")

    async def _run_bot(self) -> None:
        try:
            await self._bot.start(self._token)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "discord gateway stopped — check DISCORD_BOT_TOKEN and the privileged intents"
            )

    async def stop(self) -> None:
        bot, self._bot = self._bot, None
        if bot is not None:
            try:
                await bot.close()
            except Exception:
                log.exception("discord close failed")
        task, self._task = self._task, None
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # -- inbound -----------------------------------------------------------------

    async def on_message(self, message: Any) -> None:
        if message.author.bot or not _is_dm(message.channel):
            return
        attachments = await self._collect_attachments(message)
        if not message.content and not attachments:
            return
        self._channel = message.channel
        handle = message.author.name or str(message.author.id)
        reference = getattr(message, "reference", None)
        await self._emit_inbound(
            InboundMessage(
                surface="discord",
                handle=handle,
                text=message.content,
                chat_ref=str(message.author.id),
                # fetching the answered text costs an extra API round-trip —
                # the id alone links a later "why?" back to its trace
                reply_to_id=(
                    str(getattr(reference, "message_id", None) or "")
                    if reference is not None
                    else ""
                ),
                attachments=tuple(attachments),
            )
        )

    async def _collect_attachments(self, message: Any) -> list[Attachment]:
        """Files the user sent, downloaded now — the connector holds the
        platform credentials. A file that can't be downloaded (oversize,
        network) is logged and skipped; it must never take the surface down."""
        out: list[Attachment] = []
        for attachment in list(getattr(message, "attachments", None) or []):
            name = getattr(attachment, "filename", None) or "file"
            if reported_size_exceeds_cap(getattr(attachment, "size", None), MAX_ATTACHMENT_BYTES):
                log.warning("discord file %s is over the size cap — skipped", name)
                continue
            try:
                data = await attachment.read()
            except Exception:
                log.warning("discord file %s download failed — continuing", name, exc_info=True)
                continue
            # the one check that can't be lied to: whatever the reported
            # size claimed, a file that's actually too big is never kept
            if actual_size_exceeds_cap(data, MAX_ATTACHMENT_BYTES):
                log.warning(
                    "discord file %s exceeded the size cap after download (%d bytes) — discarded",
                    name,
                    len(data),
                )
                continue
            media = getattr(attachment, "content_type", None) or ""
            out.append(Attachment(classify_attachment(name, media), name, media, data))
        return out

    # -- outbound ------------------------------------------------------------------

    async def send_to_user(self, text: str) -> str | None:
        """Send, and answer with the platform message id — what links a
        later "why?" reply back to the trace of this send."""
        if self._channel is None:
            log.info("discord: no DM channel yet")
            return None
        sent = await self._channel.send(text)
        # getattr so test fakes without an id still send fine
        return str(getattr(sent, "id", "") or "") or None

    async def present_approval(self, approval_id: int, text: str) -> None:
        """One plain-language question plus one-tap Approve/Deny. The raw
        tool name never reaches the user — the decision record keeps it."""
        if self._channel is None:
            log.info("discord: no DM channel yet — approval visible in the web panel")
            return
        await self._channel.send(
            content=text,
            view=ApprovalView(self._decide_and_explain, approval_id),
        )
