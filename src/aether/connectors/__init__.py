"""Connectors: the MCP host, the flat tool registry, chat surfaces, and
screen perception."""

from __future__ import annotations

import logging

from ..config import AppConfig, Settings
from .attachments import AttachmentProcessor
from .base import (
    ApprovalDecider,
    Attachment,
    ConnectorUnavailableError,
    DecisionCallback,
    InboundHandler,
    InboundMessage,
    MessagingConnector,
    UnknownToolError,
)
from .mcp_host import MCPHost
from .registry import ToolRegistry
from .screenvision import ScreenVision

log = logging.getLogger("aether.connectors")

__all__ = [
    "ApprovalDecider",
    "Attachment",
    "AttachmentProcessor",
    "ConnectorUnavailableError",
    "DecisionCallback",
    "InboundHandler",
    "InboundMessage",
    "MCPHost",
    "MessagingConnector",
    "ScreenVision",
    "ToolRegistry",
    "UnknownToolError",
    "build_messaging_connectors",
]


def build_messaging_connectors(
    settings: Settings,
    config: AppConfig,
    *,
    approvals: ApprovalDecider | None = None,
    on_decision: DecisionCallback | None = None,
    on_inbound: InboundHandler | None = None,
) -> list[MessagingConnector]:
    """The enabled chat surfaces: toggle on AND the needed tokens present.

    Anything half-configured logs why it's skipped — a personal agent boots
    with whatever it has and the web chat always works.
    """
    built: list[MessagingConnector] = []

    if config.messaging.telegram.enabled:
        if settings.telegram_bot_token:
            from .telegram import TelegramConnector

            built.append(
                TelegramConnector(
                    token=settings.telegram_bot_token,
                    approvals=approvals,
                    on_decision=on_decision,
                    on_inbound=on_inbound,
                )
            )
        else:
            log.warning(
                "messaging.telegram is enabled but TELEGRAM_BOT_TOKEN is not set — skipping"
            )

    if config.messaging.discord.enabled:
        if settings.discord_bot_token:
            from .discord import DiscordConnector

            built.append(
                DiscordConnector(
                    token=settings.discord_bot_token,
                    approvals=approvals,
                    on_decision=on_decision,
                    on_inbound=on_inbound,
                )
            )
        else:
            log.warning("messaging.discord is enabled but DISCORD_BOT_TOKEN is not set — skipping")

    if config.messaging.slack.enabled:
        if settings.slack_bot_token and settings.slack_app_token:
            from .slack import SlackConnector

            built.append(
                SlackConnector(
                    bot_token=settings.slack_bot_token,
                    app_token=settings.slack_app_token,
                    approvals=approvals,
                    on_decision=on_decision,
                    on_inbound=on_inbound,
                )
            )
        else:
            log.warning(
                "messaging.slack is enabled but SLACK_BOT_TOKEN/SLACK_APP_TOKEN are not set — skipping"
            )

    return built
