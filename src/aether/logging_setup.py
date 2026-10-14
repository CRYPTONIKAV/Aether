"""Logging configuration for Aether."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

# any token riding a URL — the WebSocket handshake (browsers can't set
# headers there), a curl pasted by hand — matches this; the value never
# reaches a handler
_TOKEN_IN_URL = re.compile(r"(?i)(token=)[^&\s\"']+")


class _NoTokensInLogs(logging.Filter):
    """The access log prints whole request lines, so a `?token=` in the
    query string would land in the logs verbatim. This scrub rewrites every
    such value to *** on the record itself, before any handler sees it."""

    # walks whatever shape a logger put in msg/args — the scrub returns the
    # same structure with every token value rewritten
    def _scrub(self, value: Any) -> Any:
        if isinstance(value, str):
            return _TOKEN_IN_URL.sub(r"\1***", value)
        if isinstance(value, dict):
            return {k: self._scrub(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self._scrub(v) for v in value)
        if isinstance(value, list):
            return [self._scrub(v) for v in value]
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._scrub(record.msg)
        record.args = self._scrub(record.args)
        return True


def _redact_tokens_from_access_log() -> None:
    """Idempotent — the serve loop rebuilds the app on every chat-made
    restart, and each pass must not stack another filter."""
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _NoTokensInLogs) for f in access.filters):
        access.addFilter(_NoTokensInLogs())


def configure_logging() -> None:
    level = os.environ.get("AETHER_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    # Quiet the chatty third-party loggers
    for name in ("httpx", "httpcore", "asyncio", "slack_bolt", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)
    _redact_tokens_from_access_log()
