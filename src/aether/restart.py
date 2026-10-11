"""The self-restart switch — llm.* config changes apply on a fresh boot,
and nobody should have to restart Aether by hand.

``main()`` serves in a loop: each iteration runs one uvicorn server over a
fresh app instance (fresh lifespan, fresh boot merge), and re-registers it
here before serving. A config change to the llm section calls ``request()``;
a background task waits out the delay — long enough for the current turn to
land its words in chat — then sets the server's ``should_exit``, which starts
uvicorn's graceful shutdown: in-flight requests (websockets included) drain,
the lifespan closes the loop, connectors, host, and pool, and the serve loop
iterates. ``take_pending()`` is what tells the loop to iterate rather than
exit — a real SIGINT/SIGTERM never arms a restart, so it always exits.

When no server is registered (unit tests, the e2e dev fixture) ``request()``
returns False and the change simply applies on the next boot — a restart
request must never crash the process making it. A restart request while one
is already armed is a no-op: one restart in flight, and the fresh boot
re-reads everything anyway.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger("aether.restart")

_server: Any = None
_pending = False
_task: asyncio.Task[None] | None = None


def register(server: Any) -> None:
    """The server the current serve-loop iteration is running — re-called
    by main() before every run, so a stale handle from the last iteration is
    never the one told to exit."""
    global _server
    _server = server


def request(delay_seconds: float = 5.0) -> bool:
    """Arm a graceful self-restart once the current turn has had time to
    land its words. False when no server is registered — the change still
    persists, it just applies on the next boot. Idempotent while armed."""
    global _pending, _task
    if _server is None:
        log.warning(
            "restart requested but no server is registered — the change applies on the next boot"
        )
        return False
    if _pending:
        return True  # one restart in flight; the fresh boot re-reads everything

    async def _exit_later() -> None:
        await asyncio.sleep(delay_seconds)
        log.info("restarting to apply new configuration")
        _server.should_exit = True

    # the task is kept on the module so it can't be garbage-collected mid-sleep
    _pending = True
    _task = asyncio.get_running_loop().create_task(_exit_later(), name="aether-restart")
    return True


def take_pending() -> bool:
    """Consume the restart flag — the serve loop's answer to "iterate or
    exit?". A plain shutdown (SIGINT/SIGTERM) never armed it: exit."""
    global _pending, _task
    pending, _pending = _pending, False
    _task = None
    return pending
