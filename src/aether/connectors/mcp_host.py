"""The MCP host: one long-lived client session per configured server.

Every server in config.yaml `mcp_servers` is connected at boot over stdio
or streamable HTTP, and each tool it exposes enters the flat namespace as
`<server>__<tool>`. Sessions are held open and reconnected with backoff on
failure — a dead or misconfigured server logs a warning and keeps retrying
but never takes the agent down with it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ..config import MCPServerConfig, PollTool
from ..llm.types import ToolSpec, with_plain_param
from ..secret_env import EnvResolver
from .base import ConnectorUnavailableError, UnknownToolError

log = logging.getLogger("aether.connectors.mcp")

DEFAULT_READY_TIMEOUT = 10.0
_MAX_BACKOFF = 30.0
# how often a live session re-asks the tool list — a connection hub grows as
# the user approves apps, and its new actions must reach the namespace
# without a reconnect
_RELIST_SECONDS = 60.0


def _describe(exc: BaseException) -> str:
    """One line that carries the actual cause. str() of an ExceptionGroup is
    just "unhandled errors in a TaskGroup (1 sub-exception)" — the cause
    lives in the leaves, so unwrap groups (recursively) for the log line."""
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_describe(e) for e in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


class MCPServerConnection:
    """One server: session lifecycle, tool listing, and calls.

    The session lives inside a background task (`_run`): open the
    transport, initialize, list tools, signal ready, then park until either
    `stop()` or a failed call breaks the hold — at which point the loop
    reconnects with exponential backoff.
    """

    def __init__(self, config: MCPServerConfig, resolver: EnvResolver | None = None) -> None:
        self.name = config.name
        self.poll_tools: list[PollTool] = list(config.poll_tools)
        # a deep copy: the live config object can be edited in place from
        # chat (an enabled flip, a transport change) — the connection must
        # keep running the config it booted with, so a later diff sees the
        # change instead of the session silently mutating under itself
        self._config = config.model_copy(deep=True)
        self._resolver = resolver
        self._ready = asyncio.Event()
        self._hold = asyncio.Event()
        self._stopping = False
        self._task: asyncio.Task[None] | None = None
        self._session: Any = None
        self._tools: dict[str, Any] = {}
        self._tool_version = 0  # bumped whenever the tool-name set changes

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name=f"mcp:{self.name}")

    async def stop(self) -> None:
        self._stopping = True
        self._hold.set()
        task, self._task = self._task, None
        if task is not None:
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
        self._session = None
        self._ready.clear()

    async def _run(self) -> None:
        backoff = 1.0
        try:
            while not self._stopping:
                self._hold.clear()
                try:
                    async with self._open_session() as session:
                        await session.initialize()
                        self._session = session
                        result = await session.list_tools()
                        self._absorb(result)
                        self._ready.set()
                        log.info("mcp server %s ready — %d tools", self.name, len(self._tools))
                        backoff = 1.0
                        while not self._stopping and not self._hold.is_set():
                            with contextlib.suppress(TimeoutError):
                                # a parked wait with a heartbeat: every cadence
                                # the tool list is re-asked — a connection hub
                                # grows as the user approves apps, and its new
                                # actions must reach the namespace without a
                                # reconnect
                                await asyncio.wait_for(self._hold.wait(), timeout=_RELIST_SECONDS)
                            if self._hold.is_set() or self._stopping:
                                break
                            await self._relist(session)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if not self._stopping:
                        log.warning(
                            "mcp server %s connection problem: %s", self.name, _describe(exc)
                        )
                finally:
                    self._session = None
                    self._ready.clear()
                if self._stopping:
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)
        finally:
            self._session = None
            self._ready.clear()

    @asynccontextmanager
    async def _open_session(self) -> AsyncIterator[Any]:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        transport = self._config.transport
        if transport.type == "stdio":
            from mcp import StdioServerParameters

            params = StdioServerParameters(
                command=transport.command or "",
                args=list(transport.args),
                env=await self._expand(transport.env),
            )
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                yield session
        else:
            from mcp.client.streamable_http import streamable_http_client

            url = transport.url or ""
            if self._resolver is not None and "$" in url:
                # a hub's URL carries the user's key in it — it lives in .env
                # by name and re-resolves at every (re)connect, exactly like
                # env values and headers
                url = await self._resolver.expand_value(url)
            http_client = None
            if transport.headers:
                # mcp's own factory, never a bare AsyncClient: the factory
                # carries the transports' tuned timeouts (30s general, 300s
                # SSE read). A default client's 5s read timeout kills the
                # idle listen stream seconds after connect and flaps the
                # session in a reconnect loop.
                from mcp.shared._httpx_utils import create_mcp_http_client

                http_client = create_mcp_http_client(headers=await self._expand(transport.headers))
            try:
                async with (
                    streamable_http_client(url, http_client=http_client) as (
                        read,
                        write,
                    ),
                    ClientSession(read, write) as session,
                ):
                    yield session
            finally:
                if http_client is not None:
                    await http_client.aclose()

    # -- readiness ------------------------------------------------------------

    async def _expand(self, mapping: dict[str, str] | None) -> dict[str, str] | None:
        """$NAME references resolved at open time — every (re)connect re-reads
        .env through the resolver, so a key added after boot needs no
        restart, and a missing one fails this attempt with its name in the
        log (the SDK's stdio children otherwise only ever see a safe
        whitelist, never the parent's environment)."""
        if not mapping:
            return dict(mapping) if mapping else None
        if self._resolver is None:
            return dict(mapping)
        return await self._resolver.expand_map(dict(mapping))

    async def wait_ready(self, timeout: float = DEFAULT_READY_TIMEOUT) -> bool:
        if self._task is None:
            return False
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
            return True
        except TimeoutError:
            return False

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def config(self) -> MCPServerConfig:
        """The config this connection is running — a pinned copy, so a diff
        against the live list sees chat-made changes (see __init__)."""
        return self._config

    @property
    def tool_version(self) -> int:
        """Bumped whenever the tool-name set changed — lets the agent loop
        notice a hub that grew without diffing full tool lists each tick."""
        return self._tool_version

    # -- tools ----------------------------------------------------------------

    def _absorb(self, result: Any) -> None:
        tools = {tool.name: tool for tool in result.tools}
        if set(tools) != set(self._tools):
            self._tools = tools
            self._tool_version += 1

    async def _relist(self, session: Any) -> None:
        try:
            result = await session.list_tools()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("mcp server %s tool re-list failed: %s", self.name, exc)
            return
        before = len(self._tools)
        self._absorb(result)
        if len(self._tools) != before:
            log.info("mcp server %s — tool list changed, %d tools now", self.name, len(self._tools))

    def tool_specs(self) -> list[ToolSpec]:
        """Current tools with their flat-namespace names. Empty until connected."""
        specs: list[ToolSpec] = []
        for tool in self._tools.values():
            schema = getattr(tool, "input_schema", None) or {
                "type": "object",
                "properties": {},
            }
            description = (getattr(tool, "description", "") or "").strip()
            specs.append(
                ToolSpec(
                    name=f"{self.name}__{tool.name}",
                    description=description or f"{self.name} tool {tool.name}",
                    input_schema=with_plain_param(schema),
                    source=self.name,
                )
            )
        return specs

    async def call(self, tool: str, args: dict[str, Any]) -> str:
        if self._stopping or self._task is None:
            raise ConnectorUnavailableError(f"mcp server {self.name} is not running")
        if not await self.wait_ready():
            raise ConnectorUnavailableError(f"mcp server {self.name} is not connected")
        session = self._session
        if session is None:  # lost the race with a disconnect
            raise ConnectorUnavailableError(f"mcp server {self.name} is not connected")
        try:
            result = await session.call_tool(tool, args or {})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # the session is suspect — break the hold so the runner reconnects
            self._hold.set()
            raise ConnectorUnavailableError(f"mcp call {self.name}__{tool} failed: {exc}") from exc
        return _flatten(result)


def _flatten(result: Any) -> str:
    """A CallToolResult as plain text for the agent to read."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            parts.append(str(getattr(block, "text", "")))
        elif kind == "image":
            parts.append("[image omitted]")
        else:
            parts.append(f"[{kind}]")
    if getattr(result, "is_error", False):
        return "error: " + ("\n".join(parts) or "the tool reported an error")
    if parts:
        return "\n".join(parts)
    structured = getattr(result, "structured_content", None)
    return json.dumps(structured) if structured else "(no content)"


class MCPHost:
    """All configured servers: start/stop, tool listing, and call routing."""

    def __init__(self, servers: list[MCPServerConfig], resolver: EnvResolver | None = None) -> None:
        self._resolver = resolver
        self._connections = [MCPServerConnection(c, resolver) for c in servers if c.enabled]
        self._by_name = {c.name: c for c in self._connections}

    @property
    def connections(self) -> list[MCPServerConnection]:
        return list(self._connections)

    def start(self) -> None:
        for conn in self._connections:
            conn.start()

    async def apply_servers(self, servers: list[MCPServerConfig]) -> dict[str, Any]:
        """Make the live set match `servers`: stop what left or changed,
        start what's new, keep unchanged sessions open (a reconnect costs a
        cold start; an untouched server must never pay it). Disabled
        servers never spawn. Fresh sessions get a moment to come ready so
        the caller's immediate tool sync sees what they expose — a server
        that doesn't make it stays out of the namespace and keeps retrying.

        Returns {"started": [connections], "stopped": [names]} — the names a
        caller should drop from any derived state (the tool namespace, poll
        targets)."""
        desired = [c for c in servers if c.enabled]
        current = {c.name: c for c in self._connections}
        new_conns: list[MCPServerConnection] = []
        started: list[MCPServerConnection] = []
        stopped: list[str] = []
        for config in desired:
            existing = current.get(config.name)
            if existing is not None and existing.config == config:
                new_conns.append(existing)  # unchanged — keep the live session
                continue
            if existing is not None:
                await existing.stop()
                stopped.append(config.name)
            conn = MCPServerConnection(config, self._resolver)
            conn.start()
            new_conns.append(conn)
            started.append(conn)
        for name, conn in current.items():
            if name not in {c.name for c in desired}:
                await conn.stop()
                stopped.append(name)
        self._connections = new_conns
        self._by_name = {c.name: c for c in new_conns}
        if started:
            # best effort, concurrently — a slow server costs one timeout,
            # not one per server
            await asyncio.gather(*(c.wait_ready() for c in started))
        return {"started": started, "stopped": stopped}

    async def stop(self) -> None:
        for conn in self._connections:
            await conn.stop()

    def tool_specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for conn in self._connections:
            specs.extend(conn.tool_specs())
        return specs

    async def call(self, qualified_name: str, args: dict[str, Any]) -> str:
        server, sep, tool = qualified_name.partition("__")
        if not sep:
            raise UnknownToolError(
                f"mcp tool names look like <server>__<tool>: got {qualified_name!r}"
            )
        conn = self._by_name.get(server)
        if conn is None:
            raise UnknownToolError(f"no configured mcp server named {server!r}")
        return await conn.call(tool, args)

    async def wait_all_ready(self, timeout: float = 30.0) -> dict[str, bool]:
        """Per-server readiness, all waited for concurrently."""
        if not self._connections:
            return {}

        async def one(conn: MCPServerConnection) -> tuple[str, bool]:
            return conn.name, await conn.wait_ready(timeout)

        results = await asyncio.gather(*(one(c) for c in self._connections))
        return dict(results)
