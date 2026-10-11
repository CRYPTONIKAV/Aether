"""The flat tool namespace the agent sees and executes through.

Native tools register with an async handler; MCP tools are discovered from
the host and routed back through it. One namespace, one dispatch point —
the agent loop passes every executed name through authz first, then hands
it here.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from ..llm.types import ToolSpec
from .base import UnknownToolError
from .mcp_host import MCPHost

log = logging.getLogger("aether.connectors.registry")

NativeHandler = Callable[[dict[str, Any]], Awaitable[Any]]

# grammatical filler — without it, "[message from X via Y]" prefixes hand the
# gate 'from'/'via'/'you', which intersect with the same words in every prose
# tool description and pass nearly every schema through on every turn
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "you",
        "your",
        "me",
        "my",
        "we",
        "our",
        "i",
        "they",
        "them",
        "from",
        "via",
        "with",
        "at",
        "by",
        "as",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        "might",
        "have",
        "has",
        "not",
        "no",
        "if",
        "what",
        "which",
        "how",
        "all",
        "any",
        "some",
    }
)


def _tokens(text: str) -> set[str]:
    """Significant lowercase word tokens — no stopwords, no one-char junk
    (a 1-char query would substring-match nearly every description)."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOPWORDS}


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    kind: Literal["native", "mcp"]
    handler: NativeHandler | None = None


def _gmail_body_breaks(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """The deterministic fix for email line breaks at the MCP boundary.

    Composio's Gmail actions carry an is_html flag — as HTML, a body written
    with plain \\n newlines renders as one collapsed line, and a doubly
    escaped \\n shows up as literal text. Whatever the model wrote, the
    line breaks must survive: unescape, then \\n becomes <br> with is_html
    set. Idempotent, so the approval path's re-execution is a no-op."""
    if not name.startswith("composio__GMAIL_"):
        return params
    body = params.get("body")
    if not isinstance(body, str) or ("\\n" not in body and "\n" not in body):
        return params
    return {
        **params,
        "body": body.replace("\\n", "\n").replace("\n", "<br>"),
        "is_html": True,
    }


class ToolRegistry:
    """Everything the model can call, under one flat set of names."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._mcp_host: MCPHost | None = None

    # -- registration ----------------------------------------------------------

    def add_native(self, spec: ToolSpec, handler: NativeHandler) -> None:
        existing = self._tools.get(spec.name)
        if existing is not None and existing.kind == "native":
            raise ValueError(f"native tool already registered: {spec.name}")
        if existing is not None:
            log.warning("mcp tool %s is shadowed by a native tool of the same name", spec.name)
            return
        self._tools[spec.name] = RegisteredTool(spec, "native", handler)

    def attach_mcp(self, host: MCPHost) -> None:
        self._mcp_host = host

    def sync_mcp_tools(self) -> int:
        """Pull the host's current tool list into the namespace. Idempotent —
        safe to call repeatedly (e.g. whenever a server (re)connects). A tool
        that vanished from a still-ready server leaves with it — the
        vocabulary never overclaims. (A server that isn't ready reports
        nothing, so its tools stay until it answers or is unlinked.)"""
        if self._mcp_host is None:
            return 0
        added = 0
        reported: dict[str, set[str]] = {}
        for spec in self._mcp_host.tool_specs():
            reported.setdefault(spec.source, set()).add(spec.name)
            existing = self._tools.get(spec.name)
            if existing is not None:
                if existing.kind == "mcp":
                    self._tools[spec.name] = RegisteredTool(spec, "mcp")
                else:
                    log.warning(
                        "mcp tool %s is shadowed by a native tool of the same name",
                        spec.name,
                    )
                continue
            self._tools[spec.name] = RegisteredTool(spec, "mcp")
            added += 1
        for server, names in reported.items():
            prefix = f"{server}__"
            for name in list(self._tools):
                tool = self._tools[name]
                if tool.kind == "mcp" and name.startswith(prefix) and name not in names:
                    del self._tools[name]
                    log.info("mcp tool %s left the namespace — its server no longer lists it", name)
        return added

    # -- use ----------------------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def gated_specs(self, text: str = "", unlocked: set[str] | None = None) -> list[ToolSpec]:
        """Token-frugal tool list: every native tool, plus the MCP tools
        whose names match words in the trigger text or that a search_tools
        call unlocked this turn. Composio ships ~60 schemas per server;
        passing all of them on every call is what blows the free-tier input
        quota, so the model only sees the apps this turn actually mentions."""
        unlocked = unlocked or set()
        specs: list[ToolSpec] = []
        words = _tokens(text)
        for tool in self._tools.values():
            if tool.kind == "native":
                specs.append(tool.spec)
                continue
            if tool.spec.name in unlocked:
                specs.append(tool.spec)
                continue
            if words & _tokens(f"{tool.spec.name} {tool.spec.description}"):
                specs.append(tool.spec)
        return specs

    def search(self, query: str, limit: int = 8) -> list[ToolSpec]:
        """Keyword search over every registered tool (native included) —
        backs the search_tools meta-tool the model uses to pull in schemas
        the keyword gate didn't offer."""
        words = _tokens(query)
        if not words:
            return []
        scored: list[tuple[int, ToolSpec]] = []
        for tool in self._tools.values():
            hits = len(words & _tokens(f"{tool.spec.name} {tool.spec.description}"))
            if hits:
                scored.append((hits, tool.spec))
        scored.sort(key=lambda pair: -pair[0])
        return [spec for _, spec in scored[:limit]]

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def has_server(self, server: str) -> bool:
        """True when any tool from this MCP server is in the namespace —
        distinguishes 'the app isn't connected' from 'the action isn't
        available in a linked app'."""
        prefix = f"{server}__"
        return any(name.startswith(prefix) for name in self._tools)

    def mcp_servers(self) -> list[str]:
        """The MCP servers live in the namespace right now — the honest
        answer to 'which apps are connected?'."""
        seen: set[str] = set()
        for name, tool in self._tools.items():
            if tool.kind != "mcp":
                continue
            server, sep, _ = name.partition("__")
            if sep:
                seen.add(server)
        return sorted(seen)

    def mcp_status(self) -> dict[str, bool]:
        """Per-server readiness, straight off the live host — the honest
        answer to 'is it actually up?' (a configured server that never
        answered is still not connected, whatever the config says)."""
        if self._mcp_host is None:
            return {}
        return {conn.name: conn.ready for conn in self._mcp_host.connections}

    def server_action_count(self, server: str) -> int:
        """How many of the namespace's actions come from this server."""
        return sum(1 for t in self._tools.values() if t.kind == "mcp" and t.spec.source == server)

    def drop_server(self, server: str) -> int:
        """Remove every tool an MCP server contributed. The other half of
        sync_mcp_tools: an app unlinked or disabled at runtime must not keep
        its actions callable."""
        prefix = f"{server}__"
        names = [
            name
            for name in self._tools
            if name.startswith(prefix) and self._tools[name].kind == "mcp"
        ]
        for name in names:
            del self._tools[name]
        if names:
            log.info("dropped %d tool(s) from mcp server %r", len(names), server)
        return len(names)

    def __len__(self) -> int:
        return len(self._tools)

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(f"unknown tool: {name}")
        if tool.kind == "native":
            assert tool.handler is not None
            return _stringify(await tool.handler(params or {}))
        if self._mcp_host is None:
            raise UnknownToolError(f"mcp tool {name} has no host to route through")
        return await self._mcp_host.call(name, _gmail_body_breaks(name, params or {}))


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)
