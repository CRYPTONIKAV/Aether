"""ToolRegistry tests — flat namespace, dispatch, shadowing. All fakes."""

from __future__ import annotations

import pytest

from aether.connectors.base import UnknownToolError
from aether.connectors.registry import RegisteredTool, ToolRegistry
from aether.llm.types import ToolSpec


class FakeHost:
    """MCPHost double: canned specs, recorded calls."""

    def __init__(self, specs: list[ToolSpec]) -> None:
        self.specs_list = specs
        self.calls: list[tuple[str, dict]] = []

    def tool_specs(self) -> list[ToolSpec]:
        return self.specs_list

    async def call(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        return f"mcp:{name}"


def _native(name: str, source: str = "native") -> ToolSpec:
    return ToolSpec(name=name, description=f"native {name}", source=source)


async def _echo_handler(params: dict) -> dict:
    return {"echo": params.get("text", "")}


async def test_native_tools_execute_and_stringify() -> None:
    registry = ToolRegistry()
    registry.add_native(_native("memory_search"), _echo_handler)

    spec = registry.specs()[0]
    assert spec.source == "native"  # the default
    assert await registry.execute("memory_search", {"text": "hi"}) == '{"echo": "hi"}'


async def test_native_string_results_pass_through() -> None:
    async def plain(params: dict) -> str:
        return "kept as-is"

    registry = ToolRegistry()
    registry.add_native(_native("plain"), plain)
    assert await registry.execute("plain", {}) == "kept as-is"


async def test_duplicate_native_registration_is_rejected() -> None:
    registry = ToolRegistry()
    registry.add_native(_native("t"), _echo_handler)
    with pytest.raises(ValueError):
        registry.add_native(_native("t"), _echo_handler)


async def test_mcp_tools_sync_into_the_namespace() -> None:
    registry = ToolRegistry()
    host = FakeHost([ToolSpec(name="mail__list_unread", description="unread mail", source="mail")])
    registry.attach_mcp(host)
    assert registry.sync_mcp_tools() == 1
    assert registry.sync_mcp_tools() == 0  # idempotent

    spec = registry.get("mail__list_unread")
    assert spec is not None and spec.kind == "mcp"
    assert registry.specs()[0].source == "mail"

    assert await registry.execute("mail__list_unread", {"limit": 2}) == "mcp:mail__list_unread"
    assert host.calls == [("mail__list_unread", {"limit": 2})]


async def test_native_tools_shadow_same_named_mcp_tools() -> None:
    registry = ToolRegistry()
    registry.add_native(_native("mail__list_unread"), _echo_handler)
    host = FakeHost([ToolSpec(name="mail__list_unread", description="mcp one", source="mail")])
    registry.attach_mcp(host)
    registry.sync_mcp_tools()

    assert await registry.execute("mail__list_unread", {"text": "x"}) == '{"echo": "x"}'
    assert host.calls == []  # the native tool won


async def test_unknown_tools_do_not_execute() -> None:
    registry = ToolRegistry()
    with pytest.raises(UnknownToolError):
        await registry.execute("ghost", {})


async def test_mcp_tool_without_host_is_an_error() -> None:
    # defensive branch: an mcp entry can only exist via sync_mcp_tools, which
    # requires a host — but if one ever routes with no host attached it
    # must fail loudly, not silently do nothing
    registry = ToolRegistry()
    registry.add_native(_native("t"), _echo_handler)
    registry._tools["srv__x"] = RegisteredTool(
        ToolSpec(name="srv__x", description="", source="srv"), "mcp"
    )
    with pytest.raises(UnknownToolError):
        await registry.execute("srv__x", {})


def _gmail_registry() -> tuple[ToolRegistry, FakeHost]:
    registry = ToolRegistry()
    host = FakeHost(
        [ToolSpec(name="composio__GMAIL_SEND_EMAIL", description="send", source="composio")]
    )
    registry.attach_mcp(host)
    registry.sync_mcp_tools()
    return registry, host


async def test_gmail_bodies_get_html_line_breaks() -> None:
    # a model-written body carries line breaks as \n — as HTML that renders
    # as one collapsed line, so the boundary converts and flags it
    registry, host = _gmail_registry()
    await registry.execute(
        "composio__GMAIL_SEND_EMAIL",
        {"recipient_email": "a@b.c", "body": "hi\nthis is mail\n", "is_html": True},
    )
    assert host.calls[0][1]["body"] == "hi<br>this is mail<br>"
    assert host.calls[0][1]["is_html"] is True


async def test_a_doubly_escaped_newline_is_fixed_the_same_way() -> None:
    # the other failure shape: a literal backslash-n lands in the body and
    # shows up in the email as text — unescaped first, then converted
    registry, host = _gmail_registry()
    await registry.execute(
        "composio__GMAIL_SEND_EMAIL", {"recipient_email": "a@b.c", "body": "hi\\nthis is mail"}
    )
    assert host.calls[0][1]["body"] == "hi<br>this is mail"
    assert host.calls[0][1]["is_html"] is True


async def test_the_transform_is_idempotent() -> None:
    # the approval path re-executes stored params — the second pass must be a no-op
    registry, host = _gmail_registry()
    await registry.execute("composio__GMAIL_SEND_EMAIL", {"body": "hi<br>there", "is_html": True})
    assert host.calls[0][1] == {"body": "hi<br>there", "is_html": True}


async def test_bodies_without_newlines_pass_through_untouched() -> None:
    registry, host = _gmail_registry()
    await registry.execute("composio__GMAIL_SEND_EMAIL", {"body": "one line", "is_html": False})
    assert host.calls[0][1] == {"body": "one line", "is_html": False}


async def test_non_gmail_mcp_tools_are_untouched() -> None:
    # <br> would be wrong in a plain-text or mrkdwn surface — the fix is
    # gmail-shaped on purpose
    registry = ToolRegistry()
    host = FakeHost([ToolSpec(name="mail__send_message", description="send", source="mail")])
    registry.attach_mcp(host)
    registry.sync_mcp_tools()
    await registry.execute("mail__send_message", {"body": "hi\nthere"})
    assert host.calls[0][1] == {"body": "hi\nthere"}


async def test_drop_server_removes_only_that_servers_tools() -> None:
    """The other half of sync_mcp_tools: an app unlinked or disabled from
    chat must not keep its actions callable — and nobody else's drop."""
    registry = ToolRegistry()
    registry.add_native(_native("memory_search"), _echo_handler)
    host = FakeHost(
        [
            ToolSpec(name="mail__list_unread", description="a", source="mail"),
            ToolSpec(name="mail__send_email", description="b", source="mail"),
            ToolSpec(name="calendar__get_events", description="c", source="calendar"),
        ]
    )
    registry.attach_mcp(host)
    assert registry.sync_mcp_tools() == 3
    assert len(registry) == 4

    assert registry.drop_server("mail") == 2
    assert len(registry) == 2
    assert registry.get("mail__list_unread") is None
    assert registry.get("mail__send_email") is None
    with pytest.raises(UnknownToolError):
        await registry.execute("mail__send_email", {})  # gone means not callable
    # the neighbors and the natives are untouched
    assert registry.get("calendar__get_events") is not None
    assert registry.get("memory_search") is not None
    # the __ delimiter keeps a prefixing name from over-matching
    assert registry.drop_server("cal") == 0
    assert registry.get("calendar__get_events") is not None
    # and dropping what's already gone is a no-op
    assert registry.drop_server("mail") == 0


def _gated_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.add_native(_native("memory_search"), _echo_handler)
    host = FakeHost(
        [
            ToolSpec(
                name="hub__GMAIL_SEND_EMAIL",
                description="Send an email to a recipient",
                source="hub",
            ),
            ToolSpec(
                name="hub__CALENDAR_LIST_EVENTS",
                description="List events from your calendar",
                source="hub",
            ),
        ]
    )
    registry.attach_mcp(host)
    registry.sync_mcp_tools()
    return registry


def test_gate_ignores_stopwords_from_message_prefixes() -> None:
    registry = _gated_registry()
    # "[message from X via Y]" hands the gate from/via/you for free — those
    # must not match the same filler words in every prose description
    offered = {
        s.name
        for s in registry.gated_specs("[message from sam via telegram]\ncan you check this for me")
    }
    assert offered == {"memory_search"}  # natives only


def test_gate_matches_real_keywords_and_unlocked_names() -> None:
    registry = _gated_registry()
    offered = {s.name for s in registry.gated_specs("what's on my calendar today")}
    assert "hub__CALENDAR_LIST_EVENTS" in offered
    assert "hub__GMAIL_SEND_EMAIL" not in offered
    # explicit unlock (search_tools matches) wins regardless of text
    offered = {s.name for s in registry.gated_specs("", unlocked={"hub__GMAIL_SEND_EMAIL"})}
    assert "hub__GMAIL_SEND_EMAIL" in offered


def test_search_scores_token_overlap_not_substrings() -> None:
    registry = _gated_registry()
    assert registry.search("a") == []  # one-char junk matches nothing
    assert registry.search("the a") == []  # stopword-only query matches nothing
    found = [s.name for s in registry.search("calendar events")]
    assert found[0] == "hub__CALENDAR_LIST_EVENTS"  # both tokens hit
    assert "hub__GMAIL_SEND_EMAIL" not in found
