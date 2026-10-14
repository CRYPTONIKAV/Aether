"""ConfigManager tests — boot merge, set/reset roundtrips, plain-word
refusals, and the live-apply hooks. FakeConfigStore, no database.

The manager is the one interpretation both chat paths share (the /config
command and the set_config tool), so what's pinned here is exactly what the
user sees: in-place mutation (section objects keep their identity — Salience
and the EventStore hold references to them), the apply notes, and the
refusals that must never persist anything.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fakes import FakeApprovals, FakeConfigStore, FakeSurfaceConnector

import aether.config_store
from aether.agent.loop import SurfaceFanout
from aether.authz.policy import Policy
from aether.config import AppConfig, AuthzRule, ContactRule, MCPServerConfig, Settings
from aether.config_store import ConfigError, ConfigManager
from aether.connectors.registry import ToolRegistry
from aether.llm.types import ToolSpec
from aether.secret_env import EnvResolver


def _settings() -> Settings:
    # _env_file=None: these tests never read .env — secrets stay out of it
    return Settings(_env_file=None)


def _manager(
    rows: dict | None = None, live: AppConfig | None = None
) -> tuple[ConfigManager, FakeConfigStore, AppConfig]:
    store = FakeConfigStore(rows)
    config = live if live is not None else AppConfig()
    return ConfigManager(store, _settings(), config), store, config


class FakeLoop:
    """The agent loop as the apply hooks see it: quiet-hours reparse, poll
    rebuild, the policy slot the authz hook swaps, and the two callbacks the
    messaging factory receives."""

    def __init__(self) -> None:
        self.reparsed = 0
        self.rebuilt = 0
        self._policy = None

    def reparse_quiet_hours(self) -> None:
        self.reparsed += 1

    def rebuild_poll_targets(self) -> None:
        self.rebuilt += 1

    async def execute_decision(self, *args, **kwargs) -> None:  # never called here
        pass

    async def handle_inbound(self, *args, **kwargs) -> None:  # never called here
        pass


class FakeHost:
    """MCPHost double for the apply hook: records the server list it was
    handed, serves each configured app one action (the post-apply view of
    the namespace), answers a canned start/stop diff."""

    def __init__(self, diff: dict) -> None:
        self.diff = diff
        self.applied: list[list[str]] = []
        self.specs: list[ToolSpec] = []

    async def apply_servers(self, servers: list) -> dict:
        self.applied.append([s.name for s in servers])
        self.specs = [
            ToolSpec(name=f"{s.name}__act", description=f"{s.name} action", source=s.name)
            for s in servers
        ]
        return self.diff

    def tool_specs(self) -> list[ToolSpec]:
        return self.specs


async def _registry_with(host: FakeHost, host_specs: list[ToolSpec]) -> ToolRegistry:
    """A registry holding a native neighbor plus whatever the host serves."""

    async def noop(params: dict) -> str:
        return "native"

    registry = ToolRegistry()
    registry.add_native(
        ToolSpec(name="memory_search", description="native neighbor", source="native"), noop
    )
    host.specs = host_specs
    registry.attach_mcp(host)
    registry.sync_mcp_tools()
    return registry


# -- boot merge -----------------------------------------------------------------


async def test_boot_merge_mutates_in_place_keeping_section_identity() -> None:
    live = AppConfig()
    agent_section, salience_section = live.agent, live.salience
    manager, _, _ = _manager(rows={"agent.tick_seconds": 10, "salience.threshold": 5.0}, live=live)

    assert await manager.bootstrap() == 2
    assert live.agent is agent_section  # Salience/EventStore hold these refs
    assert live.salience is salience_section
    assert live.agent.tick_seconds == 10.0
    assert live.salience.threshold == 5.0


async def test_boot_merge_composes_item_rows_after_the_list_root() -> None:
    live = AppConfig(mcp_servers=[MCPServerConfig(name="mail")])
    rows = {
        "mcp_servers": [{"name": "mail"}, {"name": "calendar"}],
        "mcp_servers.mail.enabled": False,
    }
    manager, _, _ = _manager(rows=rows, live=live)

    assert await manager.bootstrap() == 2
    assert [s.name for s in live.mcp_servers] == ["mail", "calendar"]
    # the item row landed after the root row — reversed, the root row's mail
    # entry (enabled) would have overwritten the flip
    assert live.mcp_servers[0].enabled is False
    assert live.mcp_servers[1].enabled is True


async def test_boot_merge_skips_stale_rows_without_crashing() -> None:
    live = AppConfig()
    rows = {
        "agent.nonexistent": 1,  # the vocabulary moved on
        "agent.tick_seconds": "garbage",  # the value no longer validates
        "salience.threshold": 5.0,
    }
    manager, store, _ = _manager(rows=rows, live=live)

    assert await manager.bootstrap() == 1
    assert live.agent.tick_seconds == 30.0  # the default, untouched
    assert live.salience.threshold == 5.0
    assert store.rows == rows  # skipped rows stay — a yaml revert can resurrect them


async def test_boot_merge_applies_an_explicit_none() -> None:
    live = AppConfig()
    live.agent.quiet_hours = "23:00-07:00"  # the yaml names a window
    manager, _, _ = _manager(rows={"agent.quiet_hours": None}, live=live)

    assert await manager.bootstrap() == 1
    assert live.agent.quiet_hours is None  # chat cleared it


# -- set / reset roundtrips ------------------------------------------------------


async def test_set_then_reset_roundtrips_through_the_store() -> None:
    manager, store, live = _manager()

    reply = await manager.set(op="set", path="agent.tick_seconds", value=10)
    assert (
        reply == "agent.tick_seconds set to 10 (config.yaml says 30). — applies from the next tick"
    )
    assert store.rows == {"agent.tick_seconds": 10.0}
    assert live.agent.tick_seconds == 10.0

    reply = await manager.set(op="reset", path="agent.tick_seconds")
    assert reply == "agent.tick_seconds reset to config.yaml (30). — applies from the next tick"
    assert store.rows == {}
    assert live.agent.tick_seconds == 30.0


async def test_set_explicit_none_clears_the_setting() -> None:
    manager, store, live = _manager()
    live.llm.vision_model = "qwen-vision"  # some earlier value, not from chat

    reply = await manager.set(op="set", path="llm.vision_model", value=None)
    assert "llm.vision_model set to none" in reply
    # no server is registered in tests, so the self-restart degrades honestly
    assert reply.endswith(" — it loads on the next boot")
    assert store.rows == {"llm.vision_model": None}
    assert live.llm.vision_model is None


async def test_set_refuses_bad_values_and_persists_nothing() -> None:
    manager, store, live = _manager()

    assert await manager.set(op="set", path="agent.tick_seconds", value="garbage") == (
        "agent.tick_seconds can't be 'garbage' — it needs to be a number."
    )
    assert await manager.set(op="set", path="agent.max_tool_iterations", value="2.5") == (
        "agent.max_tool_iterations can't be '2.5' — it needs to be a whole number."
    )
    assert await manager.set(op="set", path="messaging.telegram.enabled", value="maybe") == (
        "messaging.telegram.enabled can't be 'maybe' — it needs to be true or false."
    )
    reply = await manager.set(op="set", path="contacts.mode", value="strict")
    assert "contacts.mode can't be 'strict' — it must be one of" in reply

    assert store.rows == {}  # a refusal never persists
    assert live.agent.tick_seconds == 30.0
    assert live.agent.max_tool_iterations == 12
    assert live.messaging.telegram.enabled is False
    assert live.contacts.mode == "off"


async def test_bad_operations_and_shapes_refused_in_plain_words() -> None:
    manager, store, _ = _manager()

    assert await manager.set(op="frobnicate", path="agent.tick_seconds") == (
        "the operation is one of set, add, remove, or reset."
    )
    assert await manager.set(op="set", path="") == "which setting? e.g. agent.tick_seconds."
    assert await manager.set(op="add", path="agent.tick_seconds", value=1) == (
        "add/remove work on lists — agent.tick_seconds is a single setting; "
        "use set (or reset to go back to config.yaml)."
    )
    assert await manager.set(op="add", path="contacts.allowlist") == (
        "add needs a value after contacts.allowlist — e.g. "
        "/config add contacts.allowlist telegram @friend."
    )
    assert store.rows == {}


async def test_malformed_paths_refuse_with_the_right_shape_hint() -> None:
    manager, _, _ = _manager()

    assert await manager.set(op="set", path="messaging.telegram", value=True) == (
        "messaging paths look like messaging.<platform>.enabled — e.g. messaging.telegram.enabled."
    )
    assert await manager.set(op="set", path="mcp_servers.mail.poll_tools", value=[]) == (
        "mcp_servers paths are 'mcp_servers' (the whole list) or "
        "'mcp_servers.<name>.enabled' — e.g. mcp_servers.mail.enabled."
    )

    live = AppConfig(mcp_servers=[MCPServerConfig(name="mail")])
    manager2, _, _ = _manager(live=live)
    assert await manager2.set(op="set", path="mcp_servers.ghost.enabled", value=False) == (
        "no app named 'ghost' in mcp_servers (mail)."
    )


async def test_unknown_paths_get_one_did_you_mean_guess() -> None:
    manager, _, _ = _manager()

    reply = await manager.set(op="set", path="agent.ticksecond", value=1)
    assert reply == "'agent.ticksecond' isn't a setting I know — did you mean agent.tick_seconds?"

    reply = await manager.set(op="set", path="agent.frobnicate", value=1)
    assert "'agent.frobnicate' isn't a setting I know" in reply


# -- effective / show ------------------------------------------------------------


async def test_effective_serves_sections_and_the_override_rows() -> None:
    manager, _, live = _manager()

    assert await manager.effective() == {
        "sections": live.model_dump(),
        "overrides": [],
    }

    await manager.set(op="set", path="agent.tick_seconds", value=10)
    data = await manager.effective()
    assert data["sections"]["agent"]["tick_seconds"] == 10.0
    assert data["overrides"] == [{"path": "agent.tick_seconds", "value": 10.0}]


async def test_show_marks_what_chat_changed() -> None:
    manager, _, _ = _manager()

    overview = await manager.show()
    assert overview.startswith(
        "my configuration — config.yaml plus whatever you've changed from chat:"
    )
    assert "Nothing changed from config.yaml yet." in overview

    await manager.set(op="set", path="agent.tick_seconds", value=10)
    overview = await manager.show()
    assert "Changed from config.yaml: agent.tick_seconds = 10" in overview

    assert await manager.show("agent.tick_seconds") == (
        "agent.tick_seconds = 10 (changed from config.yaml, which says 30)"
    )
    assert (
        await manager.show("salience.threshold") == "salience.threshold = 6 (config.yaml's value)"
    )

    section = await manager.show("agent")
    assert section.startswith("agent — config.yaml values with your chat changes marked:")
    assert "tick_seconds = 10 (changed from config.yaml)" in section


async def test_show_of_an_unknown_path_is_the_refusal_not_a_crash() -> None:
    manager, _, _ = _manager()
    reply = await manager.show("agent.frobnicate")
    assert "'agent.frobnicate' isn't a setting I know" in reply


# -- yaml_value: what a reset returns to ------------------------------------------


async def test_yaml_value_reads_the_base_even_after_chat_changes() -> None:
    manager, _, _ = _manager()

    assert manager.yaml_value("agent.tick_seconds") == 30.0
    assert manager.yaml_value("salience.threshold") == 6.0

    await manager.set(op="set", path="agent.tick_seconds", value=10)
    assert manager.yaml_value("agent.tick_seconds") == 30.0  # the base stands


async def test_yaml_value_of_a_list_hands_back_the_base_entries() -> None:
    live = AppConfig(contacts={"allowlist": [ContactRule(platform="telegram", handle="@mom")]})
    manager, _, _ = _manager(live=live)

    assert len(manager.yaml_value("contacts.allowlist")) == 1

    await manager.set(op="add", path="contacts.allowlist", value="discord @dad")
    assert len(manager.yaml_value("contacts.allowlist")) == 1  # yaml's entry, still


async def test_yaml_value_of_an_app_item_reads_the_base_app() -> None:
    live = AppConfig(mcp_servers=[MCPServerConfig(name="mail")])
    manager, _, _ = _manager(live=live)

    await manager.set(op="set", path="mcp_servers.mail.enabled", value=False)
    assert manager.yaml_value("mcp_servers.mail.enabled") is True  # yaml's app, on


async def test_yaml_value_of_a_chat_added_app_is_none_it_was_never_in_yaml() -> None:
    manager, _, _ = _manager()

    await manager.set(op="add", path="mcp_servers", value={"name": "mail"})
    assert manager.yaml_value("mcp_servers.mail.enabled") is None


def test_yaml_value_of_an_unknown_path_raises_like_set() -> None:
    manager, _, _ = _manager()
    with pytest.raises(ConfigError):
        manager.yaml_value("agent.frobnicate")


# -- list add/remove --------------------------------------------------------------


async def test_allowlist_add_remove_in_plain_tokens() -> None:
    manager, store, live = _manager()

    reply = await manager.set(op="add", path="contacts.allowlist", value="telegram @friend")
    assert reply == (
        "contacts.allowlist updated — 1 entries now (config.yaml has 0). "
        "— applies to the next message I see"
    )
    assert live.contacts.allowlist == [ContactRule(platform="telegram", handle="@friend")]
    assert store.rows == {"contacts.allowlist": [{"platform": "telegram", "handle": "@friend"}]}

    assert await manager.set(op="add", path="contacts.allowlist", value="telegram @friend") == (
        "telegram/@friend is already on the allowlist."
    )
    await manager.set(op="add", path="contacts.allowlist", value="discord @friend")

    # a bare handle naming two platforms refuses and says how to disambiguate
    assert await manager.set(op="remove", path="contacts.allowlist", value="@friend") == (
        "@friend is on the allowlist more than once (telegram/@friend, discord/@friend) — "
        "name the platform: remove contacts.allowlist <platform> <handle>."
    )

    await manager.set(op="remove", path="contacts.allowlist", value="telegram @friend")
    assert live.contacts.allowlist == [ContactRule(platform="discord", handle="@friend")]
    await manager.set(op="remove", path="contacts.allowlist", value="discord @friend")
    assert live.contacts.allowlist == []

    assert await manager.set(op="remove", path="contacts.allowlist", value="@friend") == (
        "@friend isn't on the allowlist."
    )


# -- per-app contacts: one list and one switch per app ----------------------------


async def test_per_app_mode_sets_and_resets_like_any_setting() -> None:
    manager, store, live = _manager()

    reply = await manager.set(op="set", path="contacts.gmail.mode", value="enforce")
    assert reply == (
        "contacts.gmail.mode set to enforce (config.yaml says none). "
        "— applies to the next message I see"
    )
    assert live.contacts.gmail.mode == "enforce"
    assert store.rows == {"contacts.gmail.mode": "enforce"}
    # the cross-app default is untouched — the app's switch is its own
    assert live.contacts.mode == "off"

    reply = await manager.set(op="reset", path="contacts.gmail.mode")
    assert reply == (
        "contacts.gmail.mode reset to config.yaml (none). — applies to the next message I see"
    )
    assert live.contacts.gmail.mode is None
    assert store.rows == {}

    reply = await manager.set(op="set", path="contacts.discord.mode", value="off")
    assert reply == (
        "contacts.discord.mode set to off (config.yaml says none). — applies to the next message I see"
    )
    assert live.contacts.discord.mode == "off"

    # a mode the vocabulary doesn't know refuses and persists nothing
    reply = await manager.set(op="set", path="contacts.gmail.mode", value="strict")
    assert "contacts.gmail.mode can't be 'strict' — it must be one of" in reply
    assert store.rows == {"contacts.discord.mode": "off"}


async def test_per_app_allowlist_add_and_remove_in_plain_tokens() -> None:
    manager, store, live = _manager()

    reply = await manager.set(op="add", path="contacts.gmail.allowlist", value="abc123@xyz.com")
    assert reply == (
        "contacts.gmail.allowlist updated — 1 entries now (config.yaml has 0). "
        "— applies to the next message I see"
    )
    assert live.contacts.gmail.allowlist == [ContactRule(handle="abc123@xyz.com")]
    assert store.rows == {
        "contacts.gmail.allowlist": [{"platform": "*", "handle": "abc123@xyz.com"}]
    }
    # the cross-app bucket and the other apps stay untouched
    assert live.contacts.allowlist == []
    assert live.contacts.telegram.allowlist == []

    assert await manager.set(op="add", path="contacts.gmail.allowlist", value="abc123@xyz.com") == (
        "*/abc123@xyz.com is already on the allowlist."
    )
    await manager.set(op="add", path="contacts.gmail.allowlist", value="gmail dhak3212@hdh.com")
    assert live.contacts.gmail.allowlist == [
        ContactRule(handle="abc123@xyz.com"),
        ContactRule(platform="gmail", handle="dhak3212@hdh.com"),
    ]

    await manager.set(op="remove", path="contacts.gmail.allowlist", value="abc123@xyz.com")
    assert live.contacts.gmail.allowlist == [
        ContactRule(platform="gmail", handle="dhak3212@hdh.com")
    ]
    assert (
        await manager.set(op="remove", path="contacts.gmail.allowlist", value="abc123@xyz.com")
        == "abc123@xyz.com isn't on the allowlist."
    )

    reply = await manager.set(op="reset", path="contacts.gmail.allowlist")
    assert reply == (
        "contacts.gmail.allowlist reset to config.yaml (0 entries). "
        "— applies to the next message I see"
    )
    assert live.contacts.gmail.allowlist == []
    assert store.rows == {}


async def test_per_app_paths_refuse_unknown_apps_and_fields_in_plain_words() -> None:
    manager, store, _ = _manager()

    assert await manager.set(op="set", path="contacts.notion.mode", value="enforce") == (
        "contacts paths are contacts.<setting>, or one app's own "
        "contacts.<app>.mode / contacts.<app>.allowlist — e.g. contacts.gmail.allowlist."
    )
    # contacts.allowlist is a real setting, not an app — a nested path under it
    # is the same refusal, not a crash
    assert await manager.set(op="set", path="contacts.allowlist.gmail", value="x") == (
        "contacts paths are contacts.<setting>, or one app's own "
        "contacts.<app>.mode / contacts.<app>.allowlist — e.g. contacts.gmail.allowlist."
    )
    reply = await manager.set(op="set", path="contacts.gmail.subjects", value="x")
    assert "'contacts.gmail.subjects' isn't a setting I know" in reply
    assert store.rows == {}


async def test_show_renders_the_app_slices_in_plain_words() -> None:
    manager, _, _ = _manager()
    await manager.set(op="set", path="contacts.mode", value="enforce")
    await manager.set(op="add", path="contacts.allowlist", value="@sam")
    await manager.set(op="set", path="contacts.gmail.mode", value="off")
    await manager.set(op="add", path="contacts.telegram.allowlist", value="@priya")

    # the overview names each app that has its own list or switch — no raw None,
    # no pydantic dump
    overview = await manager.show()
    assert "own lists: gmail off (0), telegram default (1)" in overview
    assert "Changed from config.yaml: contacts.mode = enforce · contacts.allowlist = 1 entries" in (
        overview
    )
    assert "· contacts.gmail.mode = off" in overview
    assert "· contacts.telegram.allowlist = 1 entries" in overview

    section = await manager.show("contacts")
    assert "  gmail: off (changed from config.yaml)" in section
    assert "  telegram: follows the default (changed from config.yaml)" in section
    assert "    */@priya" in section
    assert "  discord: follows the default\n" in section
    # no raw model ever reaches the words
    assert "AppContacts" not in section and "mode=None" not in section

    assert await manager.show("contacts.gmail.mode") == (
        "contacts.gmail.mode = off (changed from config.yaml, which says none)"
    )


async def test_yaml_value_of_an_app_slice_reads_the_base_app() -> None:
    live = AppConfig(contacts={"gmail": {"allowlist": [ContactRule(handle="abc123@xyz.com")]}})
    manager, _, _ = _manager(live=live)

    await manager.set(op="add", path="contacts.gmail.allowlist", value="dhak3212@hdh.com")
    base = manager.yaml_value("contacts.gmail.allowlist")
    assert [r.handle for r in base] == ["abc123@xyz.com"]  # yaml's entry, still
    await manager.set(op="reset", path="contacts.gmail.allowlist")
    assert [r.handle for r in manager.yaml_value("contacts.gmail.allowlist")] == ["abc123@xyz.com"]


async def test_authz_rules_add_remove_and_regex_refusals() -> None:
    manager, store, live = _manager()

    reply = await manager.set(
        op="add", path="authz.rules", value="mail__send allow for the mail app"
    )
    assert reply == (
        "authz.rules updated — 1 entries now (config.yaml has 0). "
        "— the gate uses the new rules from the next call"
    )
    assert live.authz.rules == [
        AuthzRule(tool_pattern="mail__send", decision="allow", note="for the mail app")
    ]

    # a pattern that can't compile is refused before it can ever persist
    reply = await manager.set(op="add", path="authz.rules", value="([ allow")
    assert reply.startswith("that rule's pattern '([' isn't valid regex: ")

    assert await manager.set(op="add", path="authz.rules", value="mail__send deny") == (
        "a rule for 'mail__send' already exists — remove it first if you mean to replace it."
    )
    assert await manager.set(op="remove", path="authz.rules", value="ghost__x") == (
        "no rule for 'ghost__x' in authz.rules."
    )

    await manager.set(op="remove", path="authz.rules", value="mail__send")
    assert live.authz.rules == []
    assert store.rows == {"authz.rules": []}  # the whole list is what persists


async def test_mcp_servers_add_remove_by_name() -> None:
    manager, _, live = _manager(live=AppConfig(mcp_servers=[MCPServerConfig(name="mail")]))

    assert await manager.set(op="add", path="mcp_servers", value="calendar") == (
        "an mcp_servers entry is a JSON object — e.g. "
        'add mcp_servers {"name": "mail", "transport": {"type": "http", "url": "…"}}.'
    )
    reply = await manager.set(op="add", path="mcp_servers", value={"name": "calendar"})
    assert reply == "mcp_servers updated — 2 entries now (config.yaml has 1)."
    assert [s.name for s in live.mcp_servers] == ["mail", "calendar"]

    assert await manager.set(op="add", path="mcp_servers", value={"name": "calendar"}) == (
        "an app named 'calendar' is already in mcp_servers."
    )
    assert await manager.set(op="remove", path="mcp_servers", value="ghost") == (
        "no app named 'ghost' in mcp_servers."
    )

    await manager.set(op="remove", path="mcp_servers", value="mail")
    assert [s.name for s in live.mcp_servers] == ["calendar"]


async def test_a_removed_app_takes_its_item_row_with_it() -> None:
    live = AppConfig(mcp_servers=[MCPServerConfig(name="mail")])
    manager, store, _ = _manager(rows={"mcp_servers.mail.enabled": False}, live=live)

    await manager.set(op="remove", path="mcp_servers", value="mail")
    assert store.rows == {"mcp_servers": []}
    assert "mcp_servers.mail.enabled" in store.deletes  # no orphan flips a gone app


async def test_reset_of_the_mcp_list_takes_the_item_rows_too() -> None:
    live = AppConfig(mcp_servers=[MCPServerConfig(name="mail")])
    manager, store, live = _manager(live=live)
    await manager.set(op="add", path="mcp_servers", value={"name": "calendar"})
    await manager.set(op="set", path="mcp_servers.mail.enabled", value=False)
    assert set(store.rows) == {"mcp_servers", "mcp_servers.mail.enabled"}

    reply = await manager.set(op="reset", path="mcp_servers")
    assert reply == "mcp_servers reset to config.yaml (1 entries)."
    assert store.rows == {}
    assert [s.name for s in live.mcp_servers] == ["mail"]
    assert live.mcp_servers[0].enabled is True  # the yaml's mail, back


async def test_reset_of_a_chat_added_apps_item_says_the_list_stands() -> None:
    live = AppConfig(mcp_servers=[MCPServerConfig(name="mail")])
    manager, store, live = _manager(live=live)
    await manager.set(op="add", path="mcp_servers", value={"name": "calendar"})
    await manager.set(op="set", path="mcp_servers.calendar.enabled", value=False)

    reply = await manager.set(op="reset", path="mcp_servers.calendar.enabled")
    assert reply == (
        "mcp_servers.calendar.enabled reset — config.yaml doesn't name an app "
        "'calendar', so the setting the list carries stands."
    )
    assert "mcp_servers.calendar.enabled" not in store.rows
    assert live.mcp_servers[1].enabled is False  # the list still carries it


# -- live-apply hooks ---------------------------------------------------------------


async def test_quiet_hours_change_reparses_the_window_live() -> None:
    manager, _, _ = _manager()
    loop = FakeLoop()
    manager.wire(loop=loop)

    reply = await manager.set(op="set", path="agent.quiet_hours", value="23:00-07:00")
    assert "the quiet window is live right away" in reply
    assert loop.reparsed == 1

    # other agent settings have nothing to reparse
    await manager.set(op="set", path="agent.tick_seconds", value=10)
    assert loop.reparsed == 1


async def test_authz_change_swaps_the_gate_and_the_approval_window() -> None:
    manager, _, _ = _manager()
    loop = FakeLoop()
    approvals = FakeApprovals()
    manager.wire(loop=loop, approvals=approvals)

    reply = await manager.set(
        op="set",
        path="authz.rules",
        value=[{"tool_pattern": "mail__send", "decision": "allow"}],
    )
    assert reply == (
        "authz.rules set — 1 entries now (config.yaml has 0). "
        "— the gate uses the new rules from the next call"
    )
    # the gate the loop routes through is the rebuilt one
    assert isinstance(loop._policy, Policy)
    assert loop._policy.classify("mail__send", {}).decision == "allow"

    reply = await manager.set(op="set", path="authz.approval_ttl_hours", value=48)
    assert reply == (
        "authz.approval_ttl_hours set to 48 (config.yaml says 24). "
        "— new approvals now wait 48h before expiring"
    )
    assert approvals.ttl_updates[-1] == 48.0


async def test_messaging_change_diffs_the_connectors_live(monkeypatch) -> None:
    manager, _, live = _manager()
    telegram = FakeSurfaceConnector("telegram")
    surfaces = SurfaceFanout([telegram])
    loop = FakeLoop()
    manager.wire(loop=loop, surfaces=surfaces)

    discord = FakeSurfaceConnector("discord")
    seen: list[dict] = []

    def fake_factory(settings, config, *, approvals=None, on_decision=None, on_inbound=None):
        seen.append(
            {
                "config_is_live": config is live,
                "has_callbacks": on_decision is not None and on_inbound is not None,
            }
        )
        return [discord]

    monkeypatch.setattr(aether.config_store, "build_messaging_connectors", fake_factory)

    reply = await manager.set(op="set", path="messaging.discord.enabled", value=True)
    assert reply == (
        "messaging.discord.enabled set to true (config.yaml says false). "
        "— telegram is stopping; discord is starting up"
    )
    # the fresh build sees the merged config and the loop's callbacks
    assert seen == [{"config_is_live": True, "has_callbacks": True}]
    assert telegram.stops == 1
    assert discord.starts == 1
    assert [c.name for c in surfaces.connectors] == ["discord"]
    assert live.messaging.discord.enabled is True


async def test_messaging_enabled_without_a_token_is_said_honestly() -> None:
    # the real factory with tokenless settings: the toggle persists, the
    # missing connector is named instead of hand-waved
    manager, _, live = _manager()
    manager.wire(loop=FakeLoop(), surfaces=SurfaceFanout([]))

    reply = await manager.set(op="set", path="messaging.telegram.enabled", value=True)
    assert reply == (
        "messaging.telegram.enabled set to true (config.yaml says false). "
        "— telegram has no token yet — it starts once one is pasted or set"
    )
    assert live.messaging.telegram.enabled is True


async def test_unwired_hooks_log_and_the_row_persists_anyway() -> None:
    manager, store, _ = _manager()  # nothing wired

    reply = await manager.set(op="set", path="messaging.telegram.enabled", value=True)
    assert reply == "messaging.telegram.enabled set to true (config.yaml says false)."
    assert store.rows == {"messaging.telegram.enabled": True}

    # quiet hours still say when they bite — the reparse just happens at boot
    reply = await manager.set(op="set", path="agent.quiet_hours", value="23:00-07:00")
    assert reply.endswith("— the quiet window is live right away")


async def test_mcp_removal_drops_the_apps_actions_from_the_namespace() -> None:
    live = AppConfig(mcp_servers=[MCPServerConfig(name="mail")])
    manager, _, _ = _manager(live=live)
    host = FakeHost(diff={"started": [], "stopped": ["mail"]})
    registry = await _registry_with(
        host, [ToolSpec(name="mail__act", description="mail action", source="mail")]
    )
    loop = FakeLoop()
    manager.wire(loop=loop, tools=registry, host=host)

    reply = await manager.set(op="remove", path="mcp_servers", value="mail")
    assert reply == (
        "mcp_servers updated — 0 entries now (config.yaml has 1). "
        "— mail stopped — their actions are gone until they're on again"
    )
    assert live.mcp_servers == []
    assert host.applied == [[]]
    assert registry.get("mail__act") is None  # dropped, not just unreachable
    assert registry.get("memory_search") is not None  # the natives survive
    assert loop.rebuilt == 1


async def test_mcp_addition_starts_the_app_and_publishes_its_actions() -> None:
    live = AppConfig(mcp_servers=[MCPServerConfig(name="mail")])
    manager, _, _ = _manager(live=live)
    host = FakeHost(diff={"started": [SimpleNamespace(name="calendar", ready=True)], "stopped": []})
    registry = await _registry_with(
        host, [ToolSpec(name="mail__act", description="mail action", source="mail")]
    )
    loop = FakeLoop()
    manager.wire(loop=loop, tools=registry, host=host)

    reply = await manager.set(op="add", path="mcp_servers", value={"name": "calendar"})
    assert reply == (
        "mcp_servers updated — 2 entries now (config.yaml has 1). "
        "— calendar connected · 1 action in my vocabulary"
    )
    assert host.applied == [["mail", "calendar"]]
    assert registry.get("calendar__act") is not None
    assert registry.get("mail__act") is not None  # the untouched app keeps its actions
    assert loop.rebuilt == 1


async def test_mcp_addition_of_a_silent_server_makes_the_promise() -> None:
    """A server that hasn't answered yet is never claimed as connected —
    the note says what's true and promises the announce."""
    live = AppConfig(mcp_servers=[MCPServerConfig(name="mail")])
    manager, _, _ = _manager(live=live)
    host = FakeHost(
        diff={"started": [SimpleNamespace(name="calendar", ready=False)], "stopped": []}
    )
    registry = await _registry_with(
        host, [ToolSpec(name="mail__act", description="mail action", source="mail")]
    )
    loop = FakeLoop()
    manager.wire(loop=loop, tools=registry, host=host)

    reply = await manager.set(op="add", path="mcp_servers", value={"name": "calendar"})
    assert reply == (
        "mcp_servers updated — 2 entries now (config.yaml has 1). "
        "— calendar hasn't answered yet — still trying, I'll tell you the moment it's up"
    )


async def test_mcp_restart_gets_the_connected_note_not_a_stopped_one() -> None:
    """A changed app restarts — its name is in both halves of the diff. The
    note speaks only of the fresh connection; 'stopped' is for apps that
    actually left, and 'their actions are gone' would be a lie here."""
    live = AppConfig(mcp_servers=[MCPServerConfig(name="mail")])
    manager, _, _ = _manager(live=live)
    host = FakeHost(
        diff={"started": [SimpleNamespace(name="mail", ready=True)], "stopped": ["mail"]}
    )
    registry = await _registry_with(
        host, [ToolSpec(name="mail__act", description="mail action", source="mail")]
    )
    loop = FakeLoop()
    manager.wire(loop=loop, tools=registry, host=host)

    changed = [
        {
            "name": "mail",
            "transport": {"type": "http", "url": "https://mcp.example.com/mcp"},
        }
    ]
    reply = await manager.set(op="set", path="mcp_servers", value=changed)
    assert reply == (
        "mcp_servers set — 1 entries now (config.yaml has 1). "
        "— mail connected · 1 action in my vocabulary"
    )
    assert loop.rebuilt == 1


async def test_a_same_value_messaging_write_still_reconciles(tmp_path, monkeypatch) -> None:
    """'It starts once the token is there,' made true: every messaging
    write rebuilds the connectors — even a same-value one — and the build
    reads tokens fresh through the resolver, so a key added after the
    toggle was set connects with no restart."""
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=fresh-token\n")
    manager, _, live = _manager()
    manager.wire(loop=FakeLoop(), surfaces=SurfaceFanout([]), resolver=EnvResolver(env_file))

    builds: list[Settings] = []

    def fake_factory(settings, config, *, approvals=None, on_decision=None, on_inbound=None):
        builds.append(settings)
        return []

    monkeypatch.setattr(aether.config_store, "build_messaging_connectors", fake_factory)

    await manager.set(op="set", path="messaging.telegram.enabled", value=True)
    assert builds[-1].telegram_bot_token == "fresh-token"  # fresh, not boot-time

    await manager.set(op="set", path="messaging.telegram.enabled", value=True)  # same value
    assert len(builds) == 2  # rebuilt anyway — that's the whole promise


# -- the render story, in miniature -------------------------------------------------


async def test_a_set_survives_a_fresh_boot() -> None:
    # ephemeral disk: a fresh process rebuilds its live config from pydantic
    # defaults, and the rows turn it back into what chat made
    manager, store, _ = _manager()
    await manager.set(op="set", path="agent.tick_seconds", value=10)
    await manager.set(op="add", path="contacts.allowlist", value="telegram @friend")

    fresh = AppConfig()
    manager2, _, _ = _manager(rows=dict(store.rows), live=fresh)
    assert await manager2.bootstrap() == 2
    assert fresh.agent.tick_seconds == 10.0
    assert fresh.contacts.allowlist == [ContactRule(platform="telegram", handle="@friend")]
