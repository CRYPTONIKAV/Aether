"""The guided /config walk — the state machine alone, with a fake applier.

The wizard owns no I/O: these tests pin the conversation (every menu, every
question, every nudge) and that each answer ends as the right (op, path,
value) handed to the loop's shared applier — the same one the one-line
grammar uses, so the gate's half is tested where it lives, in
test_agent_loop.
"""

from __future__ import annotations

import time

from aether.agent.config_wizard import ConfigWizard
from aether.config import AppConfig, AuthzRule, ContactRule, MCPServerConfig

_TOP_MENU = (
    'let\'s set me up — answer each question with a number, or "stop" any time.\n'
    "1 — see my settings\n"
    "2 — change something\n"
    "3 — put something back the way config.yaml had it"
)
_AGAIN = (
    'anything else? 1 — see my settings  2 — change something  3 — reset something  (or "done")'
)
_GOODBYE = "ok — anytime. /config to start again."

_UNSET = object()


class FakeApplier:
    """The loop's _apply_config double: records the handoff, answers canned."""

    def __init__(self, reply: str = "applied") -> None:
        self.calls: list[tuple[str, str, object]] = []
        self.reply = reply

    async def __call__(self, op: str, path: str, value: object = _UNSET) -> str:
        self.calls.append((op, path, None if value is _UNSET else value))
        return self.reply


class FakeShow:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    async def __call__(self, path: str | None) -> str:
        self.calls.append(path)
        return "the overview"


def _wizard(
    config: AppConfig | None = None,
    applier: FakeApplier | None = None,
    yaml_values: dict[str, object] | None = None,
) -> tuple[ConfigWizard, FakeApplier, FakeShow]:
    config = config or AppConfig()
    applier = applier or FakeApplier()
    show = FakeShow()
    wizard = ConfigWizard(
        config,
        apply=applier,
        show=show,
        yaml_value=(lambda path: yaml_values[path]) if yaml_values else None,
    )
    return wizard, applier, show


async def _say(wizard: ConfigWizard, *texts: str) -> list[str]:
    return [reply for text in texts if (reply := await wizard.handle(text)) is not None]


# -- the opening and the top menu -------------------------------------------------


def test_start_opens_with_the_top_menu() -> None:
    wizard, applier, _ = _wizard()
    assert wizard.start() == _TOP_MENU
    assert wizard.alive
    assert applier.calls == []


async def test_see_relays_the_overview_then_offers_the_menu_again() -> None:
    wizard, _, show = _wizard()
    wizard.start()
    (reply,) = await _say(wizard, "1")
    assert reply == f"the overview\n\n{_AGAIN}"
    assert show.calls == [None]  # the overview, not one path
    assert wizard.alive  # the walk continues


async def test_a_full_change_walk_applies_the_answered_value() -> None:
    wizard, applier, _ = _wizard()
    menu = wizard.start()
    area, setting, question, done = await _say(wizard, "2", "2", "1", "10")
    assert menu == _TOP_MENU
    assert area == (
        "which area?\n"
        "1 — llm — the model I think with\n"
        "2 — agent — my rhythm: how often I check, how much I do\n"
        "3 — salience — what's worth your attention\n"
        "4 — messaging — which chat apps I speak on\n"
        "5 — contacts — who I'm allowed to see\n"
        "6 — authz — my permission gate\n"
        "7 — mcp_servers — the apps I'm connected to"
    )
    assert setting == (
        "agent today:\n"
        "1 — tick_seconds · 30 — seconds between my checks\n"
        "2 — max_tool_iterations · 12 — tool calls I may chain in one turn\n"
        "3 — daily_surface_cap · 20 — most things I bring up unprompted per day\n"
        "4 — quiet_hours · none — hours I hold non-urgent notes\n"
        "5 — quiet_urgent_salience · 8 — how urgent something must be to interrupt quiet\n"
        "which one?"
    )
    assert question == "how many seconds between checks? (currently 30)"
    assert done == f"applied\n{_AGAIN}"
    assert applier.calls == [("set", "agent.tick_seconds", 10)]


async def test_menu_answers_are_coerced_like_the_one_line_grammar() -> None:
    wizard, applier, _ = _wizard()
    await _say(wizard, "2", "4", "1", "on")  # change → messaging → telegram
    assert applier.calls == [("set", "messaging.telegram.enabled", True)]

    wizard, applier, _ = _wizard()
    await _say(wizard, "2", "2", "4", "none")  # change → agent → quiet_hours
    assert applier.calls == [("set", "agent.quiet_hours", None)]

    wizard, applier, _ = _wizard()
    await _say(wizard, "2", "3", "1", "1.5")  # change → salience → threshold
    assert applier.calls == [("set", "salience.threshold", 1.5)]


async def test_llm_questions_warn_about_the_restart() -> None:
    wizard, _, _ = _wizard()
    _, setting, question = await _say(wizard, "2", "1", "2")
    assert setting.startswith("llm today:")
    assert "2 — model · claude-opus-5 — the model I think with" in setting
    assert question == (
        "which model? (currently claude-opus-5 — heads-up: llm changes "
        "restart me, back in a few seconds)"
    )


async def test_provider_is_a_numbered_choice() -> None:
    wizard, applier, _ = _wizard()
    _, _, question = await _say(wizard, "2", "1", "1")
    assert question == (
        "who should make me think? (currently anthropic — heads-up: "
        "llm changes restart me, back in a few seconds)\n"
        "1 — anthropic\n2 — openai\n3 — gemini\n4 — ollama\n5 — nvidia_nim"
    )
    await _say(wizard, "3")
    assert applier.calls == [("set", "llm.provider", "gemini")]


async def test_a_security_shaped_answer_relays_the_park_card_verbatim() -> None:
    park = (
        "that one's security-shaped — held for your one-tap approval (#14). "
        "Tap approve and it's done."
    )
    wizard, applier, _ = _wizard(applier=FakeApplier(reply=park))
    _, _, _, reply = await _say(wizard, "2", "4", "1", "on")
    assert reply == f"{park}\n{_AGAIN}"  # the gate's words, then the menu
    assert applier.calls == [("set", "messaging.telegram.enabled", True)]


async def test_contacts_mode_is_a_numbered_choice() -> None:
    wizard, applier, _ = _wizard()
    _, _, question = await _say(wizard, "2", "5", "1")
    assert question == (
        "who may I see?\n"
        "1 — off — everyone\n"
        "2 — enforce — only the people on the allowlist\n"
        "(currently off)"
    )
    await _say(wizard, "2")
    assert applier.calls == [("set", "contacts.mode", "enforce")]


async def test_a_value_stage_takes_the_whole_message() -> None:
    wizard, applier, _ = _wizard()
    await _say(wizard, "2", "2", "1")  # sitting at the tick_seconds question
    (done,) = await _say(wizard, "thirty seconds please")
    # never cancelled for being long — it's the value; the gate refuses junk
    assert applier.calls == [("set", "agent.tick_seconds", "thirty seconds please")]
    assert done == f"applied\n{_AGAIN}"


# -- reset, with the value on the table ---------------------------------------------


async def test_reset_confirms_with_the_yaml_value_then_applies() -> None:
    wizard, applier, _ = _wizard(yaml_values={"agent.tick_seconds": 30.0})
    _, _, confirm, done = await _say(wizard, "3", "2", "1", "1")
    assert confirm == "put tick_seconds back to config.yaml's 30?\n1 — yes  2 — no"
    assert done == f"applied\n{_AGAIN}"
    assert applier.calls == [("reset", "agent.tick_seconds", None)]


async def test_reset_declined_by_number_changes_nothing() -> None:
    wizard, applier, _ = _wizard(yaml_values={"agent.tick_seconds": 30.0})
    _, _, _, reply = await _say(wizard, "3", "2", "1", "2")
    assert reply == _AGAIN
    assert applier.calls == []


async def test_reset_declined_by_word_changes_nothing() -> None:
    wizard, applier, _ = _wizard(yaml_values={"agent.tick_seconds": 30.0})
    _, _, _, reply = await _say(wizard, "3", "2", "1", "no")
    assert reply == _AGAIN
    assert applier.calls == []


async def test_reset_without_a_yaml_answer_keeps_the_question_vague() -> None:
    wizard, _, _ = _wizard()  # no yaml_value wired at all
    _, _, confirm = await _say(wizard, "3", "2", "1")
    assert confirm == ("put tick_seconds back the way config.yaml had it?\n1 — yes  2 — no")


async def test_list_reset_counts_the_yaml_entries() -> None:
    wizard, applier, _ = _wizard(yaml_values={"contacts.allowlist": [ContactRule(handle="@mom")]})
    _, _, confirm, done = await _say(wizard, "3", "5", "2", "1")
    assert confirm == ("put the allowlist back to config.yaml's 1 entry?\n1 — yes  2 — no")
    assert applier.calls == [("reset", "contacts.allowlist", None)]
    assert done == f"applied\n{_AGAIN}"


async def test_rules_reset_counts_the_yaml_entries() -> None:
    wizard, applier, _ = _wizard(
        yaml_values={"authz.rules": [AuthzRule(tool_pattern="telegram__send", decision="allow")]}
    )
    _, _, confirm, _ = await _say(wizard, "3", "6", "1", "1")
    assert confirm == "put the rules back to config.yaml's 1 entry?\n1 — yes  2 — no"
    assert applier.calls == [("reset", "authz.rules", None)]


# -- the allowlist flow --------------------------------------------------------------


async def test_allowlist_add_is_platform_then_handle() -> None:
    wizard, applier, _ = _wizard()
    _, pick, listing, platform, handle, done = await _say(
        wizard, "2", "5", "2", "1", "telegram", "@mom"
    )
    assert pick.startswith("contacts today:")
    assert "2 — allowlist · 0 entries — who I'm allowed to see" in pick
    assert listing == (
        "the allowlist today: (nothing on it yet)\n1 — add someone  2 — remove someone  3 — back"
    )
    assert platform == "which platform? (telegram, discord, slack — or * for any)"
    assert handle == "what's their handle?"
    assert applier.calls == [("add", "contacts.allowlist", "telegram @mom")]
    assert done == f"applied\n{_AGAIN}"


async def test_allowlist_remove_picks_by_number() -> None:
    config = AppConfig(contacts={"allowlist": [ContactRule(platform="telegram", handle="@mom")]})
    wizard, applier, _ = _wizard(config=config)
    _, _, _, numbered, done = await _say(wizard, "2", "5", "2", "2", "1")
    assert numbered == "1 — telegram @mom\nreply with a number, or stop"
    assert applier.calls == [("remove", "contacts.allowlist", "telegram @mom")]
    assert done == f"applied\n{_AGAIN}"


async def test_allowlist_remove_of_a_star_platform_uses_the_handle_alone() -> None:
    config = AppConfig(contacts={"allowlist": [ContactRule(handle="friend@example.com")]})
    wizard, applier, _ = _wizard(config=config)
    await _say(wizard, "2", "5", "2", "2", "1")
    assert applier.calls == [("remove", "contacts.allowlist", "friend@example.com")]


async def test_allowlist_remove_with_nothing_on_it_is_honest() -> None:
    wizard, applier, _ = _wizard()
    _, _, _, reply = await _say(wizard, "2", "5", "2", "2")
    assert reply == (
        "there's nothing on it to remove.\n"
        "the allowlist today: (nothing on it yet)\n"
        "1 — add someone  2 — remove someone  3 — back"
    )
    assert applier.calls == []


async def test_list_back_returns_to_the_menu() -> None:
    wizard, applier, _ = _wizard()
    _, _, _, reply = await _say(wizard, "2", "5", "2", "3")
    assert reply == _AGAIN
    assert applier.calls == []


# -- the authz rules flow -------------------------------------------------------------


async def test_authz_add_is_pattern_decision_note() -> None:
    wizard, applier, _ = _wizard()
    _, _, listing, pattern, decision, note, done = await _say(
        wizard, "2", "6", "1", "1", "telegram__send", "2", "telegram sends are one-tap"
    )
    assert listing == ("the rules today: (none yet)\n1 — add a rule  2 — remove a rule  3 — back")
    assert pattern == (
        "which tool calls does it cover? (a pattern, like telegram__send or .*__delete.*)"
    )
    assert decision == (
        "should those be allowed without asking, held for your tap, or denied?\n"
        "1 — allow  2 — approve (one tap)  3 — deny"
    )
    assert note == "want to add a note for the record? (a few words, or none)"
    assert applier.calls == [
        ("add", "authz.rules", "telegram__send approve telegram sends are one-tap")
    ]
    assert done == f"applied\n{_AGAIN}"


async def test_authz_note_none_means_no_note() -> None:
    wizard, applier, _ = _wizard()
    await _say(wizard, "2", "6", "1", "1", "telegram__send", "approve", "none")
    assert applier.calls == [("add", "authz.rules", "telegram__send approve")]


async def test_authz_pattern_with_spaces_is_asked_again() -> None:
    wizard, applier, _ = _wizard()
    _, _, _, _, again = await _say(wizard, "2", "6", "1", "1", "foo bar")
    assert again == "just the pattern, no spaces — like telegram__send or .*__delete.*"
    assert applier.calls == []


async def test_authz_remove_picks_by_number() -> None:
    config = AppConfig(
        authz={"rules": [AuthzRule(tool_pattern="telegram__send", decision="allow")]}
    )
    wizard, applier, _ = _wizard(config=config)
    _, _, _, numbered, done = await _say(wizard, "2", "6", "1", "2", "1")
    assert numbered == "1 — telegram__send\nreply with a number, or stop"
    assert applier.calls == [("remove", "authz.rules", "telegram__send")]
    assert done == f"applied\n{_AGAIN}"


# -- the mcp_servers flow ---------------------------------------------------------------


async def test_mcp_apps_are_menu_items_with_their_state() -> None:
    config = AppConfig(mcp_servers=[MCPServerConfig(name="mail")])
    wizard, _, _ = _wizard(config=config)
    _, menu = await _say(wizard, "2", "7")
    assert menu == (
        "the apps today:\n1 — mail (on, 0 polls)\n2 — add an app  3 — remove an app  4 — back"
    )


async def test_mcp_app_toggle_sets_the_item_path() -> None:
    config = AppConfig(mcp_servers=[MCPServerConfig(name="mail")])
    wizard, applier, _ = _wizard(config=config)
    _, _, question, done = await _say(wizard, "2", "7", "1", "off")
    assert question == "turn mail on or off? (currently on)"
    assert applier.calls == [("set", "mcp_servers.mail.enabled", False)]
    assert done == f"applied\n{_AGAIN}"


async def test_mcp_add_stdio_builds_the_object() -> None:
    wizard, applier, _ = _wizard()
    _, _, name, kind, command, done = await _say(
        wizard, "2", "7", "1", "mail", "1", "npx -y mcp-mail-server"
    )
    assert name == "what's the app's name?"
    assert kind == (
        "how does mail connect?\n1 — stdio — a command run locally\n2 — http — a URL I call"
    )
    assert command == "what's the command? (e.g. npx -y mcp-mail-server)"
    assert applier.calls == [
        (
            "add",
            "mcp_servers",
            {
                "name": "mail",
                "transport": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "mcp-mail-server"],
                },
            },
        )
    ]
    assert "/config add mcp_servers {json}" in done  # the pointer to the one-liner


async def test_mcp_add_http_takes_the_url() -> None:
    wizard, applier, _ = _wizard()
    _, _, _, _, url, done = await _say(
        wizard, "2", "7", "1", "calendar", "http", "https://mcp.example.com/mcp"
    )
    assert url == "what's the URL?"
    assert applier.calls == [
        (
            "add",
            "mcp_servers",
            {
                "name": "calendar",
                "transport": {"type": "http", "url": "https://mcp.example.com/mcp"},
            },
        )
    ]
    assert done.endswith(_AGAIN)


async def test_mcp_remove_picks_the_app_by_number() -> None:
    config = AppConfig(mcp_servers=[MCPServerConfig(name="mail"), MCPServerConfig(name="calendar")])
    wizard, applier, _ = _wizard(config=config)
    _, _, numbered, done = await _say(wizard, "2", "7", "4", "2")  # 4 = remove an app
    assert numbered == "1 — mail\n2 — calendar\nreply with a number, or stop"
    assert applier.calls == [("remove", "mcp_servers", "calendar")]
    assert done == f"applied\n{_AGAIN}"


async def test_mcp_reset_mode_resets_an_app_without_add_or_remove() -> None:
    config = AppConfig(mcp_servers=[MCPServerConfig(name="mail")])
    wizard, applier, _ = _wizard(config=config)
    _, menu, confirm, done = await _say(wizard, "3", "7", "1", "yes")
    assert menu == "the apps today:\n1 — mail (on, 0 polls)\n2 — back"
    assert confirm == "put mail back the way config.yaml had it?\n1 — yes  2 — no"
    assert applier.calls == [("reset", "mcp_servers.mail.enabled", None)]
    assert done == f"applied\n{_AGAIN}"


# -- stepping aside: nudges, changed subjects, goodbyes, stale walks -------------------


async def test_a_wrong_number_nudges_and_keeps_the_walk() -> None:
    wizard, applier, _ = _wizard()
    wizard.start()
    nudge, area = await _say(wizard, "4", "2")
    assert nudge == "I didn't get that — reply with 1-3, or stop"
    assert area.startswith("which area?")  # the walk was never lost
    assert applier.calls == []


async def test_a_sentence_at_a_menu_ends_the_walk_quietly() -> None:
    wizard, applier, _ = _wizard()
    wizard.start()
    assert await wizard.handle("hey what's up?") is None
    assert not wizard.alive
    assert applier.calls == []


async def test_stop_ends_the_walk_from_the_opening_menu() -> None:
    wizard, applier, _ = _wizard()
    wizard.start()
    assert await wizard.handle("stop") == _GOODBYE
    assert not wizard.alive
    assert applier.calls == []


async def test_done_ends_the_walk_mid_question() -> None:
    wizard, applier, _ = _wizard()
    await _say(wizard, "2", "2", "1")  # sitting at the tick_seconds question
    assert await wizard.handle("done") == _GOODBYE
    assert not wizard.alive
    assert applier.calls == []


async def test_a_stale_walk_gets_out_of_the_way() -> None:
    wizard, applier, _ = _wizard()
    wizard.start()
    stamp = time.monotonic()
    assert await wizard.handle("2", now=stamp) is not None
    assert await wizard.handle("3", now=stamp + 901) is None  # 15 minutes on
    assert not wizard.alive
    assert applier.calls == []


async def test_start_resets_a_walk_in_progress() -> None:
    wizard, applier, _ = _wizard()
    await _say(wizard, "2", "2", "1")  # deep in the walk
    assert wizard.start() == _TOP_MENU
    area = await wizard.handle("2")
    assert area is not None and area.startswith("which area?")  # back at the top


async def test_the_mcp_alias_answers_the_area_question() -> None:
    wizard, _, _ = _wizard()
    wizard.start()
    await _say(wizard, "2")
    reply = await wizard.handle("mcp")
    assert reply == ("the apps today: (none yet)\n1 — add an app  2 — remove an app  3 — back")
