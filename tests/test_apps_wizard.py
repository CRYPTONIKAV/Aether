"""The guided /apps walk — the state machine alone, with fake callbacks.

These pin the transcripts from the design (they are the spec): bare
/apps is the status and the add hint, /apps add <name> resolves a chat
surface's recipe first, then the app hub's live catalog (exact, fuzzy
with a numbered pick, or nothing → the generic walk offered); the hub's
connect is a one-time COMPOSIO_API_KEY paste and then a hosted link,
waited out of band; and the walk's last question is the gate's — what
the app may do — with the stricter answers writing an authz rule
through the same applier, origin "walk" and all. No recorded reply ever
carries a value: the save double takes it, and every transcript asserts
names only. The gate's half — that "walk" really does allow through the
real policy — lives in test_agent_loop.
"""

from __future__ import annotations

from aether.agent.apps_wizard import AppsWizard

_ADD_HINT = "add one with /apps add <name> — like /apps add gmail"
_GOODBYE = "ok — anytime. /apps to start again."
_LINK_WAITING = "still waiting — open the link I sent and approve it there; say stop to give up."
_TELEGRAM_RECIPE = (
    "telegram — talk to me there.\n"
    "one token needed:\n"
    "1 — open https://t.me/BotFather and send it /newbot — it hands you a token\n"
    "paste the token here — I'll take it from there."
)
_SLACK_RECIPE = (
    "slack — talk to me there.\n"
    "two tokens needed:\n"
    "1 — open https://api.slack.com/apps and create an app — under OAuth & "
    "Permissions, Install to Workspace copies the bot token (xoxb…)\n"
    "2 — under Basic Info, App-Level Tokens generates the app token "
    "(xapp…) with the connections:write scope\n"
    "paste the bot token (xoxb…) here — I'll take it from there."
)
_HUB_KEY_RECIPE = (
    "the app hub — one key connects every app in the catalog.\n"
    "one key needed (one time — it connects every app):\n"
    "1 — open https://platform.composio.dev and sign in — under "
    "Settings → API Keys, copy your project key\n"
    "paste the key here — I'll take it from there."
)
# the loop's executor reply, relayed verbatim — the honest verdict is the
# applier's own words, the walk only adds its head line
_APPLY_TELEGRAM = (
    "messaging.telegram.enabled set to true (config.yaml says false). — telegram is starting up"
)
_APPLY_SLACK = (
    "messaging.slack.enabled set to true (config.yaml says false). — slack is starting up"
)
_APPLY_WEATHER = (
    "mcp_servers updated — 1 entries now (config.yaml has 0). — weather "
    "connected · 5 actions in my vocabulary"
)
# what a user rule forcing a hold still answers — relayed verbatim too
_PARK_CARD = (
    "that one's security-shaped — held for your one-tap approval (#21). Tap approve and it's done."
)

_UNSET = object()


class FakeApplier:
    """The loop's _apply_config double: records the handoff (origin and
    all), answers canned — the executor's own reply, whatever it is."""

    def __init__(self, reply: str = _APPLY_TELEGRAM) -> None:
        self.calls: list[tuple[str, str, object, str | None]] = []
        self.reply = reply

    async def __call__(
        self, op: str, path: str, value: object = _UNSET, origin: str | None = None
    ) -> str:
        self.calls.append((op, path, None if value is _UNSET else value, origin))
        return self.reply


class FakeSave:
    """save_secret(name, value, app) — records the handoff, answers canned.
    A stored name joins the env set, like the real store feeding the
    resolver; the value is kept here so tests can pin stripping — no
    reply ever carries it back."""

    def __init__(self, ok: bool = True, env: set[str] | None = None) -> None:
        self.ok = ok
        self.env = env
        self.calls: list[tuple[str, str, str]] = []  # (name, value, app)

    async def __call__(self, name: str, value: str, app: str) -> bool:
        self.calls.append((name, value, app))
        if self.ok and self.env is not None:
            self.env.add(name)
        return self.ok


class FakeToolkits:
    """find_toolkits(query) — the hub catalog double: records the ask,
    answers a scripted list of (slug, name) matches."""

    def __init__(self, matches: list[tuple[str, str]] | None = None) -> None:
        self.matches = list(matches) if matches else []
        self.calls: list[str] = []

    async def __call__(self, query: str) -> list[tuple[str, str]]:
        self.calls.append(query)
        return list(self.matches)


class FakeAuthorize:
    """authorize(toolkit) — the hub's connect-link double: records the
    toolkit, answers (request id, link), or None when the hub said no."""

    def __init__(self, url: str | None = "https://hub.example.test/connect") -> None:
        self.url = url
        self.calls: list[str] = []

    async def __call__(self, toolkit: str) -> tuple[str, str] | None:
        self.calls.append(toolkit)
        if self.url is None:
            return None
        return f"req-{toolkit}", f"{self.url}/{toolkit}"


def _wizard(
    *,
    status: str = "your apps: nothing connected yet.",
    env: set[str] | None = None,
    save: FakeSave | None = None,
    applier: FakeApplier | None = None,
    toolkits: FakeToolkits | None = None,
    authorize: FakeAuthorize | None = None,
) -> tuple[AppsWizard, FakeApplier, FakeSave, set[str], FakeToolkits, FakeAuthorize]:
    env = set() if env is None else env
    applier = applier or FakeApplier()
    save = save or FakeSave(env=env)
    toolkits = toolkits or FakeToolkits([("gmail", "Gmail")])
    authorize = authorize or FakeAuthorize()

    async def env_names() -> set[str]:
        return set(env)

    async def status_fn() -> str:
        return status

    wizard = AppsWizard(
        apply=applier,
        status=status_fn,
        env_names=env_names,
        save_secret=save,
        find_toolkits=toolkits,
        authorize=authorize,
    )
    return wizard, applier, save, env, toolkits, authorize


async def _say(wizard: AppsWizard, *texts: str) -> list[str]:
    return [reply for text in texts if (reply := await wizard.handle(text)) is not None]


# -- the opening: status, and the name ask ----------------------------------------


async def test_bare_apps_is_the_status_and_the_hint() -> None:
    wizard, applier, *_ = _wizard()
    assert await wizard.start() == f"your apps: nothing connected yet.\n{_ADD_HINT}"
    assert not wizard.alive  # no question is outstanding — nothing intercepts
    assert applier.calls == []


async def test_apps_add_without_a_name_asks_for_one() -> None:
    wizard, *_ = _wizard()
    assert await wizard.start_add("") == "which app? say the name — like gmail"
    assert wizard.alive
    (recipe,) = await _say(wizard, "telegram")
    assert recipe == _TELEGRAM_RECIPE


async def test_a_sentence_at_the_name_ask_steps_aside() -> None:
    wizard, *_ = _wizard()
    await wizard.start_add("")
    assert await _say(wizard, "hey can you just add github for me please") == []
    assert not wizard.alive  # nothing is waiting; the message is normal chat


# -- a chat surface: the recipe, the paste, the toggle ------------------------------


async def test_the_telegram_walk_pastes_and_switches_on() -> None:
    wizard, applier, save, *_ = _wizard()
    assert await wizard.start_add("telegram") == _TELEGRAM_RECIPE
    (applied,) = await _say(wizard, "123:AAbbCC")
    assert applied == f"stored — switching telegram on now.\n{_APPLY_TELEGRAM}"
    assert save.calls == [("TELEGRAM_BOT_TOKEN", "123:AAbbCC", "telegram")]
    # the pick-and-paste was the approval — the connect rides origin "walk"
    assert applier.calls == [("set", "messaging.telegram.enabled", True, "walk")]
    assert not wizard.alive


async def test_slack_asks_for_its_two_tokens_one_at_a_time() -> None:
    wizard, applier, save, *_ = _wizard(applier=FakeApplier(reply=_APPLY_SLACK))
    assert await wizard.start_add("slack") == _SLACK_RECIPE
    (next_ask,) = await _say(wizard, "xoxb-111")
    assert next_ask == "stored. now the app token (xapp…) — paste it here."
    (applied,) = await _say(wizard, "xapp-222")
    assert applied == f"stored — switching slack on now.\n{_APPLY_SLACK}"
    assert save.calls == [
        ("SLACK_BOT_TOKEN", "xoxb-111", "slack"),
        ("SLACK_APP_TOKEN", "xapp-222", "slack"),
    ]
    assert applier.calls == [("set", "messaging.slack.enabled", True, "walk")]
    assert not wizard.alive


async def test_done_before_the_key_checks_the_env_honestly() -> None:
    wizard, applier, save, env, *_ = _wizard()
    await wizard.start_add("telegram")
    (not_yet,) = await _say(wizard, "done")
    assert not_yet == (
        'not yet — TELEGRAM_BOT_TOKEN isn\'t set. paste it here, or add to .env and reply "done".'
    )
    assert wizard.alive
    assert save.calls == []

    # .env by hand is still a way in — done checks honestly and connects
    env.add("TELEGRAM_BOT_TOKEN")
    (applied,) = await _say(wizard, "done")
    assert applied == f"the key's there — connecting telegram now.\n{_APPLY_TELEGRAM}"
    assert applier.calls == [("set", "messaging.telegram.enabled", True, "walk")]
    assert not wizard.alive


async def test_keys_already_set_connect_without_asking_again() -> None:
    wizard, applier, save, *_ = _wizard(env={"TELEGRAM_BOT_TOKEN"})
    applied = await wizard.start_add("telegram")
    assert applied == f"the key's already there — connecting telegram now.\n{_APPLY_TELEGRAM}"
    assert save.calls == []  # nothing was asked for — it was all there
    assert applier.calls == [("set", "messaging.telegram.enabled", True, "walk")]
    assert not wizard.alive


async def test_a_mis_paste_is_reasked_and_a_menu_number_refused() -> None:
    wizard, applier, save, *_ = _wizard()
    await wizard.start_add("telegram")
    # two words is a mis-paste, not a changed subject — re-ask, keep waiting
    (reask,) = await _say(wizard, "abc123 def456")
    assert reask == "just the value itself — paste only the value"
    # a short all-digits paste is a menu number — storing it would shadow
    # a good .env line, so it's refused before anything is stored
    (refused,) = await _say(wizard, "1")
    assert refused == "that's a menu number — paste the key's value itself"
    assert save.calls == []
    assert applier.calls == []
    assert wizard.alive

    # a long numeric value is a value — tokens can legitimately be digits
    (applied,) = await _say(wizard, "1234567890")
    assert applied == f"stored — switching telegram on now.\n{_APPLY_TELEGRAM}"
    assert save.calls == [("TELEGRAM_BOT_TOKEN", "1234567890", "telegram")]


async def test_surrounding_quotes_are_stripped_before_storing() -> None:
    wizard, _, save, *_ = _wizard()
    await wizard.start_add("telegram")
    await _say(wizard, '"123:AAbbCC"')
    assert save.calls == [("TELEGRAM_BOT_TOKEN", "123:AAbbCC", "telegram")]


async def test_a_failed_save_falls_back_to_env_honestly() -> None:
    wizard, applier, _, *_ = _wizard(save=FakeSave(ok=False))
    await wizard.start_add("telegram")
    (refused,) = await _say(wizard, "123:AAbbCC")
    assert refused == (
        "that didn't work — nothing was stored. add TELEGRAM_BOT_TOKEN "
        'to .env and reply "done", or paste it again.'
    )
    assert wizard.alive  # the paste is re-askable — nothing was lost
    assert applier.calls == []


async def test_a_user_rule_forcing_a_hold_relays_its_card_verbatim() -> None:
    """Origin "walk" is the loop's argument, not a promise: a user rule
    that says park still parks, and the walk relays the card as-is."""
    wizard, _, save, *_ = _wizard(applier=FakeApplier(reply=_PARK_CARD))
    await wizard.start_add("telegram")
    (card,) = await _say(wizard, "123:AAbbCC")
    assert card == f"stored — switching telegram on now.\n{_PARK_CARD}"
    assert save.calls == [("TELEGRAM_BOT_TOKEN", "123:AAbbCC", "telegram")]
    assert not wizard.alive


# -- the hub connect: the key once, the link per app --------------------------------


async def test_a_hub_app_serves_the_connect_link() -> None:
    wizard, applier, _, _, toolkits, authorize = _wizard(env={"COMPOSIO_API_KEY"})
    reply = await wizard.start_add("gmail")
    assert reply == (
        "Gmail — last step, let me in:\n"
        "open this and approve: https://hub.example.test/connect/gmail\n"
        "(I'll take it from there)"
    )
    assert toolkits.calls == ["gmail"]
    assert authorize.calls == ["gmail"]
    # the loop's handoff: read once to spawn the background waiter
    assert wizard.pending_connect == ("gmail", "req-gmail")
    assert wizard.alive  # the link wait holds until the waiter lands it
    assert applier.calls == []  # nothing is applied for a hosted link


async def test_the_link_wait_nudges_and_a_changed_subject_steps_aside() -> None:
    wizard, *_ = _wizard(env={"COMPOSIO_API_KEY"})
    await wizard.start_add("gmail")
    (still,) = await _say(wizard, "ok")
    assert still == _LINK_WAITING
    assert wizard.alive
    assert await _say(wizard, "what's the weather like today") == []
    assert not wizard.alive  # the message is normal chat; the waiter still lands


async def test_no_hub_key_asks_for_it_once_then_serves_the_link() -> None:
    wizard, _, save, _, toolkits, authorize = _wizard()
    recipe = await wizard.start_add("gmail")
    assert recipe == _HUB_KEY_RECIPE
    assert toolkits.calls == []  # the catalog can't answer before the key
    (link,) = await _say(wizard, "csk-123")
    # the key paste continues the connect it was asked for
    assert link == (
        "Gmail — last step, let me in:\n"
        "open this and approve: https://hub.example.test/connect/gmail\n"
        "(I'll take it from there)"
    )
    assert save.calls == [("COMPOSIO_API_KEY", "csk-123", "the app hub")]
    assert toolkits.calls == ["gmail"]
    assert authorize.calls == ["gmail"]
    assert wizard.pending_connect == ("gmail", "req-gmail")


async def test_a_hub_that_doesnt_answer_says_so_honestly() -> None:
    wizard, *_ = _wizard(env={"COMPOSIO_API_KEY"}, authorize=FakeAuthorize(url=None))
    reply = await wizard.start_add("gmail")
    assert reply == (
        "couldn't start the Gmail connection — the app hub didn't answer. "
        "nothing was connected; try again in a bit."
    )
    assert not wizard.alive
    assert wizard.pending_connect is None


async def test_a_fuzzy_name_offers_the_numbered_pick() -> None:
    wizard, *_ = _wizard(
        env={"COMPOSIO_API_KEY"},
        toolkits=FakeToolkits([("gmail", "Gmail"), ("google-mail", "Google Mail")]),
    )
    menu = await wizard.start_add("gmil")
    assert menu == "which gmil?\n1 — Gmail\n2 — Google Mail\n3 — none of these"
    (nudge,) = await _say(wizard, "maybe")
    assert nudge == "I didn't get that — reply with 1-3, or stop"
    assert wizard.alive
    (link,) = await _say(wizard, "1")
    assert link.startswith("Gmail — last step, let me in:")
    assert wizard.pending_connect == ("gmail", "req-gmail")


async def test_none_of_these_offers_the_generic_walk() -> None:
    wizard, *_ = _wizard(
        env={"COMPOSIO_API_KEY"},
        toolkits=FakeToolkits([("gmail", "Gmail"), ("google-mail", "Google Mail")]),
    )
    await wizard.start_add("gmil")
    (offer,) = await _say(wizard, "3")
    assert offer == (
        "fair enough. Set it up by hand over MCP?\n1 — yes, walk me through it\n2 — no"
    )
    (bye,) = await _say(wizard, "2")
    assert bye == _GOODBYE
    assert not wizard.alive


async def test_an_unknown_app_offers_the_generic_walk() -> None:
    wizard, *_ = _wizard(env={"COMPOSIO_API_KEY"}, toolkits=FakeToolkits([]))
    offer = await wizard.start_add("flurmbo")
    assert offer == (
        "I don't know an app called flurmbo — the hub's catalog has "
        "nothing by that name either. Set it up by hand over MCP?\n"
        "1 — yes, walk me through it\n"
        "2 — no"
    )
    (bye,) = await _say(wizard, "no")
    assert bye == _GOODBYE
    assert not wizard.alive


# -- the generic walk: any app that speaks mcp ------------------------------------


async def test_the_generic_stdio_walk_writes_a_by_name_reference() -> None:
    wizard, applier, save, *_ = _wizard(
        env={"COMPOSIO_API_KEY"},
        toolkits=FakeToolkits([]),
        applier=FakeApplier(reply=_APPLY_WEATHER),
    )
    await wizard.start_add("weather")
    replies = await _say(wizard, "1", "1", "npx -y mcp-weather", "WEATHER_API_KEY")
    assert replies[0] == (
        "how does weather connect?\n1 — stdio — a command run locally\n2 — http — a URL I call"
    )
    assert replies[1] == "what's the command? (e.g. npx -y mcp-weather)"
    assert replies[2] == (
        "does weather need a key or token? reply with the key's name "
        '(like WEATHER_API_KEY), or "none"'
    )
    assert replies[3] == (
        "weather needs a key.\npaste the WEATHER_API_KEY here — I'll take it from there."
    )
    (applied,) = await _say(wizard, "wx-123")
    assert applied == f"stored — connecting weather now.\n{_APPLY_WEATHER}"
    assert save.calls == [("WEATHER_API_KEY", "wx-123", "weather")]
    assert applier.calls == [
        (
            "add",
            "mcp_servers",
            {
                "name": "weather",
                "transport": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "mcp-weather"],
                    # the written config references the key by name only
                    "env": {"WEATHER_API_KEY": "$WEATHER_API_KEY"},
                },
            },
            "walk",
        )
    ]


async def test_the_generic_http_walk_gets_a_bearer_header() -> None:
    wizard, applier, _, *_ = _wizard(
        env={"COMPOSIO_API_KEY"},
        toolkits=FakeToolkits([]),
        applier=FakeApplier(reply=_APPLY_WEATHER),
    )
    await wizard.start_add("weather")
    await _say(wizard, "yes", "http", "https://mcp.example.com/mcp")
    await _say(wizard, "WEATHER_API_KEY", "wx-123")
    assert applier.calls == [
        (
            "add",
            "mcp_servers",
            {
                "name": "weather",
                "transport": {
                    "type": "http",
                    "url": "https://mcp.example.com/mcp",
                    "headers": {"Authorization": "Bearer $WEATHER_API_KEY"},
                },
            },
            "walk",
        )
    ]


async def test_a_generic_app_with_no_key_applies_directly() -> None:
    wizard, applier, save, *_ = _wizard(
        env={"COMPOSIO_API_KEY"},
        toolkits=FakeToolkits([]),
        applier=FakeApplier(reply=_APPLY_WEATHER),
    )
    await wizard.start_add("weather")
    await _say(wizard, "1", "stdio", "npx -y mcp-weather")
    (applied,) = await _say(wizard, "none")
    assert applied == _APPLY_WEATHER  # no head — there was nothing to paste
    assert applier.calls == [
        (
            "add",
            "mcp_servers",
            {
                "name": "weather",
                "transport": {"type": "stdio", "command": "npx", "args": ["-y", "mcp-weather"]},
            },
            "walk",
        )
    ]
    assert save.calls == []
    assert not wizard.alive


async def test_the_generic_walk_nudges_each_question() -> None:
    wizard, *_ = _wizard(env={"COMPOSIO_API_KEY"}, toolkits=FakeToolkits([]))
    await wizard.start_add("weather")
    replies = await _say(wizard, "1", "maybe", "2", "not a url")
    assert replies[1] == "I didn't get that — reply with 1-2, or stop"
    assert replies[3] == "just the URL — like https://mcp.example.com/mcp"
    assert wizard.alive

    # a command that doesn't parse is a nudge, not a dead end — the stdio path
    wizard, *_ = _wizard(env={"COMPOSIO_API_KEY"}, toolkits=FakeToolkits([]))
    await wizard.start_add("weather")
    await _say(wizard, "1", "1")
    (reask,) = await _say(wizard, 'npx -y "unterminated')
    assert reask == "that command didn't parse — check the quotes and try again"
    assert wizard.alive


# -- the permissions ask: the gate's question, after the link lands -----------------


async def test_the_connected_announcement_asks_what_the_app_may_do() -> None:
    wizard, *_ = _wizard()
    reply = wizard.enter_permissions("gmail", "me@gmail.com")
    assert reply == (
        "gmail connected — me@gmail.com.\n"
        "what may I do with gmail?\n"
        "1 — read freely, ask before acting (the default)\n"
        "2 — everything asks first\n"
        "3 — act freely"
    )
    assert wizard.alive


async def test_the_default_answer_writes_no_rule() -> None:
    wizard, applier, *_ = _wizard()
    wizard.enter_permissions("gmail", "me@gmail.com")
    (reply,) = await _say(wizard, "1")
    assert reply == (
        "that's the default — gmail reads run free; sends, deletes and anything new will ask first."
    )
    assert applier.calls == []  # the builtin ladder already does exactly this
    assert not wizard.alive


async def test_everything_asks_first_parks_an_authz_rule() -> None:
    """authz is a security root — it parks for the one-tap even from a
    walk, and the walk relays the applier's card verbatim."""
    wizard, applier, *_ = _wizard(applier=FakeApplier(reply=_PARK_CARD))
    wizard.enter_permissions("gmail", "me@gmail.com")
    (card,) = await _say(wizard, "2")
    assert card == _PARK_CARD
    assert applier.calls == [
        (
            "add",
            "authz.rules",
            {
                "tool_pattern": "composio__GMAIL_",
                "decision": "approve",
                "note": "gmail connected in chat — everything asks first",
            },
            "walk",
        )
    ]
    assert not wizard.alive


async def test_act_freely_writes_the_allow_rule() -> None:
    wizard, applier, *_ = _wizard()
    wizard.enter_permissions("gmail", "")
    (reply,) = await _say(wizard, "free")
    assert reply == _APPLY_TELEGRAM  # the applier's own words, relayed
    assert applier.calls == [
        (
            "add",
            "authz.rules",
            {
                "tool_pattern": "composio__GMAIL_",
                "decision": "allow",
                "note": "gmail connected in chat — freed to act",
            },
            "walk",
        )
    ]


async def test_the_permissions_menu_takes_words_and_nudges() -> None:
    wizard, applier, *_ = _wizard()
    wizard.enter_permissions("gmail", "")
    (nudge,) = await _say(wizard, "hmm")
    assert nudge == "I didn't get that — reply with 1-3, or stop"
    assert wizard.alive
    (reply,) = await _say(wizard, "ask")
    assert applier.calls[0][2]["decision"] == "approve"
    assert not wizard.alive

    # a sentence at the menu is a changed subject — the walk steps aside
    wizard, *_ = _wizard()
    wizard.enter_permissions("gmail", "")
    assert await _say(wizard, "what's the weather like today") == []
    assert not wizard.alive


# -- the walk mechanics -------------------------------------------------------------


async def test_cancel_words_end_the_walk_from_any_stage() -> None:
    for goodbye in ("stop", "cancel", "quit", "never mind"):
        wizard, *_ = _wizard()
        await wizard.start_add("telegram")
        (bye,) = await _say(wizard, goodbye)
        assert bye == _GOODBYE
        assert not wizard.alive

    # mid-paste, "stop" is still the way out — and nothing was stored
    wizard, applier, save, *_ = _wizard()
    await wizard.start_add("telegram")
    (bye,) = await _say(wizard, "stop")
    assert bye == _GOODBYE
    assert applier.calls == []
    assert save.calls == []
    assert not wizard.alive

    # "done" only answers at the paste stage — at the link wait it's a way out
    wizard, *_ = _wizard(env={"COMPOSIO_API_KEY"})
    await wizard.start_add("gmail")
    (bye,) = await _say(wizard, "done")
    assert bye == _GOODBYE
    assert not wizard.alive


async def test_fifteen_minutes_of_silence_quits_intercepting() -> None:
    wizard, *_ = _wizard()
    await wizard.start_add("telegram")
    mis_paste = "just the value itself — paste only the value"
    assert await wizard.handle("abc def", now=100.0) == mis_paste
    # the clock restarts at every answer — 899 seconds later is still in
    assert await wizard.handle("abc def", now=999.0) == mis_paste
    # 900.6 after the last one, the walk is gone and the message is normal chat
    assert await wizard.handle("abc def", now=1899.6) is None
    assert not wizard.alive
