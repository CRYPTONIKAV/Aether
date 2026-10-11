"""Messaging connector tests — all with fakes; no network, no SDK clients.

The platform SDK glue (handler registration, polling, sockets) is exercised
against injected fakes here, and live-verified manually during deployment
checks. The decide/inbound plumbing shared by every surface is tested via
the MessagingConnector base class.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import discord
import httpx
import pytest
from fakes import FakeApprovals
from telegram import Chat, Message, Update
from telegram.error import NetworkError

from aether.authz.approvals import APPROVED, DENIED
from aether.config import AppConfig, MessagingConfig, PlatformToggle, Settings
from aether.connectors import build_messaging_connectors
from aether.connectors.base import InboundMessage, MessagingConnector
from aether.connectors.discord import ApprovalView, DiscordConnector
from aether.connectors.slack import (
    APPROVE_ACTION,
    DENY_ACTION,
    SlackConnector,
    approval_blocks,
    event_to_inbound,
)
from aether.connectors.telegram import SEND_ATTEMPTS, TelegramConnector

# ---------------------------------------------------------------------------
# shared decide/inbound plumbing (base class)
# ---------------------------------------------------------------------------


async def test_fresh_decisions_fire_on_decision_stale_do_not() -> None:
    approvals = FakeApprovals()
    decisions: list[tuple[int, str]] = []

    async def on_decision(approval_id: int, decision: str) -> None:
        decisions.append((approval_id, decision))

    connector = MessagingConnector(
        enabled=True, approvals=approvals, on_decision=on_decision, on_inbound=None
    )
    fresh = await connector._decide(5, APPROVED)
    assert fresh is not None and fresh.status == APPROVED
    assert approvals.calls == [(5, APPROVED)]
    assert decisions == [(5, APPROVED)]

    # a stale/double press returns None and must not re-fire the executor
    approvals.result = None
    assert await connector._decide(5, DENIED) is None
    assert approvals.calls[-1] == (5, DENIED)
    assert decisions == [(5, APPROVED)]


# ---------------------------------------------------------------------------
# connector factory: enabled flag AND tokens
# ---------------------------------------------------------------------------


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def _config(telegram=False, discord_=False, slack=False) -> AppConfig:
    return AppConfig(
        messaging=MessagingConfig(
            telegram=PlatformToggle(enabled=telegram),
            discord=PlatformToggle(enabled=discord_),
            slack=PlatformToggle(enabled=slack),
        )
    )


def test_factory_skips_half_configured_surfaces() -> None:
    settings = _settings(telegram_bot_token="tg-token")  # discord/slack tokens missing
    built = build_messaging_connectors(settings, _config(telegram=True, discord_=True, slack=True))
    assert [c.name for c in built] == ["telegram"]


def test_factory_builds_each_surface_when_fully_configured() -> None:
    settings = _settings(
        telegram_bot_token="tg",
        discord_bot_token="dc",
        slack_bot_token="xoxb-",
        slack_app_token="xoxp-",
    )
    built = build_messaging_connectors(settings, _config(telegram=True, discord_=True, slack=True))
    assert [c.name for c in built] == ["telegram", "discord", "slack"]


def test_factory_with_everything_off_builds_nothing() -> None:
    settings = _settings(telegram_bot_token="tg")
    assert build_messaging_connectors(settings, _config()) == []


# ---------------------------------------------------------------------------
# telegram (fake Application)
# ---------------------------------------------------------------------------


class FakeTelegramBot:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_message(self, chat_id: int, text: str, reply_markup=None) -> SimpleNamespace:
        self.sent.append({"chat_id": chat_id, "text": text, "reply_markup": reply_markup})
        # the way telegram answers a send — the id that links a why? back
        return SimpleNamespace(message_id=100 + len(self.sent))


class FlakyTelegramBot(FakeTelegramBot):
    """Raises NetworkError on the first `failures` sends, then behaves — the
    shape of a throttled network dropping fresh TLS connects."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    async def send_message(self, chat_id: int, text: str, reply_markup=None) -> SimpleNamespace:
        if self.failures > 0:
            self.failures -= 1
            raise NetworkError("httpx.ConnectError: tls handshake failed")
        return await super().send_message(chat_id, text, reply_markup)


class FakeTelegramApp:
    def __init__(self) -> None:
        self.handlers: list[object] = []
        self.error_handlers: list[object] = []
        self.bot = FakeTelegramBot()
        self.updater = SimpleNamespace(start_polling=self._noop, stop=self._noop, polling=False)
        self.initialized = False
        self.started = False
        self.stopped = False
        self.shutdown = False

    async def _noop(self) -> None:
        pass

    def add_handler(self, handler) -> None:
        self.handlers.append(handler)

    def add_error_handler(self, handler) -> None:
        self.error_handlers.append(handler)

    async def initialize(self) -> None:
        self.initialized = True

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def shutdown(self) -> None:
        self.shutdown = True


def _tg_update(
    text: str, username: str = "vedant", user_id: int = 1, chat_id: int = 42, reply_to=None
):
    return SimpleNamespace(
        effective_message=SimpleNamespace(text=text, chat_id=chat_id, reply_to_message=reply_to),
        effective_user=SimpleNamespace(username=username, id=user_id),
    )


async def _started_telegram(approvals=None, decisions=None, inbound=None):
    app = FakeTelegramApp()
    factory_calls: list[str] = []

    def factory(token: str):
        factory_calls.append(token)
        return app

    connector = TelegramConnector(
        token="tg-token",
        approvals=approvals,
        on_decision=decisions,
        on_inbound=inbound,
        app_factory=factory,
    )
    await connector.start()
    assert factory_calls == ["tg-token"]
    assert app.initialized and app.started and len(app.handlers) == 2
    assert len(app.error_handlers) == 1  # update failures log, never dump a traceback
    return connector, app


async def test_telegram_start_registers_message_and_button_handlers() -> None:
    _, app = await _started_telegram()
    kinds = {type(h).__name__ for h in app.handlers}
    assert kinds == {"MessageHandler", "CallbackQueryHandler"}
    # stop is clean and idempotent
    await TelegramConnector(token="", enabled=False, app_factory=None).stop()


async def test_telegram_inbound_dm_becomes_an_inbound_message() -> None:
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, app = await _started_telegram(inbound=inbound)
    message_handler = app.handlers[0].callback
    await message_handler(_tg_update("what do you remember?"), None)

    assert received == [
        InboundMessage(
            surface="telegram", handle="@vedant", text="what do you remember?", chat_ref="42"
        )
    ]
    # non-text updates are ignored quietly
    empty = SimpleNamespace(
        effective_message=SimpleNamespace(text="", chat_id=1), effective_user=None
    )
    await message_handler(empty, None)
    assert len(received) == 1


async def test_telegram_a_reply_carries_what_it_answers() -> None:
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, app = await _started_telegram(inbound=inbound)
    await app.handlers[0].callback(
        _tg_update("why?", reply_to=SimpleNamespace(message_id=101, text="the invoice is sent")),
        None,
    )
    assert received == [
        InboundMessage(
            surface="telegram",
            handle="@vedant",
            text="why?",
            chat_ref="42",
            reply_to_id="101",
            reply_to_text="the invoice is sent",
        )
    ]


async def test_telegram_send_answers_with_the_message_id() -> None:
    connector, app = await _started_telegram()
    await app.handlers[0].callback(_tg_update("hi"), None)  # establish the reply target

    ref = await connector.send_to_user("the deploy finished")

    assert ref == "101"
    assert app.bot.sent[-1] == {"chat_id": 42, "text": "the deploy finished", "reply_markup": None}


async def test_telegram_commands_reach_the_loop_not_the_void() -> None:
    # the handler's filter admits /commands: the loop owns the deterministic
    # vocabulary (/verify), and what it doesn't handle flows to the model
    # like any other message — being dropped at the door only made Aether
    # look deaf
    _, app = await _started_telegram()
    update = Update(
        update_id=1,
        message=Message(
            message_id=10,
            date=datetime.now(UTC),
            chat=Chat(id=42, type=Chat.PRIVATE),
            text="/verify",
        ),
    )
    assert app.handlers[0].filters.check_update(update)


async def test_telegram_present_approval_sends_buttons() -> None:
    connector, app = await _started_telegram()
    # an inbound DM establishes the reply target; before that there is nowhere to send
    await app.handlers[0].callback(_tg_update("hi"), None)
    await connector.present_approval(7, "send 'hi' to @friend")
    sent = app.bot.sent[-1]
    assert sent["text"] == "send 'hi' to @friend"
    buttons = [b for row in sent["reply_markup"].inline_keyboard for b in row]
    assert [b.callback_data for b in buttons] == ["aether:approve:7", "aether:deny:7"]


async def test_telegram_button_press_decides_and_retires_the_buttons() -> None:
    approvals = FakeApprovals()
    decisions: list[tuple[int, str]] = []

    async def on_decision(approval_id: int, decision: str) -> None:
        decisions.append((approval_id, decision))

    connector, app = await _started_telegram(approvals=approvals, decisions=on_decision)
    button_handler = app.handlers[1].callback

    answered: list[str | None] = []
    cleared: list[bool] = []

    async def answer(text: str | None = None) -> None:
        answered.append(text)

    async def edit_message_reply_markup(reply_markup=None) -> None:
        cleared.append(reply_markup is None)

    parked = await approvals.create(tool_name="mail__send_message", params={"to": "a@b.c"})
    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            data=f"aether:approve:{parked.id}",
            answer=answer,
            edit_message_reply_markup=edit_message_reply_markup,
        )
    )
    await button_handler(update, None)
    assert approvals.calls == [(parked.id, APPROVED)]
    assert decisions == [(parked.id, APPROVED)]
    # a fresh decision needs no toast — the outcome message from _carry_out
    # is the acknowledgment, never an "Approval approved: handled." edit
    assert answered == [None]
    assert cleared == [True]

    # a double press: decide returns None because the row is no longer
    # pending — the executor is not re-fired, the buttons still clear, and
    # this time a toast tells the user the tap didn't do anything new
    await button_handler(update, None)
    assert decisions == [(parked.id, APPROVED)]
    assert cleared == [True, True]
    assert answered[-1] == "that request was already decided."


async def test_telegram_button_press_on_an_expired_approval_says_so() -> None:
    approvals = FakeApprovals()
    connector, app = await _started_telegram(approvals=approvals)
    button_handler = app.handlers[1].callback

    answered: list[str | None] = []

    async def answer(text: str | None = None) -> None:
        answered.append(text)

    async def edit_message_reply_markup(reply_markup=None) -> None:
        pass

    parked = await approvals.create(tool_name="mail__send_message", params={"to": "a@b.c"})
    parked.expires_at = datetime.now(UTC) - timedelta(seconds=1)  # the TTL already passed
    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            data=f"aether:approve:{parked.id}",
            answer=answer,
            edit_message_reply_markup=edit_message_reply_markup,
        )
    )
    await button_handler(update, None)
    assert answered == ["that approval has expired."]


async def test_telegram_button_press_on_an_unknown_approval_says_so() -> None:
    answered: list[str | None] = []

    async def answer(text: str | None = None) -> None:
        answered.append(text)

    async def edit_message_reply_markup(reply_markup=None) -> None:
        pass

    approvals = FakeApprovals()
    approvals.result = None  # the store found no such row at all
    connector, app = await _started_telegram(approvals=approvals)
    button_handler = app.handlers[1].callback
    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            data="aether:approve:999",
            answer=answer,
            edit_message_reply_markup=edit_message_reply_markup,
        )
    )
    await button_handler(update, None)
    assert answered == ["that request could not be found."]


async def test_telegram_send_retries_through_network_blips() -> None:
    connector, app = await _started_telegram()
    app.bot = FlakyTelegramBot(failures=2)
    connector._send_backoff = 0.0
    await app.handlers[0].callback(_tg_update("hi"), None)  # establish the reply target

    await connector.send_to_user("the deploy finished")

    assert app.bot.sent == [{"chat_id": 42, "text": "the deploy finished", "reply_markup": None}]


async def test_telegram_send_gives_up_after_sustained_failure() -> None:
    connector, app = await _started_telegram()
    flaky = FlakyTelegramBot(failures=99)
    app.bot = flaky
    connector._send_backoff = 0.0
    await app.handlers[0].callback(_tg_update("hi"), None)

    with pytest.raises(NetworkError):
        await connector.send_to_user("down")
    assert flaky.failures == 99 - SEND_ATTEMPTS  # exactly the bounded attempts
    assert app.bot.sent == []


async def test_telegram_disabled_never_builds_an_app() -> None:
    factory_calls: list[str] = []

    def factory(token: str):
        factory_calls.append(token)
        raise AssertionError("must not be called")

    connector = TelegramConnector(token="t", enabled=False, app_factory=factory)
    await connector.start()
    assert factory_calls == []
    await connector.stop()


# ---------------------------------------------------------------------------
# discord (fake Bot)
# ---------------------------------------------------------------------------


class FakeDiscordChannel:
    def __init__(self, channel_type) -> None:
        self.type = channel_type
        self.id = 555
        self.sent: list[dict] = []

    async def send(self, content=None, view=None) -> SimpleNamespace:
        self.sent.append({"content": content, "view": view})
        # the way discord answers a send — the id that links a why? back
        return SimpleNamespace(id=100 + len(self.sent))


class FakeDiscordUser:
    def __init__(self, name: str, bot: bool = False) -> None:
        self.name = name
        self.bot = bot
        self.id = 99


def _dc_message(text: str, *, dm: bool = True, bot: bool = False, reply_to=None):
    channel_type = discord.ChannelType.private if dm else discord.ChannelType.text
    return SimpleNamespace(
        author=FakeDiscordUser("vedant", bot=bot),
        channel=FakeDiscordChannel(channel_type),
        content=text,
        reference=reply_to,
    )


async def _started_discord(approvals=None, decisions=None, inbound=None):
    bot = SimpleNamespace(
        events={},
        started_with=None,
        closed=False,
    )

    async def start(token: str) -> None:
        bot.started_with = token

    async def close() -> None:
        bot.closed = True

    bot.start = start
    bot.close = close

    def event(coro) -> None:
        bot.events[coro.__name__] = coro

    bot.event = event

    connector = DiscordConnector(
        token="dc-token",
        approvals=approvals,
        on_decision=decisions,
        on_inbound=inbound,
        bot_factory=lambda: bot,
    )
    await connector.start()
    await asyncio.sleep(0)  # let the gateway task run to completion
    assert bot.started_with == "dc-token"
    return connector, bot


async def test_discord_registers_on_message_and_rejects_non_dms() -> None:
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, bot = await _started_discord(inbound=inbound)
    assert "on_message" in bot.events

    await connector.on_message(_dc_message("ping"))
    assert received == [
        InboundMessage(surface="discord", handle="vedant", text="ping", chat_ref="99")
    ]
    # channel chatter, other bots, and empty messages are ignored
    await connector.on_message(_dc_message("noise", dm=False))
    await connector.on_message(_dc_message("beep", bot=True))
    await connector.on_message(_dc_message(""))
    assert len(received) == 1


async def test_discord_a_reply_carries_what_it_answers_and_sends_answer_with_ids() -> None:
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, bot = await _started_discord(inbound=inbound)
    await connector.on_message(_dc_message("why?", reply_to=SimpleNamespace(message_id=77)))
    assert received[-1].reply_to_id == "77"
    assert received[-1].reply_to_text == ""  # the answered text would cost an extra fetch

    ref = await connector.send_to_user("the deploy finished")
    assert ref == "101"


async def test_discord_approval_view_buttons_decide() -> None:
    approvals = FakeApprovals()
    decisions: list[tuple[int, str]] = []

    async def on_decision(approval_id: int, decision: str) -> None:
        decisions.append((approval_id, decision))

    connector, _ = await _started_discord(approvals=approvals, decisions=on_decision)

    # an inbound DM establishes the reply channel; before that there is nowhere to send
    await connector.on_message(_dc_message("hi"))
    await connector.present_approval(3, "dm a meme")
    channel = connector._channel
    view = channel.sent[-1]["view"]
    assert isinstance(view, ApprovalView)
    assert len(view.children) == 2

    edits: list[dict] = []

    async def edit_message(content=None, view=None) -> None:
        edits.append({"content": content, "view": view})

    interaction = SimpleNamespace(response=SimpleNamespace(edit_message=edit_message))
    await view._finish(interaction, DENIED)
    assert approvals.calls == [(3, DENIED)]
    assert decisions == [(3, DENIED)]
    assert all(child.disabled for child in view.children)
    # the question text stays (no content override) — the tap only retires
    # the buttons; the outcome message is the acknowledgment
    assert edits == [{"content": None, "view": view}]


async def test_discord_approval_view_gives_feedback_on_a_stale_or_expired_tap() -> None:
    approvals = FakeApprovals()
    connector, _ = await _started_discord(approvals=approvals)
    await connector.on_message(_dc_message("hi"))

    parked = await approvals.create(tool_name="mail__send_message", params={"to": "a@b.c"})
    await connector.present_approval(parked.id, "send it?")
    view = connector._channel.sent[-1]["view"]

    edits: list[dict] = []

    async def edit_message(content=None, view=None) -> None:
        edits.append({"content": content, "view": view})

    interaction = SimpleNamespace(response=SimpleNamespace(edit_message=edit_message))
    await view._finish(interaction, APPROVED)
    assert edits[-1]["content"] is None  # fresh decision — no extra note

    # a double tap on the same (now-decided) approval gets a note, not silence
    await view._finish(interaction, APPROVED)
    assert edits[-1]["content"] == "that request was already decided."

    # a fresh approval whose TTL already passed reads as expired, not "decided"
    expired = await approvals.create(tool_name="mail__send_message", params={"to": "b@c.d"})
    expired.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await connector.present_approval(expired.id, "send it?")
    expired_view = connector._channel.sent[-1]["view"]
    await expired_view._finish(interaction, APPROVED)
    assert edits[-1]["content"] == "that approval has expired."


async def test_discord_disabled_never_builds_a_bot() -> None:
    def factory():
        raise AssertionError("must not be called")

    connector = DiscordConnector(token="t", enabled=False, bot_factory=factory)
    await connector.start()
    await connector.stop()


# ---------------------------------------------------------------------------
# slack (fake AsyncApp + socket handler)
# ---------------------------------------------------------------------------


class FakeSlackClient:
    def __init__(self) -> None:
        self.posted: list[dict] = []
        self.updated: list[dict] = []

    async def chat_postMessage(self, channel: str, text: str, blocks=None) -> dict:
        self.posted.append({"channel": channel, "text": text, "blocks": blocks})
        # the way slack answers a send — the ts that links a why? back
        return {"ts": f"1758000000.{len(self.posted):06d}"}

    async def chat_update(self, channel: str, ts: str, text: str, blocks=None) -> None:
        self.updated.append({"channel": channel, "ts": ts, "text": text, "blocks": blocks})


class FakeSlackApp:
    def __init__(self) -> None:
        self.listeners: dict[str, object] = {}
        self.client = FakeSlackClient()

    def event(self, name: str):
        def register(fn):
            self.listeners[name] = fn
            return fn

        return register

    def action(self, action_id: str):
        def register(fn):
            self.listeners[action_id] = fn
            return fn

        return register


class FakeSlackSocket:
    def __init__(self) -> None:
        self.connected = False

    async def connect_async(self) -> None:
        self.connected = True

    async def disconnect_async(self) -> None:
        self.connected = False


def test_slack_event_to_inbound_filters() -> None:
    good = {"channel_type": "im", "user": "U123", "text": "hello", "channel": "D1"}
    assert event_to_inbound(good) == InboundMessage("slack", "U123", "hello", "D1")
    # channel noise, bot chatter, and empty events never reach the agent
    assert event_to_inbound({**good, "channel_type": "channel"}) is None
    assert event_to_inbound({**good, "subtype": "bot_message"}) is None
    assert event_to_inbound({**good, "bot_id": "B1"}) is None
    assert event_to_inbound({**good, "text": ""}) is None
    assert event_to_inbound({}) is None


def test_slack_a_thread_reply_carries_what_it_answers() -> None:
    good = {"channel_type": "im", "user": "U123", "text": "hello", "channel": "D1"}
    # a thread reply points at the message it answers — that id is what
    # links a why? back to the trace that produced it
    assert event_to_inbound({**good, "thread_ts": "123.400"}) == InboundMessage(
        "slack", "U123", "hello", "D1", reply_to_id="123.400"
    )


def test_slack_approval_blocks_carry_action_ids_and_values() -> None:
    blocks = approval_blocks(11, "dm the team")
    assert blocks[0]["text"]["text"] == "dm the team"
    elements = blocks[1]["elements"]
    assert [(e["action_id"], e["value"]) for e in elements] == [
        (APPROVE_ACTION, "11"),
        (DENY_ACTION, "11"),
    ]


async def _started_slack(approvals=None, decisions=None, inbound=None):
    app = FakeSlackApp()
    socket = FakeSlackSocket()
    connector = SlackConnector(
        bot_token="xoxb-t",
        app_token="xoxp-a",
        approvals=approvals,
        on_decision=decisions,
        on_inbound=inbound,
        app_factory=lambda token: app,
        socket_factory=lambda a, tok: socket,
    )
    await connector.start()
    assert socket.connected
    assert set(app.listeners) == {"message", APPROVE_ACTION, DENY_ACTION}
    return connector, app, socket


async def test_slack_inbound_dm_reaches_the_handler_and_replies_work() -> None:
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, app, _ = await _started_slack(inbound=inbound)
    await app.listeners["message"](
        {"channel_type": "im", "user": "U123", "text": "status?", "channel": "D1"},
        None,
        app.client,
    )
    await app.listeners["message"](
        {"channel_type": "channel", "user": "U123", "text": "noise", "channel": "C9"},
        None,
        app.client,
    )
    assert received == [InboundMessage("slack", "U123", "status?", "D1")]

    await connector.send_to_user("all quiet")
    assert app.client.posted[-1] == {"channel": "D1", "text": "all quiet", "blocks": None}

    await connector.present_approval(4, "dm the team")
    posted = app.client.posted[-1]
    assert posted["blocks"][1]["elements"][0]["action_id"] == APPROVE_ACTION


async def test_slack_send_answers_with_the_message_ts() -> None:
    async def inbound(message: InboundMessage) -> None:
        pass

    connector, app, _ = await _started_slack(inbound=inbound)
    await app.listeners["message"](
        {"channel_type": "im", "user": "U123", "text": "hi", "channel": "D1"},
        None,
        app.client,
    )

    ref = await connector.send_to_user("the deploy finished")

    assert ref == "1758000000.000001"


async def test_slack_button_flow_decides_and_updates_the_message() -> None:
    approvals = FakeApprovals()
    decisions: list[tuple[int, str]] = []

    async def on_decision(approval_id: int, decision: str) -> None:
        decisions.append((approval_id, decision))

    connector, app, _ = await _started_slack(approvals=approvals, decisions=on_decision)
    acked: list[bool] = []

    async def ack() -> None:
        acked.append(True)

    body = {
        "actions": [{"value": "12"}],
        "channel": {"id": "D1"},
        "message": {
            "ts": "123.456",
            "text": "dm the team?",
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": "dm the team?"}},
                {"type": "actions", "block_id": "aether:12"},
            ],
        },
    }
    await app.listeners[DENY_ACTION](ack, body, app.client)
    assert approvals.calls == [(12, DENIED)]
    assert decisions == [(12, DENIED)]
    assert acked == [True]
    # the tap retires the buttons and keeps the question — the outcome
    # message is the acknowledgment, never an "Approval denied: handled." edit
    updated = app.client.updated[-1]
    assert updated["text"] == "dm the team?"
    assert [b["type"] for b in updated["blocks"]] == ["section"]

    # garbage payloads are logged and dropped, never raised
    await app.listeners[APPROVE_ACTION](ack, {"actions": [{"value": "oops"}]}, app.client)
    assert approvals.calls == [(12, DENIED)]


async def test_slack_button_flow_gives_feedback_on_a_stale_or_expired_tap() -> None:
    approvals = FakeApprovals()
    connector, app, _ = await _started_slack(approvals=approvals)

    async def ack() -> None:
        pass

    def _body(approval_id: int) -> dict:
        return {
            "actions": [{"value": str(approval_id)}],
            "channel": {"id": "D1"},
            "message": {
                "ts": "123.456",
                "text": "dm the team?",
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": "dm the team?"}},
                    {"type": "actions", "block_id": f"aether:{approval_id}"},
                ],
            },
        }

    parked = await approvals.create(tool_name="mail__send_message", params={"to": "a@b.c"})
    await app.listeners[APPROVE_ACTION](ack, _body(parked.id), app.client)
    fresh = app.client.updated[-1]
    assert [b["type"] for b in fresh["blocks"]] == ["section"]  # no note on a fresh decision

    # a double tap on the same (now-decided) approval adds a small note
    await app.listeners[APPROVE_ACTION](ack, _body(parked.id), app.client)
    stale = app.client.updated[-1]
    assert stale["blocks"][-1] == {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": "_that request was already decided._"}],
    }

    # a fresh approval whose TTL already passed reads as expired
    expired = await approvals.create(tool_name="mail__send_message", params={"to": "b@c.d"})
    expired.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await app.listeners[APPROVE_ACTION](ack, _body(expired.id), app.client)
    expired_update = app.client.updated[-1]
    assert expired_update["blocks"][-1]["elements"][0]["text"] == "_that approval has expired._"


async def test_slack_without_both_tokens_stays_down() -> None:
    connector = SlackConnector(
        bot_token="xoxb", app_token="", app_factory=None, socket_factory=None
    )
    await connector.start()
    assert connector._app is None
    await connector.stop()


# ---------------------------------------------------------------------------
# shared files: photos and documents reach the agent (all platforms, fakes)
# ---------------------------------------------------------------------------

from aether.connectors.attachments import MAX_ATTACHMENT_BYTES  # noqa: E402
from aether.connectors.base import Attachment  # noqa: E402


def _tg_file_update(
    payload: bytes,
    *,
    caption: str,
    photo: bool = False,
    document: SimpleNamespace | None = None,
    name: str = "file",
    mime: str = "",
    size: int = 9,
):
    async def download():
        return bytearray(payload)

    async def get_file():
        return SimpleNamespace(download_as_bytearray=download)

    item = SimpleNamespace(file_name=name, mime_type=mime, file_size=size, get_file=get_file)
    if document is None and not photo:
        document = item
    return SimpleNamespace(
        effective_message=SimpleNamespace(
            text=None,
            caption=caption,
            chat_id=42,
            photo=[item] if photo else [],
            document=document,
            reply_to_message=None,
        ),
        effective_user=SimpleNamespace(username="vedant", id=1),
    )


async def test_telegram_a_photo_with_a_caption_flows_as_an_image() -> None:
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, app = await _started_telegram(inbound=inbound)
    await app.handlers[0].callback(
        _tg_file_update(b"fake-jpg", caption="is this our cat?", photo=True), None
    )
    assert received == [
        InboundMessage(
            surface="telegram",
            handle="@vedant",
            text="is this our cat?",
            chat_ref="42",
            attachments=(Attachment("image", "photo.jpg", "image/jpeg", b"fake-jpg"),),
        )
    ]


async def test_telegram_a_document_flows_classified_by_name() -> None:
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, app = await _started_telegram(inbound=inbound)
    await app.handlers[0].callback(
        _tg_file_update(
            b"%PDF",
            caption="the invoice",
            name="invoice.pdf",
            mime="application/pdf",
        ),
        None,
    )
    assert received[0].text == "the invoice"
    assert received[0].attachments == (
        Attachment("doc", "invoice.pdf", "application/pdf", b"%PDF"),
    )


async def test_telegram_oversize_and_failed_downloads_skip_not_die() -> None:
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, app = await _started_telegram(inbound=inbound)
    # over the cap: the file is skipped, the words still flow
    await app.handlers[0].callback(
        _tg_file_update(b"x" * 5, caption="too big", size=MAX_ATTACHMENT_BYTES + 1), None
    )
    assert received[-1].text == "too big"
    assert received[-1].attachments == ()

    # a download that raises is skipped the same way
    async def explode():
        raise RuntimeError("network down")

    broken = SimpleNamespace(file_size=5, get_file=explode)
    update = SimpleNamespace(
        effective_message=SimpleNamespace(
            text=None,
            caption="try again",
            chat_id=42,
            photo=[broken],
            document=None,
            reply_to_message=None,
        ),
        effective_user=SimpleNamespace(username="vedant", id=1),
    )
    await app.handlers[0].callback(update, None)
    assert received[-1].text == "try again"
    assert received[-1].attachments == ()

    # and a message with neither words nor a file still emits nothing
    await app.handlers[0].callback(
        _tg_file_update(b"", caption="", photo=False, document=None), None
    )
    assert len(received) == 3


async def test_telegram_missing_or_lying_size_metadata_does_not_bypass_the_cap(monkeypatch) -> None:
    """Telegram's file_size is documented Optional — a missing, zero, or
    undercounted report must never be read as proof the file is safe. The
    actual downloaded bytes are what gets checked."""
    monkeypatch.setattr("aether.connectors.telegram.MAX_ATTACHMENT_BYTES", 10)
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, app = await _started_telegram(inbound=inbound)
    oversized = b"x" * 11

    for size in (None, 0, 5):  # missing, zero, and an undercounted report
        await app.handlers[0].callback(_tg_file_update(oversized, caption="x", size=size), None)
        assert received[-1].attachments == (), f"size={size} must not bypass the cap"


async def test_telegram_download_cap_is_exact_at_the_boundary(monkeypatch) -> None:
    monkeypatch.setattr("aether.connectors.telegram.MAX_ATTACHMENT_BYTES", 10)
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, app = await _started_telegram(inbound=inbound)

    await app.handlers[0].callback(_tg_file_update(b"x" * 10, caption="at cap", size=10), None)
    assert len(received[-1].attachments) == 1

    await app.handlers[0].callback(_tg_file_update(b"x" * 11, caption="over cap", size=11), None)
    assert received[-1].attachments == ()


class FakeDiscordAttachment:
    def __init__(self, filename: str, content_type: str, size: int, payload) -> None:
        self.filename = filename
        self.content_type = content_type
        self.size = size
        self._payload = payload

    async def read(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _dc_file_message(attachments: list, *, content: str = "", dm: bool = True):
    channel_type = discord.ChannelType.private if dm else discord.ChannelType.text
    return SimpleNamespace(
        author=FakeDiscordUser("vedant", bot=False),
        channel=FakeDiscordChannel(channel_type),
        content=content,
        reference=None,
        attachments=attachments,
    )


async def test_discord_an_attachment_only_dm_flows() -> None:
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, _ = await _started_discord(inbound=inbound)
    png = FakeDiscordAttachment("cat.png", "image/png", 4, b"png-bytes")
    await connector.on_message(_dc_file_message([png]))
    assert received == [
        InboundMessage(
            surface="discord",
            handle="vedant",
            text="",
            chat_ref="99",
            attachments=(Attachment("image", "cat.png", "image/png", b"png-bytes"),),
        )
    ]


async def test_discord_attachment_woes_skip_not_die() -> None:
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, _ = await _started_discord(inbound=inbound)
    # a missing content type classifies from the filename
    mystery = FakeDiscordAttachment("photo.jpg", "", 3, b"jpg")
    # a failed read is skipped
    dead = FakeDiscordAttachment("lost.pdf", "application/pdf", 3, RuntimeError("gone"))
    # over the cap is skipped before downloading
    huge = FakeDiscordAttachment("huge.png", "image/png", MAX_ATTACHMENT_BYTES + 1, b"")
    await connector.on_message(_dc_file_message([mystery, dead, huge], content="see this?"))
    assert received[-1].text == "see this?"
    assert received[-1].attachments == (Attachment("image", "photo.jpg", "", b"jpg"),)


async def test_discord_missing_or_lying_size_metadata_does_not_bypass_the_cap(monkeypatch) -> None:
    """Discord's API normally requires `size`, but nothing stops a test
    double — or a future API quirk — from omitting or undercounting it; the
    actual downloaded bytes are what gets checked regardless."""
    monkeypatch.setattr("aether.connectors.discord.MAX_ATTACHMENT_BYTES", 10)
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, _ = await _started_discord(inbound=inbound)
    oversized = b"x" * 11

    for size in (None, 0, 5):  # missing, zero, and an undercounted report
        att = FakeDiscordAttachment("big.bin", "application/octet-stream", size, oversized)
        await connector.on_message(_dc_file_message([att], content="x"))
        assert received[-1].attachments == (), f"size={size} must not bypass the cap"


async def test_discord_download_cap_is_exact_at_the_boundary(monkeypatch) -> None:
    monkeypatch.setattr("aether.connectors.discord.MAX_ATTACHMENT_BYTES", 10)
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, _ = await _started_discord(inbound=inbound)

    at_cap = FakeDiscordAttachment("f", "application/octet-stream", 10, b"x" * 10)
    await connector.on_message(_dc_file_message([at_cap], content="x"))
    assert len(received[-1].attachments) == 1

    over_cap = FakeDiscordAttachment("f", "application/octet-stream", 11, b"x" * 11)
    await connector.on_message(_dc_file_message([over_cap], content="x"))
    assert received[-1].attachments == ()


def _slack_file_event() -> dict:
    return {
        "channel_type": "im",
        "user": "U123",
        "channel": "D1",
        "subtype": "file_share",
        "files": [
            {"id": "F1", "name": "notes.md", "mimetype": "text/markdown", "size": 5},
        ],
    }


def test_slack_a_file_share_event_with_bytes_carries_the_attachment() -> None:
    inbound = event_to_inbound(_slack_file_event(), {"F1": b"hello"})
    assert inbound is not None
    assert inbound.attachments == (Attachment("doc", "notes.md", "text/markdown", b"hello"),)


def test_slack_a_file_without_its_bytes_still_flows_without_it() -> None:
    inbound = event_to_inbound(_slack_file_event(), {})
    assert inbound is not None
    assert inbound.attachments == ()


def test_slack_odd_file_events_are_filtered() -> None:
    assert event_to_inbound({**_slack_file_event(), "bot_id": "B1"}, {"F1": b"x"}) is None
    no_files = _slack_file_event()
    no_files.pop("files")
    assert event_to_inbound(no_files) is None


async def test_slack_the_connector_downloads_and_emits() -> None:
    received: list[InboundMessage] = []

    async def inbound(message: InboundMessage) -> None:
        received.append(message)

    connector, app, _ = await _started_slack(inbound=inbound)

    async def fake_download(files):
        return {"F1": b"hello"}

    connector._download_files = fake_download  # noqa: B010
    await app.listeners["message"](_slack_file_event(), None, app.client)
    assert received == [
        InboundMessage(
            surface="slack",
            handle="U123",
            text="",
            chat_ref="D1",
            attachments=(Attachment("doc", "notes.md", "text/markdown", b"hello"),),
        )
    ]


def _patch_slack_http(monkeypatch, payload: bytes, *, fail: bool = False) -> None:
    """Routes slack.py's real httpx.AsyncClient through a MockTransport, so
    _download_files' actual streaming code runs end to end against a
    canned response instead of the network."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, content=payload)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("aether.connectors.slack.httpx.AsyncClient", client_factory)


async def test_slack_missing_or_lying_size_metadata_does_not_bypass_the_cap(monkeypatch) -> None:
    """A Slack file (e.g. an externally-shared one) can report size as 0 or
    omit it — the actual streamed bytes are what gets checked."""
    monkeypatch.setattr("aether.connectors.slack.MAX_ATTACHMENT_BYTES", 10)
    connector, _, _ = await _started_slack()
    oversized = b"x" * 11

    for size in (None, 0, 5):  # missing, zero, and an undercounted report
        _patch_slack_http(monkeypatch, oversized)
        out = await connector._download_files(
            [{"id": "F1", "name": "big.bin", "url_private_download": "https://x/f", "size": size}]
        )
        assert out == {}, f"size={size} must not bypass the cap"


async def test_slack_download_cap_is_exact_at_the_boundary(monkeypatch) -> None:
    monkeypatch.setattr("aether.connectors.slack.MAX_ATTACHMENT_BYTES", 10)
    connector, _, _ = await _started_slack()

    _patch_slack_http(monkeypatch, b"x" * 10)
    at_cap = await connector._download_files(
        [{"id": "F1", "name": "f", "url_private_download": "https://x/f", "size": 10}]
    )
    assert at_cap == {"F1": b"x" * 10}

    _patch_slack_http(monkeypatch, b"x" * 11)
    over_cap = await connector._download_files(
        [{"id": "F1", "name": "f", "url_private_download": "https://x/f", "size": 11}]
    )
    assert over_cap == {}


async def test_slack_an_interrupted_download_is_skipped_and_nothing_partial_kept(
    monkeypatch,
) -> None:
    monkeypatch.setattr("aether.connectors.slack.MAX_ATTACHMENT_BYTES", 10)
    connector, _, _ = await _started_slack()
    _patch_slack_http(monkeypatch, b"irrelevant", fail=True)

    out = await connector._download_files(
        [{"id": "F1", "name": "f", "url_private_download": "https://x/f", "size": 5}]
    )
    assert out == {}
