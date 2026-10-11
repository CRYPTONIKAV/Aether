"""Messaging connector tests — all with fakes; no network, no SDK clients.

The platform SDK glue (handler registration, polling, sockets) is exercised
against injected fakes here, and live-verified manually during deployment
checks. The decide/inbound plumbing shared by every surface is tested via
the MessagingConnector base class.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import discord
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

    answered: list[bool] = []
    cleared: list[bool] = []

    async def answer() -> None:
        answered.append(True)

    async def edit_message_reply_markup(reply_markup=None) -> None:
        cleared.append(reply_markup is None)

    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            data="aether:approve:9",
            answer=answer,
            edit_message_reply_markup=edit_message_reply_markup,
        )
    )
    await button_handler(update, None)
    assert approvals.calls == [(9, APPROVED)]
    assert decisions == [(9, APPROVED)]
    assert answered == [True]
    # the tap retires the buttons — the outcome message is the
    # acknowledgment, never an "Approval approved: handled." edit
    assert cleared == [True]

    # a double press: decide returns None, executor not re-fired, buttons
    # still cleared (the first decision's outcome already landed)
    approvals.result = None
    await button_handler(update, None)
    assert decisions == [(9, APPROVED)]
    assert cleared == [True, True]


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


async def test_slack_without_both_tokens_stays_down() -> None:
    connector = SlackConnector(
        bot_token="xoxb", app_token="", app_factory=None, socket_factory=None
    )
    await connector.start()
    assert connector._app is None
    await connector.stop()
