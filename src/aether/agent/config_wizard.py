"""The guided /config walk: numbered questions, plain words, no paths.

Bare /config starts an interviewer over the same configuration the one-line
grammar reaches: pick an area, pick a setting, answer one question — and the
change goes through the exact same executor gate, so personal tuning applies
right away and a security-shaped path still parks for the one-tap card. The
wizard itself is a pure state machine: no I/O of its own, no LLM, nothing
ingested — every write and every read happens through the callbacks the loop
hands it (the shared applier and the manager's show/yaml_value). Its state
lives in memory — a walk is a conversation, not a record — and it quietly
steps aside when the user stops answering, changes the subject, sends a
command, or an llm change restarts the process mid-walk.
"""

from __future__ import annotations

import json
import shlex
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..config import AppConfig

# the loop's shared applier: (op, path[, value]) -> the reply to relay
ApplyFn = Callable[..., Awaitable[str]]
# the manager's show: (path) -> the plain-words render
ShowFn = Callable[[str | None], Awaitable[str]]
# what config.yaml holds at a path — the value a reset returns to
YamlValueFn = Callable[[str], Any]

_CANCEL_WORDS = frozenset({"stop", "cancel", "done", "quit", "exit", "nevermind", "never mind"})
_IDLE_SECONDS = 15 * 60  # a walk nobody answers stops intercepting messages
_MAX_MENU_TOKEN = 24  # longer than this at a menu is a changed subject

# menu answers may arrive as "1", "1.", "1)" — the punctuation is politeness
_STRIP_TAIL = "\"'.),!;"

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
_MCP_TAIL = (
    "(known apps have ready recipes — their links and .env lines — and "
    "/apps walks you through them. env keys or a poll schedule can come "
    "here too: /config add mcp_servers {json})"
)

_UNKNOWN = object()  # yaml_value couldn't answer — the confirm keeps it vague


def coerce_config_value(raw: str) -> Any:
    """Best-effort typing of a value said in chat: JSON first (covers
    arrays, objects, numbers, true/false/null), then the plain chat words,
    then int/float, else the raw string. The value is the raw remainder of
    the line, so quoted spaces survive."""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        inner = text[1:-1].strip()
        try:
            return json.loads(inner)
        except ValueError:
            return inner  # quoted means "take it literally"
    try:
        return json.loads(text)
    except ValueError:
        pass
    lowered = text.lower()
    if lowered in ("on", "yes", "true"):
        return True
    if lowered in ("off", "no", "false"):
        return False
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _plain(value: Any) -> str:
    """A value the way the walk says it — booleans read on/off, floats lose
    their trailing zero, none says none."""
    if value is None:
        return "none"
    if value is True:
        return "on"
    if value is False:
        return "off"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


@dataclass(frozen=True)
class _Setting:
    path: str
    label: str
    meaning: str
    prompt: str  # the one question, with {cur} where the current value goes
    options: tuple[tuple[str, str], ...] = ()  # numbered answers: (value, display)
    flow: str = ""  # "" scalar · "allowlist" · "rules" — guided list flows


@dataclass(frozen=True)
class _Section:
    key: str
    blurb: str
    settings: tuple[_Setting, ...]


def _messaging_settings() -> tuple[_Setting, ...]:
    return tuple(
        _Setting(
            f"messaging.{platform}.enabled",
            platform,
            f"whether I speak on {platform}",
            f"turn {platform} on or off? (currently {{cur}})",
        )
        for platform in ("telegram", "discord", "slack")
    )


_SECTIONS: tuple[_Section, ...] = (
    _Section(
        "llm",
        "the model I think with",
        (
            _Setting(
                "llm.provider",
                "provider",
                "who makes me think",
                "who should make me think? (currently {cur} — heads-up: "
                "llm changes restart me, back in a few seconds)\n"
                "1 — anthropic\n2 — openai\n3 — gemini\n4 — ollama",
                (
                    ("anthropic", "anthropic"),
                    ("openai", "openai"),
                    ("gemini", "gemini"),
                    ("ollama", "ollama"),
                ),
            ),
            _Setting(
                "llm.model",
                "model",
                "the model I think with",
                "which model? (currently {cur} — heads-up: llm changes "
                "restart me, back in a few seconds)",
            ),
            _Setting(
                "llm.vision_model",
                "vision_model",
                "a separate model that reads images",
                "which model should read images? (currently {cur} — or none to use my main model)",
            ),
            _Setting(
                "llm.salience_model",
                "salience_model",
                "a cheaper model for scoring what matters",
                "which cheaper model should score what matters? (currently "
                "{cur} — or none to use my main model)",
            ),
            _Setting(
                "llm.embedding_model",
                "embedding_model",
                "a small free model that makes my memory searchable",
                "which embedding model should search my memory? (currently "
                "{cur} — or none to turn search off)",
            ),
            _Setting(
                "llm.ollama_vision",
                "ollama_vision",
                "ollama only: whether my model reads images",
                "does your ollama model read images — on or off? (currently {cur})",
            ),
            _Setting(
                "llm.max_tokens",
                "max_tokens",
                "how much room my answers get",
                "how much room do my answers get, in tokens? (currently {cur})",
            ),
        ),
    ),
    _Section(
        "agent",
        "my rhythm: how often I check, how much I do",
        (
            _Setting(
                "agent.tick_seconds",
                "tick_seconds",
                "seconds between my checks",
                "how many seconds between checks? (currently {cur})",
            ),
            _Setting(
                "agent.max_tool_iterations",
                "max_tool_iterations",
                "tool calls I may chain in one turn",
                "how many tool calls may I chain in one turn? (currently {cur})",
            ),
            _Setting(
                "agent.daily_surface_cap",
                "daily_surface_cap",
                "most things I bring up unprompted per day",
                "at most how many things should I bring up unprompted per day? (currently {cur})",
            ),
            _Setting(
                "agent.quiet_hours",
                "quiet_hours",
                "hours I hold non-urgent notes",
                "which hours should I hold non-urgent notes? like 23:00-07:00, "
                "or none to clear (currently {cur})",
            ),
            _Setting(
                "agent.quiet_urgent_salience",
                "quiet_urgent_salience",
                "how urgent something must be to interrupt quiet",
                "during quiet hours, how urgent (0-10) must something be to "
                "reach you? (currently {cur})",
            ),
        ),
    ),
    _Section(
        "salience",
        "what's worth your attention",
        (
            _Setting(
                "salience.threshold",
                "threshold",
                "how interesting something must be to reach you",
                "how interesting (0-10) must something be to reach you? (currently {cur})",
            ),
            _Setting(
                "salience.rate_cap_per_hour",
                "rate_cap_per_hour",
                "events past this an hour are noise",
                "how many events an hour before I treat them as noise? (currently {cur})",
            ),
        ),
    ),
    _Section("messaging", "which chat apps I speak on", _messaging_settings()),
    _Section(
        "contacts",
        "who I'm allowed to see",
        (
            _Setting(
                "contacts.mode",
                "mode",
                "whether I see everyone or only the allowlist",
                "who may I see?\n1 — off — everyone\n2 — enforce — only the people "
                "on the allowlist\n(currently {cur})",
                (("off", "off"), ("enforce", "enforce")),
            ),
            _Setting(
                "contacts.allowlist",
                "allowlist",
                "who I'm allowed to see",
                "",
                flow="allowlist",
            ),
        ),
    ),
    _Section(
        "authz",
        "my permission gate",
        (
            _Setting(
                "authz.rules",
                "rules",
                "extra allow/approve/deny decisions ahead of the built-ins",
                "",
                flow="rules",
            ),
            _Setting(
                "authz.approval_ttl_hours",
                "approval_ttl_hours",
                "how long approvals wait before expiring",
                "how many hours before an un-answered approval expires? (currently {cur})",
            ),
        ),
    ),
    _Section("mcp_servers", "the apps I'm connected to", ()),
)

_SECTION_ALIASES = {"mcp": "mcp_servers"}


class ConfigWizard:
    """One guided walk, in memory. `handle` returns the walk's reply, or
    None when the message wasn't part of it — the caller drops the walk and
    lets the message be normal chat. `alive` says whether a next answer has
    anywhere to land."""

    def __init__(
        self,
        config: AppConfig,
        *,
        apply: ApplyFn,
        show: ShowFn,
        yaml_value: YamlValueFn | None = None,
    ) -> None:
        self._config = config  # THE live object — "today" values stay current
        self._apply = apply
        self._show = show
        self._yaml_value = yaml_value
        self._stage = "menu"
        self._at = time.monotonic()
        self._mode = "change"  # or "reset" — where the pick stage leads
        self._section: _Section | None = None
        self._setting: _Setting | None = None
        self._list_kind = ""  # "allowlist" | "rules" | "mcp"
        self._pending: dict[str, Any] = {}  # the entry being built
        self._reset_path = ""

    @property
    def alive(self) -> bool:
        """False once the walk is over — goodbye, changed subject, or stale.
        The loop drops its reference so nothing lands in a dead menu."""
        return self._stage != "dead"

    def start(self) -> str:
        """The opening menu — also what a mid-walk /config returns: the same
        fresh start, state and all."""
        self._stage = "menu"
        self._mode = "change"
        self._section = None
        self._setting = None
        self._list_kind = ""
        self._pending = {}
        self._at = time.monotonic()
        return _TOP_MENU

    async def handle(self, text: str, now: float | None = None) -> str | None:
        """One inbound message, answered or not. A str is the walk's reply;
        None means the walk is over and the message should fall through to
        normal chat — the user changed the subject, or let the walk go
        stale. Cancel words end the walk from any stage."""
        stamp = time.monotonic() if now is None else now
        if stamp - self._at > _IDLE_SECONDS:
            self._stage = "dead"  # stale: gone quietly, never intercept this one
            return None
        self._at = stamp
        text = text.strip()
        if text.lower() in _CANCEL_WORDS:
            self._stage = "dead"
            return _GOODBYE
        stage = self._stage
        if stage == "menu":
            return await self._answer_menu(text)
        if stage == "sec":
            return self._answer_sec(text)
        if stage == "pick":
            return self._answer_pick(text)
        if stage == "value":
            return await self._answer_value(text)
        if stage == "confirm_reset":
            return await self._answer_confirm_reset(text)
        if stage == "list":
            return await self._answer_list(text)
        if stage == "al_platform":
            return self._answer_al_platform(text)
        if stage == "al_handle":
            return await self._answer_al_handle(text)
        if stage == "al_remove":
            return await self._answer_al_remove(text)
        if stage == "az_pattern":
            return self._answer_az_pattern(text)
        if stage == "az_decision":
            return self._answer_az_decision(text)
        if stage == "az_note":
            return await self._answer_az_note(text)
        if stage == "az_remove":
            return await self._answer_az_remove(text)
        if stage == "mcp_name":
            return self._answer_mcp_name(text)
        if stage == "mcp_kind":
            return self._answer_mcp_kind(text)
        if stage == "mcp_target":
            return await self._answer_mcp_target(text)
        if stage == "mcp_onoff":
            return await self._answer_mcp_onoff(text)
        if stage == "mcp_remove":
            return await self._answer_mcp_remove(text)
        self._stage = "dead"
        return None

    # -- menu plumbing -----------------------------------------------------------

    @staticmethod
    def _menu_pick(text: str, span: int) -> int | None:
        """The numbered choice in a menu answer, or None when the text isn't
        one of this menu's numbers."""
        token = text.strip().lower().rstrip(_STRIP_TAIL)
        if not token or " " in token or not token.isdigit():
            return None
        choice = int(token)
        return choice if 1 <= choice <= span else None

    @staticmethod
    def _changed_subject(text: str) -> bool:
        """A menu answer is a word or two at most. Anything longer means the
        user started talking about something else — the walk steps aside."""
        stripped = text.strip()
        return len(stripped) > _MAX_MENU_TOKEN or " " in stripped

    def _nudge(self, span: int) -> str:
        return f"I didn't get that — reply with 1-{span}, or stop"

    def _to_menu(self) -> str:
        self._stage = "menu"
        return _AGAIN

    async def _apply_and_menu(
        self, op: str, path: str, value: Any = _UNKNOWN, extra: str = ""
    ) -> str:
        """Hand one change to the shared applier and relay its reply —
        confirmation or park card, whatever the gate said — then offer the
        top menu again. The value argument omitted entirely means reset."""
        if value is _UNKNOWN:
            reply = await self._apply(op, path)
        else:
            reply = await self._apply(op, path, value)
        note = f"\n{extra}" if extra else ""
        self._stage = "menu"
        return f"{reply}{note}\n{_AGAIN}"

    def _current(self, path: str) -> Any:
        node: Any = self._config
        for part in path.split("."):
            node = getattr(node, part)
        return node

    def _yaml(self, path: str) -> Any:
        """What config.yaml holds — for the reset question. A path the
        manager can't answer keeps the question vague, never wrong."""
        if self._yaml_value is None:
            return _UNKNOWN
        try:
            return self._yaml_value(path)
        except Exception:
            return _UNKNOWN

    def _entries_word(self, count: int) -> str:
        return f"{count} entr{'y' if count == 1 else 'ies'}"

    # -- the stages ---------------------------------------------------------------

    async def _answer_menu(self, text: str) -> str | None:
        pick = self._menu_pick(text, 3)
        if pick is None:
            if self._changed_subject(text):
                self._stage = "dead"
                return None
            return self._nudge(3)
        if pick == 1:
            overview = await self._show(None)
            self._stage = "menu"
            return f"{overview}\n\n{_AGAIN}"
        self._mode = "change" if pick == 2 else "reset"
        self._stage = "sec"
        return self._sec_menu()

    def _sec_menu(self) -> str:
        lines = ["which area?"]
        for i, section in enumerate(_SECTIONS, 1):
            lines.append(f"{i} — {section.key} — {section.blurb}")
        return "\n".join(lines)

    def _answer_sec(self, text: str) -> str | None:
        token = text.strip().lower().rstrip(_STRIP_TAIL)
        key = _SECTION_ALIASES.get(token, token)
        section = next((s for s in _SECTIONS if s.key == key), None)
        if section is None:
            pick = self._menu_pick(text, len(_SECTIONS))
            if pick is None:
                if self._changed_subject(text):
                    self._stage = "dead"
                    return None
                return self._nudge(len(_SECTIONS))
            section = _SECTIONS[pick - 1]
        self._section = section
        if section.key == "mcp_servers":
            self._list_kind = "mcp"
            self._stage = "list"
            return self._mcp_menu()
        self._stage = "pick"
        return self._pick_menu()

    def _pick_menu(self) -> str:
        section = self._section
        lines = [f"{section.key} today:"]
        for i, setting in enumerate(section.settings, 1):
            if setting.flow == "allowlist":
                current = self._entries_word(len(self._config.contacts.allowlist))
            elif setting.flow == "rules":
                current = self._entries_word(len(self._config.authz.rules))
            else:
                current = _plain(self._current(setting.path))
            lines.append(f"{i} — {setting.label} · {current} — {setting.meaning}")
        lines.append("which one?")
        return "\n".join(lines)

    def _answer_pick(self, text: str) -> str | None:
        section = self._section
        span = len(section.settings)
        pick = self._menu_pick(text, span)
        if pick is None:
            if self._changed_subject(text):
                self._stage = "dead"
                return None
            return self._nudge(span)
        setting = section.settings[pick - 1]
        self._setting = setting
        if self._mode == "reset":
            # a reset confirms for every kind of setting — lists too — since
            # the guided list menu only knows add/remove/back
            self._reset_path = setting.path
            self._stage = "confirm_reset"
            return self._reset_confirm(setting)
        if setting.flow:
            self._list_kind = setting.flow
            self._stage = "list"
            return self._list_menu()
        self._stage = "value"
        return setting.prompt.format(cur=_plain(self._current(setting.path)))

    def _reset_confirm(self, setting: _Setting) -> str:
        target = self._yaml(setting.path)
        if setting.flow == "allowlist":
            base = "put the allowlist back the way config.yaml had it?"
            if target is not _UNKNOWN:
                base = f"put the allowlist back to config.yaml's {self._entries_word(len(target))}?"
        elif setting.flow == "rules":
            base = "put the rules back the way config.yaml had it?"
            if target is not _UNKNOWN:
                base = f"put the rules back to config.yaml's {self._entries_word(len(target))}?"
        else:
            base = f"put {setting.label} back the way config.yaml had it?"
            if target is not _UNKNOWN:
                base = f"put {setting.label} back to config.yaml's {_plain(target)}?"
        return f"{base}\n1 — yes  2 — no"

    async def _answer_value(self, text: str) -> str | None:
        setting = self._setting
        if setting.options:
            choice = self._menu_pick(text, len(setting.options))
            if choice is None:
                if self._changed_subject(text):
                    self._stage = "dead"
                    return None
                return self._nudge(len(setting.options))
            value = setting.options[choice - 1][0]
        else:
            value = coerce_config_value(text)
        return await self._apply_and_menu("set", setting.path, value)

    async def _answer_confirm_reset(self, text: str) -> str | None:
        token = text.strip().lower().rstrip(_STRIP_TAIL)
        if token in ("1", "yes", "y", "yeah", "sure"):
            return await self._apply_and_menu("reset", self._reset_path)
        if token in ("2", "no", "n", "nah"):
            return self._to_menu()  # left exactly as it is
        if self._changed_subject(text):
            self._stage = "dead"
            return None
        return self._nudge(2)

    # -- the guided list flows ------------------------------------------------------

    def _list_menu(self) -> str:
        if self._list_kind == "allowlist":
            entries = self._config.contacts.allowlist
            lines = ["the allowlist today: (nothing on it yet)"]
            if entries:
                lines = ["the allowlist today:"]
                for rule in entries:
                    who = f"{rule.platform} {rule.handle}" if rule.platform != "*" else rule.handle
                    lines.append(f"· {who}")
            lines.append("1 — add someone  2 — remove someone  3 — back")
            return "\n".join(lines)
        rules = self._config.authz.rules
        lines = ["the rules today: (none yet)"]
        if rules:
            lines = ["the rules today:"]
            for rule in rules:
                note = f" ({rule.note})" if rule.note else ""
                lines.append(f"· {rule.tool_pattern} → {rule.decision}{note}")
        lines.append("1 — add a rule  2 — remove a rule  3 — back")
        return "\n".join(lines)

    async def _answer_list(self, text: str) -> str | None:
        if self._list_kind == "mcp":
            return await self._answer_mcp_list(text)
        span = 3
        pick = self._menu_pick(text, span)
        if pick is None:
            if self._changed_subject(text):
                self._stage = "dead"
                return None
            return self._nudge(span)
        if pick == 1:
            if self._list_kind == "allowlist":
                self._stage = "al_platform"
                return "which platform? (telegram, discord, slack — or * for any)"
            self._stage = "az_pattern"
            return (
                "which tool calls does it cover? (a pattern, like telegram__send or .*__delete.*)"
            )
        if pick == 2:
            return self._enter_remove()
        return self._to_menu()

    def _enter_remove(self) -> str:
        entries = self._remove_entries()
        if not entries:
            return f"there's nothing on it to remove.\n{self._list_menu()}"
        self._stage = "al_remove" if self._list_kind == "allowlist" else "az_remove"
        lines = [f"{i} — {entry}" for i, entry in enumerate(entries, 1)]
        lines.append("reply with a number, or stop")
        return "\n".join(lines)

    def _remove_entries(self) -> list[str]:
        if self._list_kind == "allowlist":
            return [
                f"{rule.platform} {rule.handle}" if rule.platform != "*" else rule.handle
                for rule in self._config.contacts.allowlist
            ]
        return [rule.tool_pattern for rule in self._config.authz.rules]

    def _answer_al_platform(self, text: str) -> str:
        token = text.strip()
        if not token or " " in token:
            return "just the platform — telegram, discord, slack, or * for any"
        self._pending["platform"] = token.lower()
        self._stage = "al_handle"
        return "what's their handle?"

    async def _answer_al_handle(self, text: str) -> str | None:
        handle = text.strip()
        if not handle:
            return "what's their handle?"
        if len(handle.split()) > 4:
            return "just their handle — like @mom or friend@example.com"
        platform = self._pending.get("platform", "*")
        value = handle if platform == "*" else f"{platform} {handle}"
        return await self._apply_and_menu("add", "contacts.allowlist", value)

    async def _answer_al_remove(self, text: str) -> str | None:
        rules = self._config.contacts.allowlist
        pick = self._menu_pick(text, len(rules))
        if pick is None:
            if self._changed_subject(text):
                self._stage = "dead"
                return None
            return self._nudge(len(rules))
        rule = rules[pick - 1]
        value = rule.handle if rule.platform == "*" else f"{rule.platform} {rule.handle}"
        return await self._apply_and_menu("remove", "contacts.allowlist", value)

    def _answer_az_pattern(self, text: str) -> str:
        pattern = text.strip()
        if not pattern or " " in pattern:  # the one-line add splits on spaces
            return "just the pattern, no spaces — like telegram__send or .*__delete.*"
        self._pending["pattern"] = pattern
        self._stage = "az_decision"
        return (
            "should those be allowed without asking, held for your tap, "
            "or denied?\n1 — allow  2 — approve (one tap)  3 — deny"
        )

    def _answer_az_decision(self, text: str) -> str | None:
        words = text.strip().lower().split()
        if words and words[0] in ("allow", "approve", "deny"):
            decision = words[0]
        else:
            pick = self._menu_pick(text, 3)
            if pick is None:
                if self._changed_subject(text):
                    self._stage = "dead"
                    return None
                return self._nudge(3)
            decision = ("allow", "approve", "deny")[pick - 1]
        self._pending["decision"] = decision
        self._stage = "az_note"
        return "want to add a note for the record? (a few words, or none)"

    async def _answer_az_note(self, text: str) -> str | None:
        note = "" if text.strip().lower() == "none" else text.strip()
        value = f"{self._pending['pattern']} {self._pending['decision']}"
        if note:
            value = f"{value} {note}"
        return await self._apply_and_menu("add", "authz.rules", value)

    async def _answer_az_remove(self, text: str) -> str | None:
        rules = self._config.authz.rules
        pick = self._menu_pick(text, len(rules))
        if pick is None:
            if self._changed_subject(text):
                self._stage = "dead"
                return None
            return self._nudge(len(rules))
        return await self._apply_and_menu("remove", "authz.rules", rules[pick - 1].tool_pattern)

    # -- the mcp_servers flow — apps as first-class menu items ----------------------

    def _mcp_menu(self) -> str:
        apps = self._config.mcp_servers
        head = "the apps today: (none yet)" if not apps else "the apps today:"
        lines = [head]
        for i, app in enumerate(apps, 1):
            polls = len(app.poll_tools)
            poll_word = "poll" if polls == 1 else "polls"
            lines.append(
                f"{i} — {app.name} ({'on' if app.enabled else 'off'}, {polls} {poll_word})"
            )
        n = len(apps)
        if self._mode == "change":
            lines.append(f"{n + 1} — add an app  {n + 2} — remove an app  {n + 3} — back")
        else:
            lines.append(f"{n + 1} — back")
        return "\n".join(lines)

    async def _answer_mcp_list(self, text: str) -> str | None:
        apps = self._config.mcp_servers
        span = len(apps) + (3 if self._mode == "change" else 1)
        pick = self._menu_pick(text, span)
        if pick is None:
            if self._changed_subject(text):
                self._stage = "dead"
                return None
            return self._nudge(span)
        if pick <= len(apps):
            app = apps[pick - 1]
            self._pending["app"] = app.name
            if self._mode == "reset":
                self._reset_path = f"mcp_servers.{app.name}.enabled"
                self._stage = "confirm_reset"
                return f"put {app.name} back the way config.yaml had it?\n1 — yes  2 — no"
            self._stage = "mcp_onoff"
            return f"turn {app.name} on or off? (currently {'on' if app.enabled else 'off'})"
        if self._mode == "change" and pick == len(apps) + 1:
            self._stage = "mcp_name"
            return "what's the app's name?"
        if self._mode == "change" and pick == len(apps) + 2:
            if not apps:
                return f"there are no apps to remove.\n{self._mcp_menu()}"
            self._stage = "mcp_remove"
            lines = [f"{i} — {app.name}" for i, app in enumerate(apps, 1)]
            lines.append("reply with a number, or stop")
            return "\n".join(lines)
        return self._to_menu()

    async def _answer_mcp_onoff(self, text: str) -> str | None:
        value = coerce_config_value(text)
        return await self._apply_and_menu(
            "set", f"mcp_servers.{self._pending['app']}.enabled", value
        )

    async def _answer_mcp_remove(self, text: str) -> str | None:
        apps = self._config.mcp_servers
        pick = self._menu_pick(text, len(apps))
        if pick is None:
            if self._changed_subject(text):
                self._stage = "dead"
                return None
            return self._nudge(len(apps))
        return await self._apply_and_menu("remove", "mcp_servers", apps[pick - 1].name)

    def _answer_mcp_name(self, text: str) -> str:
        name = text.strip()
        if not name or " " in name:
            return "just the name — one word, like mail or calendar"
        self._pending["name"] = name
        self._stage = "mcp_kind"
        return (
            f"how does {name} connect?\n1 — stdio — a command run locally\n2 — http — a URL I call"
        )

    def _answer_mcp_kind(self, text: str) -> str | None:
        words = text.strip().lower().split()
        if words and words[0] in ("stdio", "http"):
            kind = words[0]
        else:
            pick = self._menu_pick(text, 2)
            if pick is None:
                if self._changed_subject(text):
                    self._stage = "dead"
                    return None
                return self._nudge(2)
            kind = ("stdio", "http")[pick - 1]
        self._pending["kind"] = kind
        self._stage = "mcp_target"
        if kind == "stdio":
            return "what's the command? (e.g. npx -y mcp-mail-server)"
        return "what's the URL?"

    async def _answer_mcp_target(self, text: str) -> str | None:
        name = self._pending["name"]
        if self._pending["kind"] == "stdio":
            try:
                parts = shlex.split(text)
            except ValueError:
                return "that command didn't parse — check the quotes and try again"
            if not parts:
                return "what's the command? (e.g. npx -y mcp-mail-server)"
            value = {
                "name": name,
                "transport": {"type": "stdio", "command": parts[0], "args": parts[1:]},
            }
        else:
            url = text.strip()
            if not url or " " in url:
                return "just the URL — like https://mcp.example.com/mcp"
            value = {"name": name, "transport": {"type": "http", "url": url}}
        # the common case is guided; the one-liner carries the extras
        return await self._apply_and_menu("add", "mcp_servers", value, extra=_MCP_TAIL)
