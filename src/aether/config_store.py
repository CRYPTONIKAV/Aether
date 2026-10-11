"""The durable side of chat-made configuration: everything config.yaml
holds, readable and changeable from chat, with the change outliving the
boot.

On Render's free tier the filesystem is ephemeral — config.yaml is wiped
on every deploy — so a change made from chat has to live in Postgres or it
never happened. One row per setting path in ``config_overrides``, the value
encrypted and bound to its path through the cipher's AAD (an MCP transport
block can carry env secrets; a blob must never be swappable between rows).

``ConfigManager`` is the one interpretation of those rows, shared by both
entry points — the ``/config`` command and the ``set_config`` tool — so the
two can never drift. It never replaces section objects (Salience and the
EventStore hold references to them): it validates first, then mutates
fields and list contents in place. The yaml file (or pydantic defaults,
where no file exists) stays the base: rows merge on top of it at boot, and
reset deletes the row and falls back.

Live-apply per section: personal tuning just takes effect (its consumers
read config on every use), quiet hours reparse, the authz gate and the
approval window rebuild, messaging and mcp server sets diff their live
connections, and llm changes ask the process to restart itself — nobody
restarts Aether by hand. An unwired ref logs and continues: persistence is
still correct, and the next boot merges the same rows.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import asyncpg
from pydantic import BaseModel, TypeAdapter, ValidationError

from . import restart
from .authz.policy import Policy, PolicyError
from .config import (
    AppConfig,
    AuthzRule,
    ContactRule,
    MCPServerConfig,
    Settings,
)
from .connectors import build_messaging_connectors
from .memory.crypto import Cipher, CryptoError

log = logging.getLogger("aether.config_store")

_AAD = "config_overrides:value_enc:"

# the plain-words apply note for a path whose consumers read config on
# every use — no hook needed, the user just deserves to know when it bites
_APPLY_NOTES = {
    "agent.tick_seconds": " — applies from the next tick",
    "agent.max_tool_iterations": " — applies from the next turn",
    "agent.quiet_hours": " — the quiet window is live right away",
    "salience.threshold": " — applies to the next scored event",
    "salience.rate_cap_per_hour": " — applies to the next scored event",
    "contacts.mode": " — applies to the next message I see",
    "contacts.allowlist": " — applies to the next message I see",
}

_SECTION_NAMES = ("llm", "agent", "salience", "messaging", "contacts", "authz", "mcp_servers")


class ConfigError(ValueError):
    """A path or value the vocabulary refuses — plain words, never a crash."""


def _plain_value(value: Any) -> str:
    """A value the way it's said in chat — none/true/false, bare strings."""
    if value is None:
        return "none"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _plain_entry(entry: Any) -> str:
    """One list entry as it reads in chat."""
    if isinstance(entry, ContactRule):
        return f"{entry.platform}/{entry.handle}"
    if isinstance(entry, AuthzRule):
        text = f"{entry.tool_pattern} → {entry.decision}"
        if entry.param_pattern:
            text += f" (params matching {entry.param_pattern})"
        if entry.note:
            text += f" — {entry.note}"
        return text
    if isinstance(entry, MCPServerConfig):
        state = "on" if entry.enabled else "off"
        return f"{entry.name} ({state}, {len(entry.poll_tools)} poll{'s' if len(entry.poll_tools) != 1 else ''})"
    return str(entry)


def actions_word(count: int) -> str:
    """N action(s) — the executor note, /apps, and the late-ready announce
    all say it, so they share one wording and can never drift apart."""
    return f"{count} action{'s' if count != 1 else ''}"


def _plain_validation_error(exc: ValidationError, path: str) -> str:
    """A pydantic refusal restated in plain words — the user never sees
    validator jargon, they see what the setting needed."""
    err = exc.errors()[0]
    got = err.get("input")
    kind = str(err.get("type", ""))
    if kind == "missing":
        need = f"the {err.get('loc', ('value',))[-1]} field is missing"
    elif kind in ("int_parsing", "int_from_float", "int_parsing_size"):
        need = "it needs to be a whole number"
    elif kind.startswith("float"):
        need = "it needs to be a number"
    elif kind.startswith("bool"):
        need = "it needs to be true or false"
    elif kind.startswith("string"):
        need = "it needs to be a word"
    elif kind.startswith("literal"):
        expected = err.get("ctx", {}).get("expected")
        options = (
            ", ".join(str(o) for o in expected)
            if isinstance(expected, (tuple, list))
            else str(expected)
        )
        need = f"it must be one of: {options}"
    else:
        need = str(err.get("msg", "that value doesn't fit"))
    shown = (
        "nothing" if got is None else "that value" if isinstance(got, (dict, list)) else repr(got)
    )
    return f"{path} can't be {shown} — {need}."


class ConfigOverrides:
    """The durable layer: one row per setting path, the value encrypted and
    bound to that path. The settings.py one-shot upsert shape — the AAD
    already carries the client-chosen path, so there is no id to wait for."""

    def __init__(self, pool: asyncpg.Pool, cipher: Cipher) -> None:
        self._pool = pool
        self._cipher = cipher

    async def load(self) -> dict[str, Any]:
        """{path: value} for every stored override. A row that can't be
        decrypted (key rotation, drift) is skipped with a warning — boot
        must never crash over a stale row."""
        rows = await self._pool.fetch("SELECT path, value_enc FROM config_overrides")
        out: dict[str, Any] = {}
        for row in rows:
            try:
                data = self._cipher.decrypt_json(row["value_enc"], aad=_AAD + row["path"])
            except CryptoError:
                log.warning("config override %r can't be decrypted — skipping", row["path"])
                continue
            out[row["path"]] = data.get("value")
        return out

    async def upsert(self, path: str, value: Any) -> None:
        blob = self._cipher.encrypt_json({"value": value}, aad=_AAD + path)
        await self._pool.execute(
            "INSERT INTO config_overrides (path, value_enc, updated_at)"
            " VALUES ($1, $2, now())"
            " ON CONFLICT (path) DO UPDATE SET value_enc = $2, updated_at = now()",
            path,
            blob,
        )

    async def delete(self, path: str) -> None:
        await self._pool.execute("DELETE FROM config_overrides WHERE path = $1", path)

    async def delete_prefixed(self, prefix: str) -> None:
        """Every row under ``prefix.`` — the mcp list's item rows."""
        await self._pool.execute(
            "DELETE FROM config_overrides WHERE starts_with(path, $1::text || '.')",
            prefix,
        )


@dataclass
class _Resolved:
    """One settable path, pinned against both config objects at once: the
    live section (mutated) and the yaml snapshot (the reset target)."""

    root: str  # first path segment — selects the live-apply hook
    container: Any  # the live object holding the field
    field: str
    kind: str  # "scalar" | "list" | "item"
    section_model: type[BaseModel]  # what validates the container's fields
    entry_model: type[BaseModel] | None  # for list kind
    base_container: Any  # the same object on the yaml snapshot (None when absent)
    canonical: str  # the exact path persisted and echoed


class ConfigManager:
    """Rows in, live config out. Both chat paths — the /config command and
    the set_config tool — land here, so there is exactly one interpretation
    of every path and value."""

    def __init__(self, store: ConfigOverrides, settings: Settings, live: AppConfig) -> None:
        self._store = store
        self._settings = settings
        self._live = live  # THE shared object — mutated in place, never replaced
        self._base = live.model_copy(deep=True)  # the yaml snapshot reset falls back to
        self._loop: Any = None
        self._surfaces: Any = None
        self._tools: Any = None
        self._host: Any = None
        self._approvals: Any = None
        self._resolver: Any = None

    def wire(
        self,
        *,
        loop: Any = None,
        surfaces: Any = None,
        tools: Any = None,
        host: Any = None,
        approvals: Any = None,
        resolver: Any = None,
    ) -> None:
        """The live actors the apply hooks talk to. Called once after the
        connectors are up; anything left None logs later and the change
        still persists — it merges at the next boot."""
        self._loop = loop
        self._surfaces = surfaces
        self._tools = tools
        self._host = host
        self._approvals = approvals
        self._resolver = resolver

    # -- boot -------------------------------------------------------------------

    async def bootstrap(self) -> int:
        """Merge stored rows onto the live config, in place, before any
        consumer is wired. List roots land before their item rows (by path
        depth), so ``mcp_servers`` composes before ``mcp_servers.mail.enabled``
        can flip one of its items. A row the vocabulary no longer
        understands, or whose value no longer validates, is skipped with a
        warning and kept — a schema drift must never crash boot, and a yaml
        revert can still resurrect the row."""
        rows = await self._store.load()
        if not rows:
            return 0
        applied = 0
        for path in sorted(rows, key=lambda p: (p.count("."), p)):
            try:
                resolved = self._resolve(path)
                validated = self._validate_value(resolved, rows[path])
            except ConfigError as exc:
                log.warning("config override %r skipped: %s", path, exc)
                continue
            self._mutate(resolved, validated)
            applied += 1
        if applied:
            log.info("boot merged %d config override(s) on top of config.yaml", applied)
        return applied

    # -- reads -------------------------------------------------------------------

    async def show(self, path: str | None = None) -> str:
        """The configuration in plain words: the overview, one section, or
        one setting — with everything changed from config.yaml marked."""
        try:
            return self._render_show(path)
        except ConfigError as exc:
            return str(exc)
        except Exception:
            log.exception("config show failed")
            return "I couldn't read the configuration just now — nothing else is affected."

    async def effective(self) -> dict[str, Any]:
        """The config as the API serves it: the live sections plus every
        stored override. Read-only — the panel never writes config."""
        overrides = await self._store.load()
        return {
            "sections": self._live.model_dump(),
            "overrides": [{"path": p, "value": v} for p, v in sorted(overrides.items())],
        }

    def yaml_value(self, path: str) -> Any:
        """What config.yaml holds at this path — the value a reset returns
        to. The guided walk asks before its "put it back?" question, so a
        user approves a value, not a leap of faith. Raises ConfigError for a
        path the vocabulary doesn't know, same as set(); an app chat added
        was never in yaml, so its base container is None."""
        resolved = self._resolve(path.strip())
        if resolved.base_container is None:
            return None
        if resolved.kind == "list":
            return list(getattr(resolved.base_container, resolved.field))
        return getattr(resolved.base_container, resolved.field)

    # -- writes -------------------------------------------------------------------

    async def set(self, *, op: str, path: str, value: Any = None, source: str = "chat") -> str:
        """One change, end to end: validate → persist → mutate live → apply
        hooks. Never raises — refusals come back as plain words, the same
        words the model relays. For op=set the callers guarantee a value is
        present (`value=None` means an explicit clear); add/remove pass the
        entry tokens or object in `value`; reset ignores it."""
        try:
            return await self._set(op, path.strip(), value, source)
        except ConfigError as exc:
            return str(exc)
        except Exception:
            log.exception("config %s %s failed", op, path)
            return "that didn't work — nothing was changed."

    async def _set(self, op: str, raw_path: str, value: Any, source: str) -> str:
        if op not in ("set", "add", "remove", "reset"):
            raise ConfigError("the operation is one of set, add, remove, or reset.")
        if not raw_path:
            raise ConfigError("which setting? e.g. agent.tick_seconds.")
        resolved = self._resolve(raw_path)
        if op == "reset":
            return await self._reset(resolved)
        if op == "set":
            return await self._write(resolved, value, source)
        return await self._mutate_list(resolved, op, value)

    async def _write(self, resolved: _Resolved, value: Any, source: str) -> str:
        """op=set on any path — scalar, list, or item."""
        validated = self._validate_value(resolved, value)
        old_entries = (
            list(getattr(resolved.container, resolved.field)) if resolved.kind == "list" else []
        )
        await self._store.upsert(resolved.canonical, self._persist_value(validated, resolved.kind))
        if resolved.root == "mcp_servers" and resolved.kind == "list":
            # names that left the list take their item rows with them
            await self._forget_gone_mcp_rows(old_entries, validated)
        self._mutate(resolved, validated)
        note = await self._hooks(resolved) or _APPLY_NOTES.get(resolved.canonical, "")
        log.info("config %s set from %s", resolved.canonical, source)
        return self._confirm_set(resolved, validated) + note

    async def _reset(self, resolved: _Resolved) -> str:
        """Back to config.yaml: delete the row (and, for the mcp list, its
        item rows), restore from the snapshot, re-apply the hooks."""
        path = resolved.canonical
        await self._store.delete(path)
        if resolved.root == "mcp_servers" and resolved.kind == "list":
            await self._store.delete_prefixed(path)
        if resolved.base_container is None:
            # a chat-added app — the yaml has nothing to fall back to
            note = await self._hooks(resolved)
            return (
                f"{path} reset — config.yaml doesn't name an app "
                f"{resolved.container.name!r}, so the setting the list carries stands{note}."
            )
        if resolved.kind == "list":
            base_list = [
                e.model_copy(deep=True) for e in getattr(resolved.base_container, resolved.field)
            ]
            getattr(resolved.container, resolved.field)[:] = base_list
            body = f"{path} reset to config.yaml ({len(base_list)} entries)."
        else:
            base_value = getattr(resolved.base_container, resolved.field)
            setattr(resolved.container, resolved.field, base_value)
            body = f"{path} reset to config.yaml ({_plain_value(base_value)})."
        note = await self._hooks(resolved) or _APPLY_NOTES.get(path, "")
        return body + note

    async def _mutate_list(self, resolved: _Resolved, op: str, value: Any) -> str:
        """op=add/remove — build the full new list, persist it as the
        list-root row, swap it in. The whole list is what's stored; the
        add/remove split is only the chat-side sugar."""
        if resolved.kind != "list":
            raise ConfigError(
                f"add/remove work on lists — {resolved.canonical} is a single "
                "setting; use set (or reset to go back to config.yaml)."
            )
        if value is None or value == [] or (isinstance(value, str) and not value.strip()):
            raise ConfigError(
                f"{op} needs a value after {resolved.canonical} — e.g. "
                f"/config {op} contacts.allowlist telegram @friend."
            )
        current = list(getattr(resolved.container, resolved.field))
        if resolved.entry_model is ContactRule:
            new_entries = self._allowlist_change(current, value, op)
        elif resolved.entry_model is AuthzRule:
            new_entries = self._authz_rules_change(current, value, op)
        else:
            new_entries = self._mcp_servers_change(current, value, op)
        await self._store.upsert(resolved.canonical, [e.model_dump() for e in new_entries])
        if resolved.root == "mcp_servers":
            await self._forget_gone_mcp_rows(current, new_entries)
        getattr(resolved.container, resolved.field)[:] = new_entries
        note = await self._hooks(resolved) or _APPLY_NOTES.get(resolved.canonical, "")
        base_count = len(getattr(resolved.base_container, resolved.field))
        return (
            f"{resolved.canonical} updated — {len(new_entries)} entries now "
            f"(config.yaml has {base_count}).{note}"
        )

    async def _forget_gone_mcp_rows(self, old_entries: list[Any], new_entries: list[Any]) -> None:
        gone = {s.name for s in old_entries} - {s.name for s in new_entries}
        for name in gone:
            await self._store.delete(f"mcp_servers.{name}.enabled")

    # -- list entries ------------------------------------------------------------

    def _allowlist_change(
        self, current: list[ContactRule], value: Any, op: str
    ) -> list[ContactRule]:
        if isinstance(value, dict):
            try:
                entry = ContactRule.model_validate(value)
            except ValidationError as exc:
                raise ConfigError(_plain_validation_error(exc, "the allowlist entry")) from exc
            explicit_platform = True
        elif isinstance(value, str):
            tokens = value.split()
            if len(tokens) == 1:
                entry, explicit_platform = ContactRule(handle=tokens[0]), False
            elif len(tokens) == 2:
                entry, explicit_platform = ContactRule(platform=tokens[0], handle=tokens[1]), True
            else:
                raise ConfigError(
                    "an allowlist entry is a platform and a handle — e.g. telegram @friend."
                )
        else:
            raise ConfigError(
                "an allowlist entry is a platform and a handle — e.g. telegram @friend."
            )
        if not entry.handle.strip():
            raise ConfigError("an allowlist entry needs a handle.")
        if op == "add":
            if any(r.platform == entry.platform and r.handle == entry.handle for r in current):
                raise ConfigError(f"{entry.platform}/{entry.handle} is already on the allowlist.")
            return current + [entry]
        # remove: a bare handle matches any platform; a named platform must match too
        if explicit_platform:
            matches = [
                r for r in current if r.platform == entry.platform and r.handle == entry.handle
            ]
        else:
            matches = [r for r in current if r.handle == entry.handle]
        if not matches:
            raise ConfigError(f"{entry.handle} isn't on the allowlist.")
        if len(matches) > 1:
            listing = ", ".join(f"{r.platform}/{r.handle}" for r in matches)
            raise ConfigError(
                f"{entry.handle} is on the allowlist more than once ({listing}) — "
                "name the platform: remove contacts.allowlist <platform> <handle>."
            )
        return [r for r in current if r is not matches[0]]

    def _authz_rules_change(self, current: list[AuthzRule], value: Any, op: str) -> list[AuthzRule]:
        if op != "add":
            # remove — the value is just the pattern the rule was added with
            pattern = (
                str(value.get("tool_pattern", "")).strip()
                if isinstance(value, dict)
                else str(value).strip()
            )
            matches = [r for r in current if r.tool_pattern == pattern]
            if not matches:
                raise ConfigError(f"no rule for {pattern!r} in authz.rules.")
            return [r for r in current if r.tool_pattern != pattern]
        if isinstance(value, dict):
            try:
                entry = AuthzRule.model_validate(value)
            except ValidationError as exc:
                raise ConfigError(_plain_validation_error(exc, "the authz rule")) from exc
        elif isinstance(value, str):
            tokens = value.split(maxsplit=2)
            if len(tokens) < 2:
                raise ConfigError(
                    "a rule is a tool pattern and a decision — e.g. "
                    "add authz.rules mail__send allow."
                )
            note = tokens[2].strip() if len(tokens) == 3 else ""
            try:
                entry = AuthzRule(tool_pattern=tokens[0], decision=tokens[1].lower(), note=note)
            except ValidationError as exc:
                raise ConfigError(_plain_validation_error(exc, "the authz rule")) from exc
        else:
            raise ConfigError(
                "a rule is a tool pattern and a decision — e.g. add authz.rules mail__send allow."
            )
        # a bad regex is refused before it can persist — the gate must never
        # be handed a rule it can't compile
        try:
            re.compile(entry.tool_pattern)
            if entry.param_pattern:
                re.compile(entry.param_pattern)
        except re.error as exc:
            raise ConfigError(
                f"that rule's pattern {entry.tool_pattern!r} isn't valid regex: {exc}"
            ) from exc
        if any(
            r.tool_pattern == entry.tool_pattern and r.param_pattern == entry.param_pattern
            for r in current
        ):
            raise ConfigError(
                f"a rule for {entry.tool_pattern!r} already exists — remove it "
                "first if you mean to replace it."
            )
        return current + [entry]

    def _mcp_servers_change(
        self, current: list[MCPServerConfig], value: Any, op: str
    ) -> list[MCPServerConfig]:
        if op != "add":
            name = str(value.get("name", "") if isinstance(value, dict) else value).strip()
            if not name:
                raise ConfigError("remove mcp_servers needs the app's name.")
            if not any(s.name == name for s in current):
                raise ConfigError(f"no app named {name!r} in mcp_servers.")
            return [s for s in current if s.name != name]
        if not isinstance(value, dict):
            raise ConfigError(
                "an mcp_servers entry is a JSON object — e.g. "
                'add mcp_servers {"name": "mail", "transport": {"type": "http", "url": "…"}}.'
            )
        try:
            entry = MCPServerConfig.model_validate(value)
        except ValidationError as exc:
            raise ConfigError(_plain_validation_error(exc, "the mcp_servers entry")) from exc
        if any(s.name == entry.name for s in current):
            raise ConfigError(f"an app named {entry.name!r} is already in mcp_servers.")
        return current + [entry]

    # -- validate / mutate --------------------------------------------------------

    def _validate_value(self, resolved: _Resolved, value: Any) -> Any:
        """The validated form of `value` for this path — nothing is touched
        until validation has passed. Scalars and items re-validate the whole
        section copy (validate-then-mutate); lists go through a TypeAdapter."""
        if resolved.kind == "list":
            if not isinstance(value, list):
                raise ConfigError(
                    f"{resolved.canonical} is a list — pass a JSON array, or use add/remove."
                )
            try:
                entries = TypeAdapter(list[resolved.entry_model]).validate_python(value)  # type: ignore[arg-type]
            except ValidationError as exc:
                raise ConfigError(_plain_validation_error(exc, resolved.canonical)) from exc
            if resolved.root == "authz":
                # same refusal as _authz_rules_change, for the JSON-array path
                for rule in entries:
                    try:
                        re.compile(rule.tool_pattern)
                        if rule.param_pattern:
                            re.compile(rule.param_pattern)
                    except re.error as exc:
                        raise ConfigError(
                            f"that rule's pattern {rule.tool_pattern!r} isn't valid regex: {exc}"
                        ) from exc
            return entries
        try:
            candidate = resolved.section_model.model_validate(
                {**resolved.container.model_dump(), resolved.field: value}
            )
        except ValidationError as exc:
            raise ConfigError(_plain_validation_error(exc, resolved.canonical)) from exc
        return getattr(candidate, resolved.field)

    def _mutate(self, resolved: _Resolved, validated: Any) -> None:
        """Apply the validated value without replacing any section object
        the consumers hold references to: scalars setattr, lists swap their
        contents in place."""
        if resolved.kind == "list":
            getattr(resolved.container, resolved.field)[:] = validated
        else:
            setattr(resolved.container, resolved.field, validated)

    def _persist_value(self, validated: Any, kind: str) -> Any:
        if kind == "list":
            return [e.model_dump() for e in validated]
        return validated

    def _confirm_set(self, resolved: _Resolved, validated: Any) -> str:
        if resolved.kind == "list":
            base = getattr(resolved.base_container, resolved.field)
            return (
                f"{resolved.canonical} set — {len(validated)} entries now "
                f"(config.yaml has {len(base)})."
            )
        base = (
            getattr(resolved.base_container, resolved.field)
            if resolved.base_container is not None
            else None
        )
        marker = f" (config.yaml says {_plain_value(base)})" if validated != base else ""
        return f"{resolved.canonical} set to {_plain_value(validated)}{marker}."

    # -- live-apply hooks -----------------------------------------------------------

    async def _hooks(self, resolved: _Resolved) -> str:
        """Make the change take effect now, per section. Returns the
        plain-words apply note ("" when there's nothing to say). Unwired
        refs log and continue — the row is already persisted."""
        root = resolved.root
        if root == "llm":
            if restart.request():
                return " — restarting myself to load it, back in a few seconds"
            return " — it loads on the next boot"
        if root == "agent":
            if resolved.canonical == "agent.quiet_hours" and self._loop is not None:
                self._loop.reparse_quiet_hours()
            return ""
        if root == "authz":
            if self._approvals is not None:
                self._approvals.set_ttl(self._live.authz.approval_ttl_hours)
            if self._loop is not None:
                try:
                    self._loop._policy = Policy(self._live.authz.rules)
                except PolicyError as exc:
                    log.warning("the rebuilt gate rejected the rules (kept the old set): %s", exc)
            if resolved.canonical == "authz.approval_ttl_hours":
                hours = self._live.authz.approval_ttl_hours
                return f" — new approvals now wait {hours:g}h before expiring"
            return " — the gate uses the new rules from the next call"
        if root == "messaging":
            return await self._apply_messaging()
        if root == "mcp_servers":
            return await self._apply_mcp()
        return ""  # salience and contacts are read on every use

    async def _apply_messaging(self) -> str:
        """Diff the live connectors against the toggles: stop what went off,
        start what went on (a half-configured platform is named honestly —
        it stays off until the token exists in .env). Runs on every
        messaging write, even a same-value one, so a token that landed after
        boot still starts its platform — that's what 'it starts once one is
        there' promises."""
        if self._surfaces is None or self._loop is None:
            log.info("messaging changed but the fanout isn't wired — applies on the next boot")
            return ""
        fresh = build_messaging_connectors(
            await self._messaging_settings(),
            self._live,
            approvals=self._approvals,
            on_decision=self._loop.execute_decision,
            on_inbound=self._loop.handle_inbound,
        )
        desired = {c.name: c for c in fresh}
        current = {c.name: c for c in self._surfaces.connectors}
        notes: list[str] = []
        for name in current:
            if name in desired:
                continue
            connector = self._surfaces.remove_connector(name)
            if connector is not None:
                await connector.stop()
            notes.append(f"{name} is stopping")
        for connector in fresh:
            if connector.name in current:
                continue
            try:
                await connector.start()
            except Exception:
                log.exception("%s failed to start", connector.name)
                notes.append(f"{connector.name} couldn't start — check its token")
                continue
            self._surfaces.add_connectors([connector])
            notes.append(f"{connector.name} is starting up")
        for platform in ("telegram", "discord", "slack"):
            if getattr(self._live.messaging, platform).enabled and platform not in desired:
                notes.append(f"{platform} has no token yet — it starts once one is pasted or set")
        if not notes:
            return ""
        return " — " + "; ".join(notes)

    async def _messaging_settings(self) -> Settings:
        """Tokens read fresh through the resolver when one is wired — a key
        added to .env after boot is picked up here without a restart.
        Unwired (tests, early boot) keeps the boot snapshot."""
        if self._resolver is None:
            return self._settings
        values: dict[str, str] = {}
        for field in (
            "telegram_bot_token",
            "discord_bot_token",
            "slack_bot_token",
            "slack_app_token",
        ):
            value = await self._resolver.resolve(field.upper())
            if value:
                values[field] = value
        return Settings(_env_file=None, **values)

    async def _apply_mcp(self) -> str:
        """Make the host match the list: stop what left or changed, start
        what's new, drop the stopped apps' actions from the namespace, and
        re-sync so the started ones' actions are callable. The note tells
        the truth per started server — one that has answered is 'connected'
        with its actions counted, one that hasn't gets the promise the
        loop's tick reconcile later fulfills out loud."""
        if self._host is None or self._tools is None:
            log.info("mcp_servers changed but the host isn't wired — applies on the next boot")
            return ""
        diff = await self._host.apply_servers(self._live.mcp_servers)
        for name in diff["stopped"]:
            self._tools.drop_server(name)
        self._tools.sync_mcp_tools()
        if self._loop is not None:
            self._loop.rebuild_poll_targets()
        notes = [self._server_note(conn) for conn in diff["started"]]
        restarted = {c.name for c in diff["started"]}
        if gone := [n for n in diff["stopped"] if n not in restarted]:
            notes.append(
                f"{', '.join(gone)} stopped — their actions are gone until they're on again"
            )
        if not notes:
            return ""
        return " — " + "; ".join(notes)

    def _server_note(self, conn: Any) -> str:
        """One started server, honestly: answered → connected with its
        actions counted; silent so far → still trying, and the announce
        comes unprompted the moment it answers."""
        if conn.ready:
            return (
                f"{conn.name} connected · {actions_word(self._tools.server_action_count(conn.name))}"
                " in my vocabulary"
            )
        return f"{conn.name} hasn't answered yet — still trying, I'll tell you the moment it's up"

    # -- path resolution -----------------------------------------------------------

    def _resolve(self, path: str) -> _Resolved:
        """Pin one settable path against the live config and the yaml
        snapshot. Unknown paths refuse with a did-you-mean."""
        parts = [p for p in path.split(".") if p.strip()]
        if not parts:
            raise ConfigError("which setting? e.g. agent.tick_seconds.")
        root = parts[0].lower()
        rest = parts[1:]

        if root == "mcp_servers":
            if not rest:
                return _Resolved(
                    root,
                    self._live,
                    "mcp_servers",
                    "list",
                    AppConfig,
                    MCPServerConfig,
                    self._base,
                    "mcp_servers",
                )
            if len(rest) == 2 and rest[1].lower() == "enabled":
                name = rest[0]
                live_item = next((s for s in self._live.mcp_servers if s.name == name), None)
                base_item = next((s for s in self._base.mcp_servers if s.name == name), None)
                if live_item is None:
                    known = (
                        ", ".join(s.name for s in self._live.mcp_servers) or "none are configured"
                    )
                    raise ConfigError(f"no app named {name!r} in mcp_servers ({known}).")
                return _Resolved(
                    root,
                    live_item,
                    "enabled",
                    "item",
                    MCPServerConfig,
                    None,
                    base_item,
                    f"mcp_servers.{name}.enabled",
                )
            raise ConfigError(
                "mcp_servers paths are 'mcp_servers' (the whole list) or "
                "'mcp_servers.<name>.enabled' — e.g. mcp_servers.mail.enabled."
            )

        if root == "messaging":
            platform = rest[0].lower() if rest else ""
            if (
                len(rest) == 2
                and rest[1].lower() == "enabled"
                and platform in type(self._live.messaging).model_fields
            ):
                canonical = f"messaging.{platform}.enabled"
                return _Resolved(
                    root,
                    getattr(self._live.messaging, platform),
                    "enabled",
                    "scalar",
                    type(getattr(self._live.messaging, platform)),
                    None,
                    getattr(self._base.messaging, platform),
                    canonical,
                )
            raise ConfigError(
                "messaging paths look like messaging.<platform>.enabled — "
                "e.g. messaging.telegram.enabled."
            )

        if len(rest) != 1:
            raise ConfigError(
                f"{path.strip()!r} isn't a setting I know{self._suggest(path.strip())}"
            )
        section = getattr(self._live, root, None)
        if section is None or not isinstance(section, BaseModel):
            raise ConfigError(
                f"{path.strip()!r} isn't a setting I know{self._suggest(path.strip())}"
            )
        field = rest[0].lower()
        actual = {name.lower(): name for name in type(section).model_fields}.get(field)
        if actual is None:
            raise ConfigError(
                f"{path.strip()!r} isn't a setting I know{self._suggest(path.strip())}"
            )
        if (root, actual) == ("contacts", "allowlist"):
            return _Resolved(
                root,
                section,
                actual,
                "list",
                type(section),
                ContactRule,
                getattr(self._base, root),
                f"{root}.{actual}",
            )
        if (root, actual) == ("authz", "rules"):
            return _Resolved(
                root,
                section,
                actual,
                "list",
                type(section),
                AuthzRule,
                getattr(self._base, root),
                f"{root}.{actual}",
            )
        return _Resolved(
            root,
            section,
            actual,
            "scalar",
            type(section),
            None,
            getattr(self._base, root),
            f"{root}.{actual}",
        )

    def _known_paths(self) -> list[str]:
        """Every path the vocabulary accepts right now — the did-you-mean
        pool and the changed-from-yaml walk."""
        paths = ["mcp_servers"] + [f"mcp_servers.{s.name}.enabled" for s in self._live.mcp_servers]
        for section_name, model in (
            ("llm", type(self._live.llm)),
            ("agent", type(self._live.agent)),
            ("salience", type(self._live.salience)),
            ("contacts", type(self._live.contacts)),
            ("authz", type(self._live.authz)),
        ):
            paths.extend(f"{section_name}.{field}" for field in model.model_fields)
        paths.extend(
            f"messaging.{platform}.enabled" for platform in type(self._live.messaging).model_fields
        )
        return paths

    def _suggest(self, path: str) -> str:
        """A did-you-mean from rapidfuzz — near-miss paths get one guess,
        hopeless ones get nothing."""
        try:
            from rapidfuzz import process
        except ImportError:  # pragma: no cover - the dependency is declared
            return ""
        match = process.extractOne(
            path.lower(), [p.lower() for p in self._known_paths()], score_cutoff=70
        )
        if match is None:
            return ""
        return f" — did you mean {match[0]}?"

    # -- rendering ----------------------------------------------------------------

    def _render_show(self, path: str | None) -> str:
        if not path:
            return self._overview()
        key = path.strip().lower()
        if key in _SECTION_NAMES:
            return self._section_show(key)
        return self._one_show(self._resolve(path.strip()))

    def _overview(self) -> str:
        lines = ["my configuration — config.yaml plus whatever you've changed from chat:"]
        for name in _SECTION_NAMES:
            lines.append(self._describe_section(name))
        changed = self._changed_paths()
        if changed:
            parts = []
            for p in changed:
                resolved = self._resolve(p)
                live_v = getattr(resolved.container, resolved.field)
                if resolved.kind == "list":
                    parts.append(f"{p} = {len(live_v)} entries")
                else:
                    parts.append(f"{p} = {_plain_value(live_v)}")
            lines.append("Changed from config.yaml: " + " · ".join(parts))
        else:
            lines.append("Nothing changed from config.yaml yet.")
        lines.append(
            "/config show <section> for one section's settings · /config show <path> for one value"
        )
        return "\n".join(lines)

    def _changed_paths(self) -> list[str]:
        changed = []
        for p in self._known_paths():
            try:
                resolved = self._resolve(p)
            except ConfigError:
                continue
            live_v = getattr(resolved.container, resolved.field)
            if resolved.base_container is None:
                changed.append(p)
                continue
            if live_v != getattr(resolved.base_container, resolved.field):
                changed.append(p)
        return changed

    def _describe_section(self, name: str) -> str:
        """One compact line per section for the overview."""
        live = self._live
        if name == "llm":
            llm = live.llm
            text = f"llm: {llm.provider} · {llm.model}"
            if llm.vision_model:
                text += f" · vision {llm.vision_model}"
            if llm.salience_model:
                text += f" · salience {llm.salience_model}"
            text += f" · max {_plain_value(llm.max_tokens)} tokens"
            return text
        if name == "agent":
            a = live.agent
            return (
                f"agent: tick {_plain_value(a.tick_seconds)}s · up to {a.max_tool_iterations} "
                f"tool calls a turn · daily cap {a.daily_surface_cap} · quiet hours "
                f"{a.quiet_hours or 'off'} (urgent at {_plain_value(a.quiet_urgent_salience)})"
            )
        if name == "salience":
            s = live.salience
            return f"salience: notice at {_plain_value(s.threshold)} · at most {s.rate_cap_per_hour} an hour"
        if name == "messaging":
            m = live.messaging
            onoff = lambda t: "on" if t.enabled else "off"  # noqa: E731
            return f"messaging: telegram {onoff(m.telegram)} · discord {onoff(m.discord)} · slack {onoff(m.slack)}"
        if name == "contacts":
            c = live.contacts
            senders = (
                f" · {len(c.allowlist)} allowed senders"
                if c.allowlist
                else " · everyone can reach me"
            )
            return f"contacts: allowlist {c.mode}{senders}"
        if name == "authz":
            a = live.authz
            plural = "rule" if len(a.rules) == 1 else "rules"
            return f"authz: {len(a.rules)} user {plural} · approvals wait {_plain_value(a.approval_ttl_hours)}h"
        servers = live.mcp_servers
        if not servers:
            return "mcp_servers: none configured"
        return "mcp_servers: " + ", ".join(_plain_entry(s) for s in servers)

    def _section_show(self, name: str) -> str:
        lines = [f"{name} — config.yaml values with your chat changes marked:"]
        if name == "mcp_servers":
            if not self._live.mcp_servers:
                lines.append("  (no apps configured)")
            for s in self._live.mcp_servers:
                base_s = next((b for b in self._base.mcp_servers if b.name == s.name), None)
                marker = "" if base_s is not None and base_s == s else " (changed from config.yaml)"
                lines.append(f"  {_plain_entry(s)}{marker}")
            return "\n".join(lines)
        if name == "messaging":
            for platform in ("telegram", "discord", "slack"):
                live_v = getattr(self._live.messaging, platform).enabled
                base_v = getattr(self._base.messaging, platform).enabled
                marker = " (changed from config.yaml)" if live_v != base_v else ""
                lines.append(f"  {platform}.enabled = {_plain_value(live_v)}{marker}")
            return "\n".join(lines)
        section_live = getattr(self._live, name)
        section_base = getattr(self._base, name)
        for field in type(section_live).model_fields:
            live_v = getattr(section_live, field)
            base_v = getattr(section_base, field)
            if isinstance(live_v, list):
                lines.append(f"  {field}: {len(live_v)} entries")
                lines.extend(f"    {_plain_entry(e)}" for e in live_v)
            else:
                marker = " (changed from config.yaml)" if live_v != base_v else ""
                lines.append(f"  {field} = {_plain_value(live_v)}{marker}")
        return "\n".join(lines)

    def _one_show(self, resolved: _Resolved) -> str:
        live_v = getattr(resolved.container, resolved.field)
        if resolved.base_container is None:
            return f"{resolved.canonical} = {_plain_value(live_v)} (not in config.yaml — set from chat)"
        base_v = getattr(resolved.base_container, resolved.field)
        if resolved.kind == "list":
            lines = [f"{resolved.canonical}: {len(live_v)} entries (config.yaml has {len(base_v)})"]
            lines.extend(f"  {_plain_entry(e)}" for e in live_v)
            return "\n".join(lines)
        if live_v != base_v:
            return (
                f"{resolved.canonical} = {_plain_value(live_v)} "
                f"(changed from config.yaml, which says {_plain_value(base_v)})"
            )
        return f"{resolved.canonical} = {_plain_value(live_v)} (config.yaml's value)"
