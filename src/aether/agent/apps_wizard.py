"""The guided /apps walk: connect an app without leaving chat.

Bare /apps is just the status — what's connected and what the gate does
with it. /apps add <name> is the front door: a chat surface (telegram,
discord, slack) hands over its recipe — the exact link, the paste ask —
and every other name is looked up in the app hub's live catalog, exactly
or fuzzily (a fuzzy hit asks which one, numbered). The hub's connect is
a hosted link: Aether asks for the project's COMPOSIO_API_KEY once (the
paste stage, consumed before ingest, encrypted, never echoed), then
serves the app's Connect Link and waits out of band. When the click
lands, the walk's last question is the gate's: what may Aether do with
this app — read freely (the default), everything asks first, or act
freely — and the stricter answers write an authz rule through the same
applier, which parks it for the one-tap because authz is a security
root even from a walk. Nothing matched? The generic MCP walk asks its
standardized questions.

The wizard stays a pure state machine in the ConfigWizard's discipline:
no I/O of its own, nothing ingested, memory-only state, a 15-minute idle
timeout, cancel words from any stage, and a menu answer that's a word or
two at most (anything longer is a changed subject and the walk steps
aside — except a pasted key at `paste`, which is exactly the one long
answer that's wanted). The Connect Link pause (`link_wait`) completes
out of band: the loop's background waiter announces it and moves the
walk to `permissions` itself.
"""

from __future__ import annotations

import re
import shlex
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..apps_catalog import AppRecipe, recipe_for

# the loop's shared applier: (op, path[, value][, origin]) -> the reply to relay
ApplyFn = Callable[..., Awaitable[str]]
# which secret names exist right now — names only, never values
EnvNamesFn = Callable[[], Awaitable[set[str]]]
# the /apps status render, loop-side (config + live registry + the hub)
StatusFn = Callable[[], Awaitable[str]]
# key name, value, app name -> was it stored? False answers honestly below
SaveSecretFn = Callable[[str, str, str], Awaitable[bool]]
# a name the user typed -> the hub catalog's closest (slug, name) pairs
FindToolkitsFn = Callable[[str], Awaitable[list[tuple[str, str]]]]
# toolkit slug -> (request id, connect link); None when the hub said no
AuthorizeFn = Callable[[str], Awaitable[tuple[str, str] | None]]

_CANCEL_WORDS = frozenset({"stop", "cancel", "done", "quit", "exit", "nevermind", "never mind"})
_IDLE_SECONDS = 15 * 60  # a walk nobody answers stops intercepting messages
_MAX_MENU_TOKEN = 24  # longer than this at a menu is a changed subject
_STRIP_TAIL = "\"'.),!;"
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")

_GOODBYE = "ok — anytime. /apps to start again."
_NOT_A_VALUE = "just the value itself — paste only the value"
_MENU_NUMBER = "that's a menu number — paste the key's value itself"
_LINK_WAITING = "still waiting — open the link I sent and approve it there; say stop to give up."
_ADD_HINT = "add one with /apps add <name> — like /apps add gmail"


def _slugify(name: str) -> str:
    """A spoken app name as a server-name-safe token ("google calendar" ->
    "google-calendar") — the generic walk builds config with it."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "app"


def _hub_key_recipe() -> AppRecipe:
    """The one-time COMPOSIO_API_KEY ask as a recipe, so the ordinary paste
    stage drives it — consumed before ingest, encrypted, named never
    echoed, exactly like any other key."""
    return AppRecipe(
        key="composio",
        name="the app hub",
        blurb="one key connects every app in the catalog",
        kind="mcp",
        env_names=("COMPOSIO_API_KEY",),
        keys_line="one key needed (one time — it connects every app):",
        steps=(
            "open https://platform.composio.dev and sign in — under "
            "Settings → API Keys, copy your project key",
        ),
        asks=("paste the key here — I'll take it from there.",),
    )


class AppsWizard:
    """One guided walk, in memory. `handle` returns the walk's reply, or
    None when the message wasn't part of it — the caller drops the walk and
    lets the message be normal chat. `alive` says whether a next answer has
    anywhere to land."""

    def __init__(
        self,
        *,
        apply: ApplyFn,
        status: StatusFn,
        env_names: EnvNamesFn,
        save_secret: SaveSecretFn,
        find_toolkits: FindToolkitsFn,
        authorize: AuthorizeFn,
    ) -> None:
        self._apply = apply
        self._status = status
        self._env_names = env_names
        self._save_secret = save_secret
        self._find_toolkits = find_toolkits
        self._authorize = authorize
        # the loop's handoff: (toolkit slug, request id) once a link is
        # served — read once to spawn the background waiter, then cleared
        self.pending_connect: tuple[str, str] | None = None
        self._reset()

    def _reset(self) -> None:
        self._stage = "dead"
        self._at = time.monotonic()
        self._recipe: AppRecipe | None = None
        # the keys this walk is still waiting to receive, in paste order
        self._paste_names: list[str] = []
        # the hub app being connected (slug, display name), when one is
        self._pending_toolkit: tuple[str, str] | None = None
        # a name asked for before the hub's key existed — resolved after
        self._wanted_name: str | None = None
        # fuzzy matches the toolkit_pick menu is answering
        self._toolkit_matches: list[tuple[str, str]] = []
        # the generic walk's answers, one at a time
        self._generic_name = ""
        self._generic_kind = ""
        self._generic_server: dict[str, Any] = {}
        self.pending_connect = None

    @property
    def alive(self) -> bool:
        """False once the walk is over — goodbye, changed subject, stale, or
        done (an applied change ends at its reply; nothing is waiting)."""
        return self._stage != "dead"

    async def start(self) -> str:
        """Bare /apps: what's connected, honestly, and how to add one. No
        question is outstanding — the walk ends with the answer."""
        self._reset()
        return f"{await self._status()}\n{_ADD_HINT}"

    async def start_add(self, name: str) -> str:
        """/apps add <name> — resolve the name and walk the connect."""
        self._reset()
        name = name.strip().strip("\"'").lower()
        if not name:
            self._stage = "app_query"
            return "which app? say the name — like gmail"
        return await self._resolve_name(name)

    async def handle(self, text: str, now: float | None = None) -> str | None:
        """One inbound message, answered or not. A str is the walk's reply;
        None means the message should fall through to normal chat."""
        stamp = time.monotonic() if now is None else now
        if stamp - self._at > _IDLE_SECONDS:
            self._stage = "dead"  # stale: gone quietly, never intercept this one
            return None
        self._at = stamp
        text = text.strip()
        lowered = text.lower()
        # "done" is the paste stage's .env-fallback check; everywhere else
        # it's a way out like the rest of the cancel words
        done_answers = lowered == "done" and self._stage == "paste"
        if lowered in _CANCEL_WORDS and not done_answers:
            self._stage = "dead"
            return _GOODBYE
        stage = self._stage
        if stage == "app_query":
            return await self._answer_app_query(text)
        if stage == "toolkit_pick":
            return await self._answer_toolkit_pick(text)
        if stage == "generic_offer":
            return await self._answer_generic_offer(text)
        if stage == "paste":
            return await self._answer_paste(text)
        if stage == "link_wait":
            return self._answer_link_wait(text)
        if stage == "permissions":
            return await self._answer_permissions(text)
        if stage == "generic_kind":
            return await self._answer_generic_kind(text)
        if stage == "generic_target":
            return self._answer_generic_target(text)
        if stage == "generic_env":
            return await self._answer_generic_env(text)
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

    # -- the stages ---------------------------------------------------------------

    async def _resolve_name(self, name: str) -> str:
        """Chat-surface recipe first (those stay native), then the hub. No
        hub key yet? The key paste comes first — the name resolves for real
        once the catalog can answer."""
        recipe = recipe_for(name)
        if recipe is not None:
            self._recipe = recipe
            return await self._serve_recipe(recipe)
        self._generic_name = _slugify(name)  # a later "by hand" keeps the name
        if "COMPOSIO_API_KEY" not in await self._env_names():
            self._wanted_name = name
            self._recipe = _hub_key_recipe()
            self._paste_names = ["COMPOSIO_API_KEY"]
            self._stage = "paste"
            return self._recipe_text(self._recipe, "COMPOSIO_API_KEY")
        return await self._resolve_in_hub(name)

    async def _resolve_in_hub(self, name: str) -> str:
        """The hub's live catalog: an exact hit connects, a fuzzy handful
        asks which one, nothing matched gets the generic walk offered."""
        matches = await self._find_toolkits(name)
        if len(matches) == 1:
            return await self._begin_connect(*matches[0])
        if matches:
            self._toolkit_matches = matches
            self._stage = "toolkit_pick"
            lines = [f"which {name}?"]
            for i, (_slug, title) in enumerate(matches, 1):
                lines.append(f"{i} — {title}")
            lines.append(f"{len(matches) + 1} — none of these")
            return "\n".join(lines)
        return self._offer_generic(
            f"I don't know an app called {name} — the hub's catalog has "
            "nothing by that name either."
        )

    def _offer_generic(self, why: str) -> str:
        self._stage = "generic_offer"
        return f"{why} Set it up by hand over MCP?\n1 — yes, walk me through it\n2 — no"

    async def _answer_app_query(self, text: str) -> str | None:
        name = text.strip().strip("\"'").lower()
        if not name or len(name) > 40 or "\n" in name:
            self._stage = "dead"  # a sentence isn't an app name — let it be chat
            return None
        return await self._resolve_name(name)

    async def _answer_toolkit_pick(self, text: str) -> str | None:
        span = len(self._toolkit_matches) + 1
        pick = self._menu_pick(text, span)
        if pick is None:
            if self._changed_subject(text):
                self._stage = "dead"
                return None
            return self._nudge(span)
        if pick == span:  # none of these — by hand over MCP instead
            return self._offer_generic("fair enough.")
        slug, name = self._toolkit_matches[pick - 1]
        return await self._begin_connect(slug, name)

    async def _answer_generic_offer(self, text: str) -> str | None:
        pick = self._menu_pick(text, 2)
        if pick is None:
            word = text.strip().lower()
            if word in ("yes", "y", "sure", "ok"):
                pick = 1
            elif word in ("no", "n", "nope"):
                pick = 2
            else:
                if self._changed_subject(text):
                    self._stage = "dead"
                    return None
                return self._nudge(2)
        if pick == 2:
            self._stage = "dead"
            return _GOODBYE
        self._stage = "generic_kind"
        name = self._generic_name
        return (
            f"how does {name} connect?\n1 — stdio — a command run locally\n2 — http — a URL I call"
        )

    # -- the hub connect: key once, link per app, permissions last --------------------

    async def _begin_connect(self, slug: str, name: str) -> str:
        """One hub app picked. The project key is asked once ever (a paste
        like any other); the per-app part is just the hosted link."""
        self._pending_toolkit = (slug, name)
        if "COMPOSIO_API_KEY" not in await self._env_names():
            self._recipe = _hub_key_recipe()
            self._paste_names = ["COMPOSIO_API_KEY"]
            self._stage = "paste"
            return self._recipe_text(self._recipe, "COMPOSIO_API_KEY")
        return await self._serve_link()

    async def _serve_link(self) -> str:
        """The Connect Link is the whole sign-in: credentials pass between
        the user and the hub only. The loop picks up `pending_connect` and
        waits out of band."""
        pending = self._pending_toolkit
        assert pending is not None
        slug, name = pending
        result = await self._authorize(slug)
        if result is None:
            self._stage = "dead"
            return (
                f"couldn't start the {name} connection — the app hub didn't "
                "answer. nothing was connected; try again in a bit."
            )
        request_id, url = result
        self.pending_connect = (slug, request_id)
        self._stage = "link_wait"
        return (
            f"{name} — last step, let me in:\n"
            f"open this and approve: {url}\n"
            "(I'll take it from there)"
        )

    def _answer_link_wait(self, text: str) -> str | None:
        if self._changed_subject(text):
            self._stage = "dead"
            return None
        return _LINK_WAITING

    def enter_permissions(self, slug: str, identity: str) -> str:
        """The background connect finished — the loop moves the walk (this
        one or a fresh one) to the gate's question: what may Aether do with
        the new app."""
        self._reset()
        self._pending_toolkit = (slug, slug)
        self._stage = "permissions"
        who = f" — {identity}" if identity else ""
        return (
            f"{slug} connected{who}.\n"
            f"what may I do with {slug}?\n"
            "1 — read freely, ask before acting (the default)\n"
            "2 — everything asks first\n"
            "3 — act freely"
        )

    async def _answer_permissions(self, text: str) -> str | None:
        pending = self._pending_toolkit
        if pending is None:
            self._stage = "dead"
            return None
        slug, _name = pending
        pick = self._menu_pick(text, 3)
        if pick is None:
            word = text.strip().lower()
            if word in ("read", "reading", "default"):
                pick = 1
            elif word in ("ask", "everything"):
                pick = 2
            elif word in ("free", "freely", "act"):
                pick = 3
            else:
                if self._changed_subject(text):
                    self._stage = "dead"
                    return None
                return self._nudge(3)
        self._stage = "dead"
        if pick == 1:
            return (
                f"that's the default — {slug} reads run free; sends, "
                "deletes and anything new will ask first."
            )
        # the gate's own vocabulary: one authz rule over the app's tools.
        # authz is a security root — it parks for the one-tap even from a
        # walk, and the applier's own words relay that honestly
        decision = "approve" if pick == 2 else "allow"
        note = (
            f"{slug} connected in chat — everything asks first"
            if pick == 2
            else f"{slug} connected in chat — freed to act"
        )
        return await self._apply(
            "add",
            "authz.rules",
            {
                "tool_pattern": f"composio__{slug.upper()}_",
                "decision": decision,
                "note": note,
            },
            origin="walk",
        )

    async def _answer_paste(self, text: str) -> str | None:
        """One key's value, straight to the encrypted store. The value dies
        with this call — never kept, never echoed, never ingested; the
        reply names only what was stored."""
        recipe = self._recipe
        assert recipe is not None
        if text.lower() == "done":
            # the .env fallback: nothing pasted, but maybe the user added
            # the line by hand — check honestly and keep waiting if not
            missing = await self._missing_names(recipe)
            self._paste_names = missing
            if missing:
                names = " and ".join(missing)
                word = "it" if len(missing) == 1 else "them"
                verb = "isn't" if len(missing) == 1 else "aren't"
                return (
                    f"not yet — {names} {verb} set. paste {word} here, or "
                    'add to .env and reply "done".'
                )
            head = f"the key's there — connecting {recipe.name} now."
            return await self._finish_keys(recipe, head)
        if " " in text or "\n" in text:
            # a paste is the one long answer that's wanted — a space in it
            # means a mis-paste, not a changed subject. The way out still
            # works from here (cancel words answer above), so the re-ask
            # never traps anyone.
            return _NOT_A_VALUE
        if text.isdigit() and len(text) <= 6:
            return _MENU_NUMBER
        name = self._paste_names[0]
        value = text.strip("\"'`")
        if not value:
            return _NOT_A_VALUE
        stored = await self._save_secret(name, value, recipe.name)
        if not stored:
            return (
                "that didn't work — nothing was stored. add "
                f'{name} to .env and reply "done", or paste it again.'
            )
        self._paste_names.pop(0)
        if self._paste_names:
            return f"stored. {recipe.ask_for(self._paste_names[0])}"
        return await self._finish_keys(recipe)

    async def _missing_names(self, recipe: AppRecipe) -> list[str]:
        known = await self._env_names()
        return [n for n in recipe.env_names if n not in known]

    async def _serve_recipe(self, recipe: AppRecipe) -> str:
        # every path that serves a recipe stores it — the catalog picks and
        # the generic walk's built one alike — so the paste stage always
        # has a recipe
        self._recipe = recipe
        missing = await self._missing_names(recipe)
        if not missing:
            found = (
                "the keys are already there"
                if len(recipe.env_names) > 1
                else "the key's already there"
            )
            head = f"{found} — connecting {recipe.name} now."
            return await self._apply_recipe(recipe, head=head)
        self._paste_names = missing
        self._stage = "paste"
        return self._recipe_text(recipe, missing[0])

    def _recipe_text(self, recipe: AppRecipe, first_missing: str) -> str:
        """The recipe in chat: what it gives, where to click, and the ask
        for the first key still missing — links always, values never."""
        ask = recipe.ask_for(first_missing)
        if not recipe.steps:
            # a generic walk's key ask — the recipe is just the key's name
            return f"{recipe.name} needs a key.\n{ask}"
        lines = [f"{recipe.name} — {recipe.blurb}."]
        if recipe.keys_line:
            lines.append(recipe.keys_line)
        for i, step in enumerate(recipe.steps, 1):
            lines.append(f"{i} — {step}")
        lines.append(ask)
        return "\n".join(lines)

    async def _finish_keys(self, recipe: AppRecipe, head: str = "") -> str:
        """The last key landed. A hub key continues the connect it was
        asked for — resolve the wanted name now the catalog answers, or
        serve the picked app's link. Anything else connects now — the
        pick-and-paste was the approval."""
        if self._pending_toolkit is not None:
            reply = await self._serve_link()
            return f"{head}\n{reply}" if head else reply
        if self._wanted_name is not None:
            name, self._wanted_name = self._wanted_name, None
            reply = await self._resolve_in_hub(name)
            return f"{head}\n{reply}" if head else reply
        if not head:
            if recipe.kind == "messaging":
                head = f"stored — switching {recipe.name} on now."
            else:
                head = f"stored — connecting {recipe.name} now."
        return await self._apply_recipe(recipe, head=head)

    async def _apply_recipe(self, recipe: AppRecipe, head: str) -> str:
        """Hand the change to the shared applier and relay its reply. The
        pick-and-paste was the approval, so a walk-driven connect applies
        directly through the executor; a user rule that forces a park
        anyway relays verbatim. The walk ends here: the honest verdict is
        the applier's own words."""
        if recipe.kind == "messaging":
            reply = await self._apply("set", recipe.apply_path, True, origin="walk")
        else:
            reply = await self._apply("add", "mcp_servers", recipe.server, origin="walk")
        self._stage = "dead"
        if head:
            return f"{head}\n{reply}"
        return reply

    # -- the generic walk ------------------------------------------------------------

    async def _answer_generic_kind(self, text: str) -> str | None:
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
        self._generic_kind = kind
        self._stage = "generic_target"
        if kind == "stdio":
            return "what's the command? (e.g. npx -y mcp-weather)"
        return "what's the URL?"

    def _answer_generic_target(self, text: str) -> str | None:
        name = self._generic_name
        if self._generic_kind == "stdio":
            try:
                parts = shlex.split(text)
            except ValueError:
                return "that command didn't parse — check the quotes and try again"
            if not parts:
                return "what's the command? (e.g. npx -y mcp-weather)"
            self._generic_server = {
                "name": name,
                "transport": {"type": "stdio", "command": parts[0], "args": parts[1:]},
            }
        else:
            url = text.strip()
            if not url or " " in url:
                return "just the URL — like https://mcp.example.com/mcp"
            self._generic_server = {"name": name, "transport": {"type": "http", "url": url}}
        self._stage = "generic_env"
        return (
            f"does {name} need a key or token? reply with the key's name "
            '(like WEATHER_API_KEY), or "none"'
        )

    async def _answer_generic_env(self, text: str) -> str | None:
        token = text.strip()
        if token.lower() in ("none", "no", "n"):
            return await self._apply_recipe(self._generic_recipe(()), head="")
        if _ENV_NAME.fullmatch(token):
            return await self._serve_recipe(self._generic_recipe((token,)))
        if self._changed_subject(text):
            self._stage = "dead"
            return None
        return 'just the key\'s name (like WEATHER_API_KEY) — or "none"'

    def _generic_recipe(self, env_names: tuple[str, ...]) -> AppRecipe:
        """The generic walk's answers as one recipe — the written config
        references the key by name only, exactly like a catalog one."""
        server = dict(self._generic_server)
        transport = dict(server["transport"])
        if env_names and transport["type"] == "stdio":
            transport["env"] = {env_names[0]: f"${env_names[0]}"}
        elif env_names:
            transport["headers"] = {"Authorization": f"Bearer ${env_names[0]}"}
        server["transport"] = transport
        return AppRecipe(
            key="generic",
            name=self._generic_name,
            blurb="",
            kind="mcp",
            env_names=env_names,
            keys_line="one key needed:",
            server=server,
        )
