"""The app catalog: the chat surfaces ship as recipes — links, the key
asks, and the toggle, all plain data.

A recipe is everything the /apps walk needs to connect a chat surface
without the user going looking for any of it: where to click (exact
links, numbered steps), what to paste (one ask per key, by name only —
values are handed over in chat and consumed before ingest), and the
messaging toggle to write. External apps (gmail, github, notion, …) are
NOT here: they connect through Composio, whose own toolkit catalog is
fetched live by the bridge — Aether keeps no per-app list of someone
else's ecosystem. Anything that speaks MCP directly gets the generic
walk ("something else"), whose server fragments reference keys by name
(`$NAME`), expanded at open time by the resolver, so the pinned config
copy never carries a secret value either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class AppRecipe:
    key: str  # the walk's token for this app (menu order)
    name: str  # how it reads in chat
    blurb: str  # what it gives the user
    kind: Literal["mcp", "messaging"]
    env_names: tuple[str, ...] = ()  # the key names the recipe needs
    steps: tuple[str, ...] = ()  # numbered plain-words setup, links first
    keys_line: str = ""  # the lead-in above the steps
    asks: tuple[str, ...] = ()  # one paste ask per key, in env_names order
    stake: str = ""  # "your code" — what connecting it touches
    apply_path: str = ""  # messaging: the toggle this recipe turns on
    server: dict[str, Any] = field(default_factory=dict)  # mcp: the add value, $NAME refs intact

    def ask_for(self, name: str) -> str:
        """The chat line that asks for one key's value — the recipe's own
        wording when it has it, the name-keyed default otherwise."""
        if name in self.env_names:
            idx = self.env_names.index(name)
            if idx < len(self.asks):
                return self.asks[idx]
        return f"paste the {name} here — I'll take it from there."


CATALOG: tuple[AppRecipe, ...] = (
    AppRecipe(
        key="telegram",
        name="telegram",
        blurb="talk to me there",
        kind="messaging",
        apply_path="messaging.telegram.enabled",
        env_names=("TELEGRAM_BOT_TOKEN",),
        keys_line="one token needed:",
        stake="your chats there",
        steps=("open https://t.me/BotFather and send it /newbot — it hands you a token",),
        asks=("paste the token here — I'll take it from there.",),
    ),
    AppRecipe(
        key="discord",
        name="discord",
        blurb="talk to me there",
        kind="messaging",
        apply_path="messaging.discord.enabled",
        env_names=("DISCORD_BOT_TOKEN",),
        keys_line="one token needed:",
        stake="your chats there",
        steps=(
            "open https://discord.com/developers/applications and create an "
            "application — under Bot, Reset Token copies its token",
        ),
        asks=("paste the token here — I'll take it from there.",),
    ),
    AppRecipe(
        key="slack",
        name="slack",
        blurb="talk to me there",
        kind="messaging",
        apply_path="messaging.slack.enabled",
        env_names=("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"),
        keys_line="two tokens needed:",
        stake="your workspace",
        steps=(
            "open https://api.slack.com/apps and create an app — under OAuth & "
            "Permissions, Install to Workspace copies the bot token (xoxb…)",
            "under Basic Info, App-Level Tokens generates the app token "
            "(xapp…) with the connections:write scope",
        ),
        asks=(
            "paste the bot token (xoxb…) here — I'll take it from there.",
            "now the app token (xapp…) — paste it here.",
        ),
    ),
)


def recipe_for(key: str) -> AppRecipe | None:
    """The catalog recipe by its walk token, or None (the generic walk's)."""
    return next((r for r in CATALOG if r.key == key), None)
