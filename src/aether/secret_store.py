"""Keys pasted in chat, kept encrypted — the .env-free half of /apps.

A key pasted in an /apps answer is consumed before ingest (the walk
answers in the deterministic command path — no model, no memory, no
trace), never echoed back (replies name the NAME only), and lands here
with the same AES-256-GCM as all other content, bound to its row through
the cipher's AAD. The resolver reads this store before .env, so a paste
is the most recent deliberate act and wins over an older line — and .env
keeps working for every name that never gets pasted.

The trade-off is stated plainly: the value transits the chat platform
and sits in its history until the user deletes the message. That is the
price of never editing a file on a server; docs advise deleting the sent
message right after the "stored." reply.
"""

from __future__ import annotations

from typing import Any

from .memory.crypto import Cipher


class SecretStore:
    """App keys in Postgres, encrypted at rest — the OAuthTokenStore shape
    (one-step upsert, name-bound AAD), name-keyed."""

    def __init__(self, pool: Any, cipher: Cipher) -> None:
        self._pool = pool
        self._cipher = cipher

    async def set(self, name: str, value: str) -> None:
        await self._pool.execute(
            "INSERT INTO app_secrets (name, value_enc, updated_at)"
            " VALUES ($1, $2, now())"
            " ON CONFLICT (name) DO UPDATE SET"
            " value_enc = EXCLUDED.value_enc, updated_at = now()",
            name,
            self._cipher.encrypt_text(value, f"app_secrets:value:{name}"),
        )

    async def get(self, name: str) -> str | None:
        row = await self._pool.fetchrow("SELECT value_enc FROM app_secrets WHERE name = $1", name)
        if row is None:
            return None
        return self._cipher.decrypt_text(bytes(row["value_enc"]), f"app_secrets:value:{name}")

    async def names(self) -> set[str]:
        """Which key names exist right now — names only, never values."""
        rows = await self._pool.fetch("SELECT name FROM app_secrets")
        return {row["name"] for row in rows}
