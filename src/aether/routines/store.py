"""Routines — standing event triggers the user teaches once ("when Alice
emails me, note it under billing") and Aether evaluates on every new event.

Matching is deterministic code, never an LLM judgment: a trigger is a small
structured spec — `source`, `kind`, `from`, `contains`, AND-combined, each
optional — so whether a routine fired is explainable from the stored row
alone. The *action* is a plain tool call, and it passes the same
authorization gate at fire time as anything the model proposes: teaching a
routine is never a way around review, only a way to stop re-teaching.

Both the trigger and the action are encrypted at rest like every other
human-readable content. Every fire is recorded in the hash-chained audit
log with its provenance (`rules_matched` = "routine:<id>"), so "why did you
do that?" always has a row to point at.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg

from ..authz.audit import AuditLog
from ..memory.crypto import Cipher
from ..memory.entities import normalize_handle
from ..memory.events import SENDER_KEY, Event, _bare, event_text

log = logging.getLogger("aether.routines")


def _aad(routine_id: int, column: str) -> str:
    return f"routines:{column}:{routine_id}"


def sender_matches(pattern: str, event: Event) -> bool:
    """A trigger's `from` condition — the same two-way comparison the
    contact allowlist uses (normalized form, then bare handle), so a
    routine taught with "alice@example.com" matches mail from her on any
    provider, and "@alice" matches her on any handle-based platform."""
    sender = event.payload.get(SENDER_KEY)
    if not isinstance(sender, dict):
        return False  # no sender: poll results, captures, internal observations
    platform = str(sender.get("platform", ""))
    handle = str(sender.get("handle", ""))
    if normalize_handle(platform, handle) == normalize_handle("*", pattern):
        return True
    return bool(_bare(handle)) and _bare(pattern) == _bare(handle)


def trigger_matches(trigger: dict[str, Any], event: Event) -> bool:
    """Every condition present in the trigger must hold (AND). Missing or
    "*" conditions match anything; an empty trigger matches everything
    (create_routine refuses to arm one, but matching stays total)."""
    if not isinstance(trigger, dict):
        return False
    source = str(trigger.get("source", "")).strip()
    if source and source != "*" and source != event.source:
        return False
    kind = str(trigger.get("kind", "")).strip()
    if kind and kind != "*" and kind != event.kind:
        return False
    sender = str(trigger.get("from", "")).strip()
    if sender and sender != "*" and not sender_matches(sender, event):
        return False
    contains = str(trigger.get("contains", "")).strip()
    return not contains or contains.lower() in event_text(event).lower()


@dataclass
class Routine:
    id: int
    label: str
    trigger: dict[str, Any]  # decrypted — {source?, kind?, from?, contains?}
    action: dict[str, Any]  # decrypted — {"type": "tool", "tool": ..., "params": ...}
    enabled: bool
    cooldown_seconds: int
    fire_count: int
    last_fired_at: datetime | None
    created_at: datetime


class Routines:
    """Database-backed store for standing triggers."""

    def __init__(self, pool: asyncpg.Pool, cipher: Cipher, audit: AuditLog) -> None:
        self._pool = pool
        self._cipher = cipher
        self._audit = audit

    async def create(
        self,
        *,
        label: str,
        trigger: dict[str, Any],
        action: dict[str, Any],
        cooldown_seconds: int = 300,
        actor: str = "agent",
    ) -> Routine:
        """Persist one routine — it exists (and can fire) the moment this
        returns. Arming a standing instruction is audited, like scheduling
        one: the row itself deserves a trail entry."""
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "INSERT INTO routines (label, trigger_enc, action_enc,"
                " cooldown_seconds) VALUES ($1, $2, $3, $4)"
                " RETURNING id, created_at",
                label,
                b"",  # placeholders until the id exists; replaced below, same transaction
                b"",
                cooldown_seconds,
            )
            routine_id = row["id"]
            await conn.execute(
                "UPDATE routines SET trigger_enc = $1, action_enc = $2 WHERE id = $3",
                self._cipher.encrypt_json(trigger, aad=_aad(routine_id, "trigger_enc")),
                self._cipher.encrypt_json(action, aad=_aad(routine_id, "action_enc")),
                routine_id,
            )
        await self._audit.append(
            actor=actor,
            tool_name="create_routine",
            decision="info",
            params={
                "label": label,
                "trigger": dict(trigger),
                "tool": str(action.get("tool", "")),
            },
            outcome="routine armed",
        )
        return Routine(
            id=routine_id,
            label=label,
            trigger=dict(trigger),
            action=dict(action),
            enabled=True,
            cooldown_seconds=cooldown_seconds,
            fire_count=0,
            last_fired_at=None,
            created_at=row["created_at"],
        )

    async def get(self, routine_id: int) -> Routine | None:
        row = await self._pool.fetchrow(
            "SELECT id, label, trigger_enc, action_enc, enabled, cooldown_seconds,"
            " fire_count, last_fired_at, created_at FROM routines WHERE id = $1",
            routine_id,
        )
        return self._to_routine(row) if row else None

    async def list(self, limit: int = 50) -> list[Routine]:
        """Newest first, any status — the listing tool's view."""
        rows = await self._pool.fetch(
            "SELECT id, label, trigger_enc, action_enc, enabled, cooldown_seconds,"
            " fire_count, last_fired_at, created_at"
            " FROM routines ORDER BY id DESC LIMIT $1",
            limit,
        )
        return [self._to_routine(r) for r in rows]

    async def list_enabled(self) -> list[Routine]:
        """What the loop evaluates, oldest first so earlier-taught routines
        fire in the order the user taught them."""
        rows = await self._pool.fetch(
            "SELECT id, label, trigger_enc, action_enc, enabled, cooldown_seconds,"
            " fire_count, last_fired_at, created_at"
            " FROM routines WHERE enabled ORDER BY id"
        )
        return [self._to_routine(r) for r in rows]

    async def mark_fired(self, routine_id: int) -> None:
        """Count the fire and start the cooldown window."""
        await self._pool.execute(
            "UPDATE routines SET fire_count = fire_count + 1, last_fired_at = now() WHERE id = $1",
            routine_id,
        )

    async def set_enabled(self, routine_id: int, enabled: bool) -> Routine | None:
        row = await self._pool.fetchrow(
            "UPDATE routines SET enabled = $1 WHERE id = $2"
            " RETURNING id, label, trigger_enc, action_enc, enabled,"
            " cooldown_seconds, fire_count, last_fired_at, created_at",
            enabled,
            routine_id,
        )
        return self._to_routine(row) if row else None

    async def delete(self, routine_id: int) -> bool:
        row = await self._pool.fetchrow(
            "DELETE FROM routines WHERE id = $1 RETURNING id", routine_id
        )
        return row is not None

    def _to_routine(self, row: asyncpg.Record) -> Routine:
        routine_id = row["id"]
        try:
            trigger = self._cipher.decrypt_json(
                row["trigger_enc"], aad=_aad(routine_id, "trigger_enc")
            )
        except Exception:
            log.warning("routine %s trigger undecryptable — treated as empty", routine_id)
            trigger = {}
        try:
            action = self._cipher.decrypt_json(
                row["action_enc"], aad=_aad(routine_id, "action_enc")
            )
        except Exception:
            log.warning("routine %s action undecryptable — treated as empty", routine_id)
            action = {}
        return Routine(
            id=routine_id,
            label=row["label"],
            trigger=trigger,
            action=action,
            enabled=row["enabled"],
            cooldown_seconds=row["cooldown_seconds"],
            fire_count=row["fire_count"],
            last_fired_at=row["last_fired_at"],
            created_at=row["created_at"],
        )
