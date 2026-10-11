"""Decision traces — the encrypted "why" behind every action.

The audit log is the *what*: append-only, hash-chained, parameters kept
as digests so the chain proves what was decided without storing what was
in the call. A trace is the *why*: what triggered the act, what the model
saw and said, how the authorization gate ruled each proposed call, and
what came back. Together they answer "why did you do that?" with a record
instead of a reconstruction.

One row per act, four kinds:
- turn      — one LLM turn: the trigger (messages, observations), every
              gated call with its ruling, the model's own words between
              calls, and its reply
- routine   — one routine fire: the taught trigger, the matching event,
              the call it ran
- scheduled — one scheduled action firing: the label, the call, the result
- carry_out — an approved call running: the approval, the result

Traces are encrypted at rest like all human-readable content, and they
are bookkeeping, not decisions: writing one must never break the act it
records, so the loop wraps every trace write in its own try/except.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg

from ..memory.crypto import Cipher

log = logging.getLogger("aether.agent.traces")

# what a trace row can be
TURN = "turn"
ROUTINE = "routine"
SCHEDULED = "scheduled"
CARRY_OUT = "carry_out"


def _aad(trace_id: int) -> str:
    return f"decision_traces:trace_enc:{trace_id}"


@dataclass
class Trace:
    id: int
    kind: str
    label: str
    payload: dict[str, Any]  # decrypted
    created_at: datetime


class Traces:
    """Database-backed store for decision traces."""

    def __init__(self, pool: asyncpg.Pool, cipher: Cipher) -> None:
        self._pool = pool
        self._cipher = cipher

    async def create(self, *, kind: str, label: str = "", payload: dict[str, Any]) -> Trace:
        """Persist one trace — the same two-step encrypt pattern as every
        store: insert a placeholder, then write ciphertext bound to the
        row id via AAD, in the same transaction."""
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "INSERT INTO decision_traces (kind, label, trace_enc)"
                " VALUES ($1, $2, $3) RETURNING id, created_at",
                kind,
                label,
                b"",  # placeholder until the id exists; replaced below, same transaction
            )
            trace_id = row["id"]
            await conn.execute(
                "UPDATE decision_traces SET trace_enc = $1 WHERE id = $2",
                self._cipher.encrypt_json(payload, aad=_aad(trace_id)),
                trace_id,
            )
        return Trace(
            id=trace_id,
            kind=kind,
            label=label,
            payload=dict(payload),
            created_at=row["created_at"],
        )

    async def get(self, trace_id: int) -> Trace | None:
        row = await self._pool.fetchrow(
            "SELECT id, kind, label, trace_enc, created_at FROM decision_traces WHERE id = $1",
            trace_id,
        )
        return self._to_trace(row) if row else None

    async def recent(self, limit: int = 20) -> list[Trace]:
        """Newest first — the explain tool's and the web panel's view."""
        rows = await self._pool.fetch(
            "SELECT id, kind, label, trace_enc, created_at FROM decision_traces"
            " ORDER BY id DESC LIMIT $1",
            limit,
        )
        return [self._to_trace(r) for r in rows]

    def _to_trace(self, row: asyncpg.Record) -> Trace:
        trace_id = row["id"]
        try:
            payload = self._cipher.decrypt_json(row["trace_enc"], aad=_aad(trace_id))
        except Exception:
            log.warning("trace %s undecryptable — treated as empty", trace_id)
            payload = {}
        return Trace(
            id=trace_id,
            kind=row["kind"],
            label=row["label"],
            payload=payload,
            created_at=row["created_at"],
        )
