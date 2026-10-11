"""Append-only, hash-chained audit log.

    entry_hash = sha256(prev_hash || canonical(entry))

Each row commits to the entire history before it, so any edit, deletion,
or reordering is detectable by re-walking the chain with verify_chain() —
the chain check is the tamper evidence, not the database's own grants.

Only non-sensitive metadata is stored (who, what tool, which rule, the
digest of the parameters) — never the parameters themselves.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import asyncpg

GENESIS_HASH = "0" * 64


def params_digest(params: Mapping[str, Any]) -> str:
    """sha256 of the canonical parameter JSON — identifies the call without
    storing what was in it."""
    blob = json.dumps(
        dict(params), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def canonical_entry(
    *,
    seq: int,
    actor: str,
    tool_name: str,
    decision: str,
    rules_matched: str,
    params_digest: str,
    outcome: str,
    created_at: str,
    prev_hash: str,
) -> str:
    """The exact byte string every entry hash commits to. Field order is
    pinned by sort_keys, so it never depends on how a caller built a dict."""
    payload = {
        "seq": seq,
        "actor": actor,
        "tool_name": tool_name,
        "decision": decision,
        "rules_matched": rules_matched,
        "params_digest": params_digest,
        "outcome": outcome,
        "created_at": created_at,
        "prev_hash": prev_hash,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_entry_hash(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _created_at_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)


@dataclass(frozen=True)
class ChainVerification:
    ok: bool
    entries: int  # entries examined
    first_bad_seq: int | None = None
    problem: str | None = None


def verify_entries(rows: Iterable[Mapping[str, Any]]) -> ChainVerification:
    """Walk rows (in seq order) and recompute the chain. Works on plain
    dicts, asyncpg Records — anything with key access."""
    prev = GENESIS_HASH
    count = 0
    for row in rows:
        count += 1
        seq = row["seq"]
        if row["prev_hash"] != prev:
            return ChainVerification(
                False,
                count,
                seq,
                f"seq {seq}: prev_hash does not chain to the previous entry",
            )
        expected = compute_entry_hash(
            canonical_entry(
                seq=seq,
                actor=row["actor"],
                tool_name=row["tool_name"],
                decision=row["decision"],
                rules_matched=row["rules_matched"],
                params_digest=row["params_digest"],
                outcome=row["outcome"],
                created_at=_created_at_iso(row["created_at"]),
                prev_hash=row["prev_hash"],
            )
        )
        if expected != row["entry_hash"]:
            return ChainVerification(
                False, count, seq, f"seq {seq}: stored hash does not match the entry contents"
            )
        prev = row["entry_hash"]
    return ChainVerification(True, count)


def _age(now: datetime, then: datetime) -> str:
    """Coarse elapsed words, "moments ago" to "N days ago" — the badge's
    'last act' is a feeling, not a timestamp."""
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)  # naive stamps are UTC here
    seconds = max(0.0, (now - then).total_seconds())
    if seconds < 60:
        return "moments ago"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)} minute{'s' if minutes >= 2 else ''} ago"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)} hour{'s' if hours >= 2 else ''} ago"
    days = hours / 24
    return f"{int(days)} day{'s' if days >= 2 else ''} ago"


def verification_text(verification: ChainVerification, newest_at: datetime | None = None) -> str:
    """A chain verification as the user reads it in chat — the same facts
    the panel badge shows, in plain words. Pure: the caller reads the rows."""
    if verification.entries == 0:
        return "decision record: empty — nothing recorded yet."
    if verification.ok:
        last = f", last act {_age(datetime.now(UTC), newest_at)}" if newest_at is not None else ""
        plural = "s" if verification.entries != 1 else ""
        return (
            f"decision record: {verification.entries:,} decision{plural}, "
            f"chain intact — every entry still hashes to the one before it{last}."
        )
    where = f" at entry #{verification.first_bad_seq}" if verification.first_bad_seq else ""
    problem = f" — {verification.problem}" if verification.problem else ""
    return f"decision record: BROKEN{where}{problem}. Everything from there on can't be trusted."


class AuditLog:
    """Database-backed chain writer/verifier. Appends are serialized with a
    process-local lock so concurrent callers can't fork the chain."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._lock = asyncio.Lock()

    async def append(
        self,
        *,
        actor: str,
        tool_name: str,
        decision: str,
        rules_matched: str = "",
        params: Mapping[str, Any] | None = None,
        outcome: str = "",
    ) -> int:
        created_at = datetime.now(UTC)
        digest = params_digest(params or {})
        async with self._lock, self._pool.acquire() as conn, conn.transaction():
            seq = await conn.fetchval("SELECT COALESCE(MAX(seq), 0) + 1 FROM audit_log")
            prev_hash = (
                await conn.fetchval("SELECT entry_hash FROM audit_log ORDER BY seq DESC LIMIT 1")
                or GENESIS_HASH
            )
            entry_hash = compute_entry_hash(
                canonical_entry(
                    seq=seq,
                    actor=actor,
                    tool_name=tool_name,
                    decision=decision,
                    rules_matched=rules_matched,
                    params_digest=digest,
                    outcome=outcome,
                    created_at=created_at.isoformat(),
                    prev_hash=prev_hash,
                )
            )
            await conn.execute(
                "INSERT INTO audit_log (seq, actor, tool_name, decision,"
                " rules_matched, params_digest, outcome, prev_hash,"
                " entry_hash, created_at)"
                " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
                seq,
                actor,
                tool_name,
                decision,
                rules_matched,
                digest,
                outcome,
                prev_hash,
                entry_hash,
                created_at,
            )
        return seq

    async def verify_chain(self) -> ChainVerification:
        rows = await self._pool.fetch(
            "SELECT seq, actor, tool_name, decision, rules_matched, params_digest,"
            " outcome, prev_hash, entry_hash, created_at FROM audit_log ORDER BY seq"
        )
        return verify_entries(rows)

    async def recent(self, limit: int = 100) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            "SELECT seq, actor, tool_name, decision, rules_matched, params_digest,"
            " outcome, entry_hash, created_at FROM audit_log ORDER BY seq DESC LIMIT $1",
            limit,
        )
