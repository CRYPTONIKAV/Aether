"""Event ingestion: dedup, the contact allowlist, and encrypted storage.

ingest() is the single enforcement point for the contact allowlist — every
channel (native connectors, MCP-derived senders, chat surfaces) funnels
here, so a non-allowlisted sender's content is dropped before it ever
touches storage. The only trace is an audit entry saying a drop happened;
the content itself is never written anywhere.

Dedup is structural: content_hash over the canonical payload, with
UNIQUE(source, content_hash) in the database — the same thing arriving
twice is stored once, without a lock or a race.

When a sender is provided it is merged into the encrypted payload under
"_sender" (so the salience judge and the context builder can see who it
came from, and the DB never stores the handle in plaintext).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import asyncpg

from ..authz.audit import AuditLog
from ..config import ContactsConfig
from .crypto import Cipher
from .entities import Sender, normalize_handle

if TYPE_CHECKING:
    from .vectors import VectorMemory

log = logging.getLogger("aether.memory.events")

SENDER_KEY = "_sender"


def content_hash(kind: str, payload: dict[str, Any]) -> str:
    blob = json.dumps(
        {"kind": kind, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _bare(handle: str) -> str:
    h = (handle or "").strip().lower()
    if h.startswith("mailto:"):
        h = h[7:]
    return h.lstrip("@")


def is_allowlisted(contacts: ContactsConfig, sender: Sender | None) -> bool:
    """The user decides which contacts the agent may see.

    mode "off" sees everything. A senderless event (scheduler ticks, screen
    captures, internal observations) can't be filtered. A sender matches a
    rule when their normalized handles agree, or — for platform-agnostic
    ("*") rules — when the bare handle agrees across platforms.
    """
    if contacts.mode != "enforce" or sender is None:
        return True
    sender_norm = normalize_handle(sender.platform, sender.handle)
    sender_bare = _bare(sender.handle)
    for rule in contacts.allowlist:
        if normalize_handle(rule.platform, rule.handle) == sender_norm:
            return True
        if rule.platform == "*" and sender_bare and _bare(rule.handle) == sender_bare:
            return True
    return False


@dataclass
class Event:
    id: int
    source: str
    kind: str
    occurred_at: datetime
    payload: dict[str, Any]
    salience_score: float
    memorable: bool
    meta: dict[str, Any]


@dataclass
class IngestResult:
    stored: bool
    reason: str  # "new" | "duplicate" | "filtered:contacts"
    event_id: int | None = None


def event_text(event: Event) -> str:
    """The stable string the salience judge and search score against."""
    return json.dumps(event.payload, sort_keys=True, ensure_ascii=False, default=str)


class EventStore:
    def __init__(
        self,
        pool: asyncpg.Pool,
        cipher: Cipher,
        audit: AuditLog,
        contacts: ContactsConfig,
        indexer: VectorMemory | None = None,
    ) -> None:
        self._pool = pool
        self._cipher = cipher
        self._audit = audit
        self._contacts = contacts
        self._indexer = indexer

    # -- ingestion ----------------------------------------------------------

    async def ingest(
        self,
        *,
        source: str,
        kind: str,
        payload: dict[str, Any],
        sender: Sender | None = None,
        occurred_at: datetime | None = None,
        meta: dict[str, Any] | None = None,
    ) -> IngestResult:
        if not is_allowlisted(self._contacts, sender):
            await self._audit.append(
                actor="system",
                tool_name=f"ingest:{source}",
                decision="filtered",
                params={
                    "sender_platform": sender.platform,
                    "sender_handle": sender.handle,
                },
                outcome="filtered by contact allowlist",
            )
            return IngestResult(stored=False, reason="filtered:contacts")

        stored_payload = dict(payload)
        if sender is not None:
            stored_payload[SENDER_KEY] = {
                "platform": sender.platform,
                "handle": sender.handle,
            }
        digest = content_hash(kind, stored_payload)
        meta_json = json.dumps(meta or {}, default=str)

        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "INSERT INTO events (source, kind, occurred_at, content_hash,"
                " payload_enc, meta)"
                " VALUES ($1, $2, COALESCE($3, now()), $4, $5, $6::jsonb)"
                " ON CONFLICT (source, content_hash) DO NOTHING"
                " RETURNING id",
                source,
                kind,
                occurred_at,
                digest,
                b"",
                meta_json,
            )
            if row is None:
                return IngestResult(stored=False, reason="duplicate")
            event_id = row["id"]
            blob = self._cipher.encrypt_json(stored_payload, aad=f"events:payload_enc:{event_id}")
            await conn.execute(
                "UPDATE events SET payload_enc = $1 WHERE id = $2",
                blob,
                event_id,
            )
        # the single indexing choke point: outside the transaction (the FK
        # row is committed), duplicates never reach it — one event, one
        # indexing. shared_doc is skipped by the vector layer itself (the
        # attachment processor indexes the full document text). The call
        # returns at once — the embed+insert runs on a background task, so
        # ingest never waits on the embedding API; a failure leaves
        # indexed_at unset for the loop's backfill to retry.
        if self._indexer is not None:
            try:
                await self._indexer.index_event(
                    event_id,
                    source,
                    kind,
                    stored_payload,
                    occurred_at or datetime.now(tz=UTC),
                )
            except Exception:
                log.warning(
                    "indexing failed for event %s — backfill will retry", event_id, exc_info=True
                )
        return IngestResult(stored=True, reason="new", event_id=event_id)

    # -- reads ----------------------------------------------------------------

    async def get(self, event_id: int) -> Event | None:
        row = await self._pool.fetchrow(
            "SELECT id, source, kind, occurred_at, payload_enc, salience_score,"
            " memorable, meta FROM events WHERE id = $1",
            event_id,
        )
        if row is None:
            return None
        return self._to_event(row)

    async def recent_memorable(self, limit: int = 50, hours: float | None = None) -> list[Event]:
        sql = (
            "SELECT id, source, kind, occurred_at, payload_enc, salience_score,"
            " memorable, meta FROM events WHERE memorable"
        )
        args: list[Any] = []
        if hours is not None:
            sql += " AND occurred_at > now() - make_interval(hours => $1)"
            args.append(hours)
        sql += f" ORDER BY occurred_at DESC LIMIT ${len(args) + 1}"
        args.append(limit)
        rows = await self._pool.fetch(sql, *args)
        return [self._to_event(r) for r in rows]

    async def count_recent(self, source: str, hours: float = 1.0) -> int:
        return await self._pool.fetchval(
            "SELECT COUNT(*) FROM events"
            " WHERE source = $1 AND occurred_at > now() - make_interval(hours => $2)",
            source,
            hours,
        )

    async def list_since(self, after_id: int, limit: int = 50) -> list[Event]:
        """Events ingested after an id, oldest first — the agent loop's
        observation stream; `after_id` starts at max_id() on boot so the
        agent never replays its whole history."""
        rows = await self._pool.fetch(
            "SELECT id, source, kind, occurred_at, payload_enc, salience_score,"
            " memorable, meta FROM events WHERE id > $1 ORDER BY id LIMIT $2",
            after_id,
            limit,
        )
        return [self._to_event(r) for r in rows]

    async def max_id(self) -> int:
        return await self._pool.fetchval("SELECT COALESCE(MAX(id), 0) FROM events")

    async def recent(self, limit: int = 50) -> list[Event]:
        """Newest events, memorable or not — the web feed's source."""
        rows = await self._pool.fetch(
            "SELECT id, source, kind, occurred_at, payload_enc, salience_score,"
            " memorable, meta FROM events ORDER BY id DESC LIMIT $1",
            limit,
        )
        return [self._to_event(r) for r in rows]

    async def count_memorable_since(self, since: datetime) -> int:
        return await self._pool.fetchval(
            "SELECT COUNT(*) FROM events WHERE memorable AND occurred_at >= $1",
            since,
        )

    # -- updates ---------------------------------------------------------------

    async def update_salience(
        self,
        event_id: int,
        score: float,
        memorable: bool,
        category: str = "",
        actionability: str = "",
    ) -> None:
        extra = {
            k: v for k, v in {"category": category, "actionability": actionability}.items() if v
        }
        meta_json = json.dumps(extra, default=str)
        await self._pool.execute(
            "UPDATE events SET salience_score = $1, memorable = $2,"
            " meta = COALESCE(meta, '{}'::jsonb) || $3::jsonb WHERE id = $4",
            score,
            memorable,
            meta_json,
            event_id,
        )

    # -- internals ----------------------------------------------------------------

    def _to_event(self, row: asyncpg.Record) -> Event:
        try:
            payload = self._cipher.decrypt_json(
                row["payload_enc"], aad=f"events:payload_enc:{row['id']}"
            )
        except Exception:
            log.warning("event %s undecryptable — payload replaced with {}", row["id"])
            payload = {}
        # jsonb comes back from asyncpg as a raw JSON string — parse it here
        meta = json.loads(row["meta"]) if row["meta"] else {}
        return Event(
            id=row["id"],
            source=row["source"],
            kind=row["kind"],
            occurred_at=row["occurred_at"],
            payload=payload,
            salience_score=row["salience_score"],
            memorable=row["memorable"],
            meta=meta,
        )


def start_of_today(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)
