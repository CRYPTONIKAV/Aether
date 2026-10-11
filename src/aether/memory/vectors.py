"""Vector memory: every event's meaningful text as embedded, sourced chunks.

One retrieval system for everything the agent remembers — chat messages,
shared documents (per page/slide), image descriptions, poll results. Each
chunk keeps the identity of what it came from (event, source, kind,
filename, page/slide), so results from different documents never merge and
every answer can say where it came from.

Indexing rides the single ingest choke point (EventStore) plus a rich
doc path from the attachment processor, which has the *full* extracted
text — the event payload keeps only a 20k-char preview, but the chunks
cover the whole document. `backfill` picks up events that predate the
feature (or lost a race with a rate limit), a batch per tick.

Cosine similarity runs in SQL over `real[]` with precomputed norms —
Postgres does the math in C, and chunk counts are personal-agent scale,
so no pgvector, no numpy, no new service. Chunk text is encrypted like
event payloads; the embedding itself is derived data, stored plain.

Every public method is failure-tolerant: retrieval must never kill a
tick, and a missing embedder turns the whole layer into a quiet no-op
(memory search then reports the failure plainly).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import asyncpg

from ..llm.embeddings import Embedder, vector_norm
from .crypto import Cipher

log = logging.getLogger("aether.memory.vectors")

# the event kinds with a rich indexing path of their own — the attachment
# processor indexes the full document text, so the generic event hook must
# not double-chunk the payload preview
_RICH_INDEX_KINDS = frozenset({"shared_doc"})


@dataclass(frozen=True)
class MemoryHit:
    """One retrieved chunk, with everything needed to cite it."""

    event_id: int
    chunk_id: int
    source: str
    kind: str
    filename: str
    location: str
    text: str
    score: float
    occurred_at: datetime


def split_text(text: str, max_chars: int = 3500, overlap: int = 200) -> list[str]:
    """Long text -> overlapping chunks on paragraph/line boundaries, so a
    sentence never straddles two chunks mid-word. Falls back to hard
    windows only when a single paragraph itself exceeds the budget."""
    text = text.strip()
    if not text or len(text) <= max_chars:
        return [text] if text else []
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # prefer a break at the last paragraph/line/space before the cap
            for sep in ("\n\n", "\n", " "):
                cut = text.rfind(sep, start + max_chars // 2, end)
                if cut != -1:
                    end = cut
                    break
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]


class VectorMemory:
    """The chunked, embedded, searchable half of the event store."""

    def __init__(self, pool: asyncpg.Pool, cipher: Cipher, embedder: Embedder | None) -> None:
        self._pool = pool
        self._cipher = cipher
        self._embedder = embedder
        # background index tasks in flight — tracked so the loop is never
        # garbage-collected mid-await and shutdown can drain them
        self._inflight: set[asyncio.Task[None]] = set()

    @property
    def enabled(self) -> bool:
        return self._embedder is not None

    # -- indexing ---------------------------------------------------------------

    async def index_event(
        self,
        event_id: int,
        source: str,
        kind: str,
        payload: dict[str, Any],
        occurred_at: datetime,
    ) -> None:
        """One event in, chunks out — the generic path EventStore calls on
        every stored ingest. `shared_doc` is skipped: the attachment
        processor indexes the full document text itself. The embed+insert
        runs on a background task so ingest returns at once — indexing is
        catch-up work, never part of the reply path. A failure leaves
        indexed_at unset, and the loop's backfill heals it."""
        if not self.enabled or kind in _RICH_INDEX_KINDS:
            return
        parts = _chunk_text(source, kind, payload)
        self._spawn(self._index_and_mark(event_id, source, kind, "", parts, occurred_at))

    async def _index_and_mark(
        self,
        event_id: int,
        source: str,
        kind: str,
        filename: str,
        parts: list[tuple[str, str]],
        occurred_at: datetime,
    ) -> None:
        await self._index_chunks(event_id, source, kind, filename, parts, occurred_at)
        await self._mark_indexed(event_id)

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._inflight.add(task)
        task.add_done_callback(self._reap)

    def _reap(self, task: asyncio.Task[None]) -> None:
        self._inflight.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            # the event stays indexed_at NULL — backfill retries it
            log.warning("background event indexing failed — backfill will retry", exc_info=exc)

    async def wait_idle(self) -> None:
        """Every background index currently in flight — shutdowns and tests
        wait here so they never assert on work still mid-air."""
        while self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)

    async def index_doc(
        self, event_id: int, source: str, filename: str, parts: list[tuple[str, str]]
    ) -> None:
        """The rich path for shared documents: full extracted text, one part
        per page/slide (location) or size-split for unstructured formats.
        Called by the attachment processor right after the ingest lands."""
        if not self.enabled:
            return
        chunks: list[tuple[str, str]] = []
        for location, text in parts:
            for piece in split_text(text):
                chunks.append((location, piece))
        when = await self._pool.fetchval("SELECT occurred_at FROM events WHERE id = $1", event_id)
        await self._index_chunks(event_id, source, "shared_doc", filename, chunks, when)
        await self._mark_indexed(event_id)

    async def _index_chunks(
        self,
        event_id: int,
        source: str,
        kind: str,
        filename: str,
        parts: list[tuple[str, str]],
        occurred_at: datetime,
    ) -> None:
        """Embed and store `(location, text)` parts as chunk rows. The
        parent event must already be committed (FK). Two-step
        insert-then-update mirrors events.ingest, so the cipher's AAD can
        carry the row id. A failure raises — callers decide whether to
        retry later (backfill leaves indexed_at NULL)."""
        assert self._embedder is not None
        if not parts:
            return
        vectors = await self._embedder.embed([text for _, text in parts], task="document")
        rows: list[tuple[str, str, list[float], float]] = []
        for (location, text), vector in zip(parts, vectors, strict=True):
            rows.append((location, text, vector, vector_norm(vector)))
        async with self._pool.acquire() as conn, conn.transaction():
            for location, text, vector, norm in rows:
                row = await conn.fetchrow(
                    "INSERT INTO memory_chunks (event_id, source, kind, filename,"
                    " location, text_enc, embedding, norm, embed_model, occurred_at)"
                    " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)"
                    " RETURNING id",
                    event_id,
                    source,
                    kind,
                    filename,
                    location,
                    b"",
                    vector,
                    norm,
                    self._embedder.model,
                    occurred_at,
                )
                if row is None:
                    raise RuntimeError(f"chunk insert for event {event_id} returned no id")
                chunk_id = row["id"]
                blob = self._cipher.encrypt_text(text, aad=f"chunks:text_enc:{chunk_id}")
                await conn.execute(
                    "UPDATE memory_chunks SET text_enc = $1 WHERE id = $2",
                    blob,
                    chunk_id,
                )

    async def _mark_indexed(self, event_id: int) -> None:
        await self._pool.execute(
            "UPDATE events SET indexed_at = now() WHERE id = $1 AND indexed_at IS NULL",
            event_id,
        )

    # -- backfill -----------------------------------------------------------------

    async def backfill(self, batch: int = 50) -> int | None:
        """Index the oldest events that have never been attempted — a
        feature-preexisting event, or one whose realtime embed was refused
        by a rate limit. `shared_doc` rows get their payload preview chunked:
        the bytes were never stored, so the preview is all that exists.
        Returns how many were handled, or None when caught up."""
        if not self.enabled:
            return None
        rows = await self._pool.fetch(
            "SELECT id, source, kind, payload_enc, occurred_at FROM events"
            " WHERE indexed_at IS NULL ORDER BY id LIMIT $1",
            batch,
        )
        if not rows:
            return None

        for row in rows:
            try:
                payload = self._cipher.decrypt_json(
                    row["payload_enc"], aad=f"events:payload_enc:{row['id']}"
                )
            except Exception:
                payload = {}
            kind = row["kind"]
            if kind in _RICH_INDEX_KINDS:
                # an old shared document: the attachment processor indexed
                # new ones at ingest; here the preview is all that exists
                text = split_text(str(payload.get("text") or ""))
                parts = [("", piece) for piece in text]
                await self._index_chunks(
                    row["id"],
                    row["source"],
                    kind,
                    str(payload.get("filename") or ""),
                    parts,
                    row["occurred_at"],
                )
            else:
                # _chunk_text already returns (location, text) parts
                parts = _chunk_text(row["source"], kind, payload)
                await self._index_chunks(
                    row["id"], row["source"], kind, "", parts, row["occurred_at"]
                )
            await self._mark_indexed(row["id"])
        return len(rows)

    # -- retrieval -----------------------------------------------------------------

    async def search(
        self, query: str, limit: int = 8, min_similarity: float = 0.35
    ) -> list[MemoryHit]:
        """Semantic search: embed the question, rank every chunk by cosine.
        Postgres does the dot product over `real[]`; the query norm is
        precomputed so the SQL only divides. Hits under the bar drop."""
        q = await self.embed_query(query)
        if q is None:
            return []
        return await self.search_vector(q, limit, min_similarity)

    async def embed_query(self, query: str) -> list[float] | None:
        """The embedding half of search, on its own so a turn can start it
        early and let the round trip overlap its context assembly. None on
        any failure — that turn simply recalls nothing."""
        if not self.enabled or not query.strip():
            return None
        assert self._embedder is not None
        try:
            vectors = await self._embedder.embed([query], task="query")
        except Exception:
            log.warning("query embedding failed — no semantic recall this turn", exc_info=True)
            return None
        return vectors[0] if vectors else None

    async def search_vector(
        self, q: list[float], limit: int = 8, min_similarity: float = 0.35
    ) -> list[MemoryHit]:
        """The SQL half: rank every chunk of this model's space by cosine
        against an already-embedded query."""
        if self._embedder is None:
            return []
        qnorm = vector_norm(q) or 1.0
        rows = await self._pool.fetch(
            "SELECT id, event_id, source, kind, filename, location, text_enc,"
            " occurred_at,"
            " (SELECT sum(u * v)::float8 FROM unnest(embedding, $1::real[]) AS t(u, v))"
            "   / ($2::float8 * norm) AS score"
            " FROM memory_chunks WHERE embed_model = $3"
            " ORDER BY score DESC LIMIT $4",
            q,
            qnorm,
            self._embedder.model,
            limit,
        )
        hits: list[MemoryHit] = []
        for row in rows:
            if row["score"] < min_similarity:
                continue
            try:
                text = self._cipher.decrypt_text(
                    row["text_enc"], aad=f"chunks:text_enc:{row['id']}"
                )
            except Exception:
                log.warning("chunk %s undecryptable — skipped", row["id"])
                continue
            hits.append(
                MemoryHit(
                    event_id=row["event_id"],
                    chunk_id=row["id"],
                    source=row["source"],
                    kind=row["kind"],
                    filename=row["filename"],
                    location=row["location"],
                    text=text,
                    score=float(row["score"]),
                    occurred_at=row["occurred_at"],
                )
            )
        return hits

    async def read_document(self, event_id: int) -> str:
        """Every stored chunk of one event, joined with its location
        headers — the full untruncated document, page by page. The
        channel memory_search uses to re-read a shared document by id."""
        rows = await self._pool.fetch(
            "SELECT id, location, text_enc FROM memory_chunks WHERE event_id = $1 ORDER BY id",
            event_id,
        )
        parts: list[str] = []
        for row in rows:
            try:
                text = self._cipher.decrypt_text(
                    row["text_enc"], aad=f"chunks:text_enc:{row['id']}"
                )
            except Exception:
                continue
            header = f"{row['location']}: " if row["location"] else ""
            parts.append(header + text)
        return "\n\n".join(parts)


def _chunk_text(source: str, kind: str, payload: dict[str, Any]) -> list[tuple[str, str]]:
    """The one chunk a plain event contributes — its meaningful text,
    not the raw JSON. Messages are their words; image memories are the
    description; feed events are their payload's own rendering."""
    p = payload or {}
    if kind == "chat_message" and p.get("text"):
        return [("", str(p["text"]))]
    if kind == "shared_image":
        bits = [
            str(p.get("description") or ""),
            str(p.get("caption") or ""),
            "people: " + ", ".join(p["people"]) if p.get("people") else "",
            "visible text: " + " | ".join(p["text_visible"]) if p.get("text_visible") else "",
        ]
        text = "; ".join(b for b in bits if b)
        return [("", text)] if text else []
    from .events import event_text  # local: events imports vectors' protocol

    rendered = event_text(SimpleNamespace(payload=p))  # reads .payload only
    if rendered and len(rendered) <= 6000:
        return [("", rendered)]
    return []
