"""Vector memory: chunking, the scratch-DB round trip, and the no-op path.

The integration tests live behind the AETHER_TEST_DATABASE_URL marker like
the other memory tests: a migrated, truncated database, a fresh key, and a
FakeEmbedder whose word-orthogonal vectors make similarity deterministic.
"""

from __future__ import annotations

import pytest
from fakes import FakeEmbedder

from aether.memory.crypto import Cipher, generate_key_b64
from aether.memory.events import EventStore, start_of_today
from aether.memory.vectors import VectorMemory, split_text


def _store(db, embedder=None):
    cipher = Cipher.from_b64(generate_key_b64())
    return VectorMemory(db, cipher, embedder), cipher


# ---------------------------------------------------------------------------
# split_text: boundaries, overlap, whole-doc passthrough
# ---------------------------------------------------------------------------


def test_split_text_short_passes_through_whole() -> None:
    assert split_text("hello world") == ["hello world"]
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_split_text_breaks_on_paragraph_boundaries() -> None:
    a = "alpha paragraph\n\nbeta paragraph"
    long_a = a.replace("alpha", "alpha " * 400).replace("beta", "beta " * 400)
    parts = split_text(long_a, max_chars=1500, overlap=100)
    assert len(parts) >= 2
    # every part starts clean, no mid-word windows when a boundary exists
    assert all(p and p == p.strip() for p in parts)


def test_split_text_respects_the_cap_and_overlaps() -> None:
    text = " ".join(f"word{i}" for i in range(2000))  # no paragraph breaks
    parts = split_text(text, max_chars=3500, overlap=200)
    assert all(len(p) <= 3500 for p in parts)
    assert len(parts) > 1
    # consecutive parts overlap: the first words of the next part reappear
    # in the tail of the one before (200 chars ≈ 28 words, so 20 is inside
    # the overlap even with boundary stripping)
    assert " ".join(parts[1].split()[:20]) in parts[0]
    # and the whole text is covered — every word survives somewhere
    for i in (0, 500, 1000, 1999):
        assert f"word{i}" in " ".join(parts)


# ---------------------------------------------------------------------------
# the disabled layer is a quiet no-op
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_disabled_vector_memory_never_touches_the_db(db) -> None:
    vectors, _ = _store(db, embedder=None)
    assert vectors.enabled is False
    # every public method stays a safe no-op — memory search then says the
    # plain failure message, indexing just doesn't happen
    await vectors.index_event(9999, "telegram", "chat_message", {"text": "hi"}, start_of_today())
    await vectors.index_doc(9999, "telegram", "a.pdf", [("page 1", "text")])
    assert await vectors.backfill() is None
    assert await vectors.search("anything") == []
    assert await vectors.read_document(9999) == ""


# ---------------------------------------------------------------------------
# round trip: index -> search ranks by similarity -> read_document
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_index_event_chunks_and_search_finds_semantic_neighbors(db) -> None:
    embedder = FakeEmbedder()
    vectors, cipher = _store(db, embedder=embedder)
    from aether.authz.audit import AuditLog
    from aether.config import ContactsConfig

    events = EventStore(db, cipher, AuditLog(db), ContactsConfig(), indexer=vectors)

    await events.ingest(
        source="telegram",
        kind="chat_message",
        payload={"text": "the quarterly tax filing deadline is april 15"},
    )
    await events.ingest(
        source="telegram",
        kind="chat_message",
        payload={"text": "pizza place order number four two two"},
    )
    # indexing runs on a background task — ingest returns before the
    # embed lands, so wait for the in-flight work before asserting
    await vectors.wait_idle()
    hits = await vectors.search("when is the tax deadline?", limit=5)
    assert hits, "expected at least the tax message to come back"
    assert "tax" in hits[0].text
    assert hits[0].kind == "chat_message"
    assert hits[0].score > 0.35

    # the event is marked — backfill has nothing left to do
    assert await vectors.backfill() is None


@pytest.mark.integration
async def test_index_doc_keeps_pages_separate_and_readable(db) -> None:
    embedder = FakeEmbedder()
    vectors, cipher = _store(db, embedder=embedder)
    from aether.authz.audit import AuditLog
    from aether.config import ContactsConfig

    # the store carries the vector hook, so this also proves the generic
    # path leaves shared_doc entirely to the rich one
    events = EventStore(db, cipher, AuditLog(db), ContactsConfig(), indexer=vectors)
    result = await events.ingest(
        source="telegram",
        kind="shared_doc",
        payload={"filename": "deck.pdf", "text": "preview only"},
    )
    assert result.stored and result.event_id is not None
    # the rich path: the same parts the attachment processor would extract
    await vectors.index_doc(
        result.event_id,
        "telegram",
        "deck.pdf",
        [("page 1", "roadmap alpha beta"), ("page 2", "budget gamma delta")],
    )
    # the fake's count-based vectors dilute cosine below the production floor
    # (one shared word over norms sqrt(6)·sqrt(3) ≈ 0.24) — real embeddings
    # sit well above it, so this test names its own bar
    hits = await vectors.search("what is on the roadmap page?", limit=3, min_similarity=0.1)
    assert hits
    assert hits[0].filename == "deck.pdf"
    assert hits[0].location == "page 1"
    assert "roadmap" in hits[0].text

    # the whole document re-reads page by page, location headers included
    doc = await vectors.read_document(result.event_id)
    assert "page 1: roadmap alpha beta" in doc
    assert "page 2: budget gamma delta" in doc

    # generic ingest of a shared_doc event never double-chunks it
    other = await events.ingest(
        source="telegram",
        kind="shared_doc",
        payload={"filename": "other.pdf", "text": "gamma delta epsilon"},
    )
    assert other.stored and other.event_id is not None
    rows = await db.fetch("SELECT filename FROM memory_chunks WHERE event_id = $1", other.event_id)
    assert rows == [], "the generic hook must leave shared_doc to the rich path"


@pytest.mark.integration
async def test_backfill_indexes_unindexed_events_once(db) -> None:
    embedder = FakeEmbedder()
    vectors, cipher = _store(db, embedder=embedder)
    from aether.authz.audit import AuditLog
    from aether.config import ContactsConfig

    # events stored before the vector layer existed: no indexed_at
    plain = EventStore(db, cipher, AuditLog(db), ContactsConfig())
    first = await plain.ingest(
        source="telegram", kind="chat_message", payload={"text": "zeta eta theta reminder"}
    )
    second = await plain.ingest(
        source="telegram", kind="chat_message", payload={"text": "iota kappa lambda memo"}
    )
    assert first.stored and second.stored

    handled = await vectors.backfill(batch=1)
    assert handled == 1  # batched: oldest first
    handled = await vectors.backfill(batch=1)
    assert handled == 1
    assert await vectors.backfill(batch=1) is None  # caught up

    rows = await db.fetch("SELECT COUNT(*) FROM events WHERE indexed_at IS NULL")
    assert rows[0][0] == 0
    # and the backfilled content is searchable
    hits = await vectors.search("zeta reminder", limit=3)
    assert hits and "zeta" in hits[0].text


# ---------------------------------------------------------------------------
# encryption: the vector is ciphertext at rest, legacy rows seal in place
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_indexed_embeddings_are_ciphertext_at_rest(db) -> None:
    from aether.authz.audit import AuditLog
    from aether.config import ContactsConfig
    from aether.llm.embeddings import unpack_vector

    embedder = FakeEmbedder()
    vectors, cipher = _store(db, embedder=embedder)
    events = EventStore(db, cipher, AuditLog(db), ContactsConfig(), indexer=vectors)
    result = await events.ingest(
        source="telegram", kind="chat_message", payload={"text": "sphinx of black quartz"}
    )
    assert result.stored
    await vectors.wait_idle()

    row = await db.fetchrow("SELECT id, embedding, norm, embedding_enc FROM memory_chunks")
    assert row is not None
    # nothing plain remains — the vector lives only in the sealed blob
    assert row["embedding"] is None and row["norm"] is None
    assert row["embedding_enc"] is not None
    # and the blob decrypts, under its own AAD, back to the packed vector
    raw = cipher.decrypt_bytes(row["embedding_enc"], aad=f"chunks:embedding_enc:{row['id']}")
    assert unpack_vector(raw) == embedder._vector("sphinx of black quartz")


@pytest.mark.integration
async def test_legacy_plaintext_embeddings_seal_in_place(db) -> None:
    from aether.authz.audit import AuditLog
    from aether.config import ContactsConfig
    from aether.llm.embeddings import unpack_vector

    embedder = FakeEmbedder()
    vectors, cipher = _store(db, embedder=embedder)
    events = EventStore(db, cipher, AuditLog(db), ContactsConfig(), indexer=vectors)
    result = await events.ingest(
        source="telegram", kind="chat_message", payload={"text": "legacy row"}
    )
    assert result.stored
    await vectors.wait_idle()
    # rewind the row to its pre-encryption shape: plain vector, no sealed one
    plain = embedder._vector("legacy row")
    await db.execute(
        "UPDATE memory_chunks SET embedding_enc = NULL, embedding = $1::real[], norm = $2",
        plain,
        1.0,
    )

    handled = await vectors.encrypt_legacy_chunks(batch=10)
    assert handled == 1
    assert await vectors.encrypt_legacy_chunks(batch=10) == 0  # converged

    row = await db.fetchrow("SELECT id, embedding, embedding_enc FROM memory_chunks")
    assert row["embedding"] is None
    sealed = unpack_vector(
        cipher.decrypt_bytes(row["embedding_enc"], aad=f"chunks:embedding_enc:{row['id']}")
    )
    # float-identical to the vector it started as — sealed, not re-embedded
    assert sealed == plain
    # and the sealed row still comes back from search
    hits = await vectors.search("legacy row", limit=3)
    assert hits and "legacy row" in hits[0].text


@pytest.mark.integration
async def test_search_ranks_encrypted_and_legacy_rows_together(db) -> None:
    from aether.authz.audit import AuditLog
    from aether.config import ContactsConfig

    embedder = FakeEmbedder()
    vectors, cipher = _store(db, embedder=embedder)
    events = EventStore(db, cipher, AuditLog(db), ContactsConfig(), indexer=vectors)
    first = await events.ingest(
        source="telegram", kind="chat_message", payload={"text": "alpha beetle cicada"}
    )
    second = await events.ingest(
        source="telegram", kind="chat_message", payload={"text": "alpha beetle dragonfly"}
    )
    assert first.stored and second.stored
    await vectors.wait_idle()
    # leave the second row mid-transition: plain vector, sealed blob gone
    await db.execute(
        "UPDATE memory_chunks SET embedding_enc = NULL, embedding = $1::real[], norm = $2"
        " WHERE event_id = $3",
        embedder._vector("alpha beetle dragonfly"),
        1.0,
        second.event_id,
    )

    # both score — one from ciphertext, one from the legacy plain column
    hits = await vectors.search("alpha beetle", limit=5, min_similarity=0.1)
    assert {h.text for h in hits} == {"alpha beetle cicada", "alpha beetle dragonfly"}
