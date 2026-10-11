"""Memory store integration tests — real Postgres, real encryption.

These live behind the integration marker (AETHER_TEST_DATABASE_URL): the
db fixture gives each test a migrated, truncated database, so they can
assert on round-trips, ciphertext-at-rest, and the audit side-effects of
ingest / resolve / note.
"""

from __future__ import annotations

import pytest
from fakes import FakePersonJudge

from aether.authz.approvals import Approvals
from aether.authz.audit import AuditLog
from aether.config import AgentConfig, ContactRule, ContactsConfig
from aether.memory.context import ContextBuilder
from aether.memory.crypto import Cipher, generate_key_b64
from aether.memory.entities import Entities, Sender
from aether.memory.events import EventStore, start_of_today


def _contacts(mode: str = "off", *rules: dict) -> ContactsConfig:
    return ContactsConfig(mode=mode, allowlist=[ContactRule(**r) for r in rules])


async def _fresh(
    db, *, contacts: ContactsConfig | None = None, judge: FakePersonJudge | None = None
):
    """A tuple of stores over one fresh key: (events, entities, approvals)."""
    cipher = Cipher.from_b64(generate_key_b64())
    audit = AuditLog(db)
    events = EventStore(db, cipher, audit, contacts or _contacts())
    entities = Entities(db, cipher, audit, judge or FakePersonJudge([]))
    approvals = Approvals(db, cipher, audit)
    return events, entities, approvals


# ---------------------------------------------------------------------------
# events: dedup, allowlist, encryption at rest
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_ingest_dedups_identical_content_per_source(db) -> None:
    events, _, _ = await _fresh(db)
    first = await events.ingest(
        source="mail", kind="message", payload={"text": "hello", "subject": "hi"}
    )
    # key order in the dict is irrelevant — the hash is over canonical JSON
    second = await events.ingest(
        source="mail", kind="message", payload={"subject": "hi", "text": "hello"}
    )
    assert (first.stored, first.reason) == (True, "new")
    assert (second.stored, second.reason) == (False, "duplicate")

    # the dedup key is (source, content_hash): the same words from another
    # source are a different event
    third = await events.ingest(
        source="telegram", kind="message", payload={"text": "hello", "subject": "hi"}
    )
    assert third.stored is True
    assert await db.fetchval("SELECT COUNT(*) FROM events") == 2


@pytest.mark.integration
async def test_allowlist_drop_never_stores_content_and_audits(db) -> None:
    events, _, _ = await _fresh(
        db, contacts=_contacts("enforce", {"platform": "telegram", "handle": "@vedant"})
    )

    dropped = await events.ingest(
        source="telegram",
        kind="message",
        payload={"text": "you never saw this"},
        sender=Sender("telegram", "@stranger"),
    )
    assert (dropped.stored, dropped.reason) == (False, "filtered:contacts")
    assert await db.fetchval("SELECT COUNT(*) FROM events") == 0  # nothing written

    # the only trace is an audit row — and it carries no content
    row = await db.fetchrow(
        "SELECT actor, tool_name, decision, outcome FROM audit_log ORDER BY seq DESC LIMIT 1"
    )
    assert row["tool_name"] == "ingest:telegram"
    assert row["decision"] == "filtered"
    assert row["outcome"] == "filtered by contact allowlist"

    # allowlisted senders still get through
    ok = await events.ingest(
        source="telegram",
        kind="message",
        payload={"text": "hi"},
        sender=Sender("telegram", "@vedant"),
    )
    assert ok.stored is True


@pytest.mark.integration
async def test_sender_rides_inside_the_encrypted_payload(db) -> None:
    events, _, _ = await _fresh(
        db, contacts=_contacts("enforce", {"platform": "*", "handle": "friend@example.com"})
    )
    res = await events.ingest(
        source="gmail",
        kind="message",
        payload={"text": "dinner friday?"},
        sender=Sender("gmail", "friend@example.com"),
    )
    assert res.stored is True

    event = await events.get(res.event_id)
    assert event.payload["_sender"] == {"platform": "gmail", "handle": "friend@example.com"}

    # the handle exists only inside the ciphertext
    raw = await db.fetchval("SELECT payload_enc FROM events WHERE id = $1", res.event_id)
    assert b"friend@example.com" not in raw


# ---------------------------------------------------------------------------
# events: salience reads + search
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_memorable_events_surface_in_reads(db) -> None:
    events, _, _ = await _fresh(db)
    a = await events.ingest(
        source="mail", kind="message", payload={"text": "invoice from acme arrived"}
    )
    b = await events.ingest(
        source="mail", kind="message", payload={"text": "random newsletter blast"}
    )
    await events.update_salience(a.event_id, 8.0, True, category="finance", actionability="review")
    await events.update_salience(b.event_id, 2.0, False)

    memorable = await events.recent_memorable(limit=10)
    assert [e.id for e in memorable] == [a.event_id]
    assert memorable[0].meta.get("category") == "finance"

    assert await events.count_memorable_since(start_of_today()) == 1
    assert await events.count_recent("mail", hours=1) == 2


@pytest.mark.integration
async def test_search_scores_the_recent_window(db) -> None:
    events, _, _ = await _fresh(db)
    await events.ingest(
        source="mail", kind="message", payload={"text": "quarterly tax filing deadline is april 15"}
    )
    await events.ingest(source="telegram", kind="message", payload={"text": "movie night friday"})

    hits = await events.search("tax filing", limit=2)
    assert hits and hits[0].payload["text"].startswith("quarterly")


# ---------------------------------------------------------------------------
# the ingest hook: every stored event is indexed exactly once
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_ingest_indexes_the_event_and_a_duplicate_does_not_reindex(db) -> None:
    from fakes import FakeEmbedder

    from aether.memory.vectors import VectorMemory

    cipher = Cipher.from_b64(generate_key_b64())
    audit = AuditLog(db)
    vectors = VectorMemory(db, cipher, FakeEmbedder())
    events = EventStore(db, cipher, audit, _contacts(), indexer=vectors)

    first = await events.ingest(
        source="telegram", kind="chat_message", payload={"text": "hello from the hook test"}
    )
    second = await events.ingest(
        source="telegram", kind="chat_message", payload={"text": "hello from the hook test"}
    )
    assert (first.stored, second.reason) == (True, "duplicate")

    assert first.event_id is not None
    row = await db.fetchrow(
        "SELECT COUNT(*) AS n, BOOL_OR(indexed_at IS NOT NULL) AS marked FROM events WHERE id = $1",
        first.event_id,
    )
    assert row["n"] == 1 and row["marked"] is True
    # one message, one chunk — the dedup gate sits ahead of the hook
    assert await db.fetchval("SELECT COUNT(*) FROM memory_chunks") == 1
    # and the chunk is searchable
    assert await vectors.search("hello hook", limit=3)


@pytest.mark.integration
async def test_a_filtered_sender_is_never_indexed(db) -> None:
    from fakes import FakeEmbedder

    from aether.memory.vectors import VectorMemory

    cipher = Cipher.from_b64(generate_key_b64())
    audit = AuditLog(db)
    vectors = VectorMemory(db, cipher, FakeEmbedder())
    events = EventStore(
        db,
        cipher,
        audit,
        _contacts("enforce", {"platform": "telegram", "handle": "@vedant"}),
        indexer=vectors,
    )

    dropped = await events.ingest(
        source="telegram",
        kind="chat_message",
        payload={"text": "you never saw this"},
        sender=Sender("telegram", "@stranger"),
    )
    assert dropped.reason == "filtered:contacts"
    assert await db.fetchval("SELECT COUNT(*) FROM memory_chunks") == 0


# ---------------------------------------------------------------------------
# entities: resolution, notes, merges
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_exact_resolution_links_same_email_across_platforms(db) -> None:
    _, entities, _ = await _fresh(db)
    first = await entities.resolve("gmail", "vedant@work.com", "Vedant Mehta")
    # normalized emails are platform-agnostic — outlook hits the same row
    again = await entities.resolve("outlook", "VEDANT@work.com")
    assert again.id == first.id
    assert ("gmail", "vedant@work.com") in again.handles
    assert await db.fetchval("SELECT COUNT(*) FROM identities") == 1


@pytest.mark.integration
async def test_fuzzy_resolution_links_confirmed_candidates(db) -> None:
    judge = FakePersonJudge([0.95])
    _, entities, _ = await _fresh(db, judge=judge)

    first = await entities.resolve("email", "vedant@work.com", "Vedant Mehta")
    # different platform, similar name — a candidate the judge must confirm
    second = await entities.resolve("telegram", "@ved_mehta", "Vedant Mehta")
    assert second.id == first.id
    assert judge.calls  # the LLM was actually consulted
    assert ("telegram", "@ved_mehta") in second.handles


@pytest.mark.integration
async def test_unconfirmed_candidates_create_new_identities(db) -> None:
    judge = FakePersonJudge([0.3, 0.3, 0.3])  # every candidate rejected
    _, entities, _ = await _fresh(db, judge=judge)

    first = await entities.resolve("email", "vedant@work.com", "Vedant Mehta")
    second = await entities.resolve("telegram", "@vm", "Vedant M")
    assert second.id != first.id
    assert await db.fetchval("SELECT COUNT(*) FROM identities") == 2


@pytest.mark.integration
async def test_notes_are_encrypted_and_audited(db) -> None:
    _, entities, _ = await _fresh(db)
    entity = await entities.resolve("telegram", "@vedant", "Vedant")

    await entities.note(
        entity.id, {"kind": "relationship", "text": "owes me a reply about the book club"}
    )

    notes = await entities.notes_for(entity.id)
    assert notes[0].payload["text"] == "owes me a reply about the book club"

    raw = await db.fetchval("SELECT payload_enc FROM entity_notes ORDER BY id DESC LIMIT 1")
    assert b"book club" not in raw

    row = await db.fetchrow("SELECT tool_name, decision FROM audit_log ORDER BY seq DESC LIMIT 1")
    assert (row["tool_name"], row["decision"]) == ("note_entity", "info")


@pytest.mark.integration
async def test_merge_folds_duplicate_into_primary(db) -> None:
    judge = FakePersonJudge([0.0, 0.0, 0.0])  # never confirms -> identities stay split
    _, entities, _ = await _fresh(db, judge=judge)
    primary = await entities.resolve("email", "vedant@work.com", "Vedant Mehta")
    dup = await entities.resolve("telegram", "@vedant", "Vedant Mehta")
    assert dup.id != primary.id

    await entities.merge(primary.id, dup.id)

    assert await entities.get(dup.id) is None  # merged-away identities stop resolving
    moved = await entities.resolve_exact("telegram", "@vedant")
    assert moved is not None and moved.id == primary.id
    row = await db.fetchrow("SELECT tool_name, decision FROM audit_log ORDER BY seq DESC LIMIT 1")
    assert (row["tool_name"], row["decision"]) == ("entity_merge", "info")


# ---------------------------------------------------------------------------
# context builder, end to end over the real stores
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_context_builder_assembles_events_notes_and_approvals(db) -> None:
    events, entities, approvals = await _fresh(db)

    memorable = await events.ingest(
        source="telegram",
        kind="message",
        payload={"text": "dinner friday at 7?"},
        sender=Sender("telegram", "@vedant"),
    )
    await events.update_salience(memorable.event_id, 8.0, True, category="personal")
    noise = await events.ingest(
        source="rss", kind="post", payload={"text": "some syndicated blog item"}
    )
    await events.update_salience(noise.event_id, 1.0, False)

    entity = await entities.resolve("telegram", "@vedant", "Vedant")
    await entities.note(entity.id, {"kind": "relationship", "text": "owes me a reply"})
    parked = await approvals.create(
        tool_name="telegram__send_message",
        params={"text": "happy birthday!"},
        rules_matched="builtin:risky",
    )

    ctx = await ContextBuilder(events, entities, approvals, AgentConfig()).build()

    assert [e.id for e in ctx.events] == [memorable.event_id]  # noise excluded
    assert ctx.notes and ctx.notes[0].payload["text"] == "owes me a reply"
    assert [a.id for a in ctx.pending_approvals] == [parked.id]
    assert ctx.today_memorable == 1
