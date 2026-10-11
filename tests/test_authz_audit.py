"""Audit chain tests — pure verification logic, plus DB round-trip and
tamper detection behind the integration marker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aether.authz.audit import (
    GENESIS_HASH,
    AuditLog,
    ChainVerification,
    canonical_entry,
    compute_entry_hash,
    params_digest,
    verification_text,
    verify_entries,
)


def _make_entry(prev_hash: str, seq: int, **overrides) -> dict:
    """Build one chain-valid entry dict, the way AuditLog.append does."""
    fields: dict = {
        "seq": seq,
        "actor": "agent",
        "tool_name": "mail__send_email",
        "decision": "approve",
        "rules_matched": "builtin:risky",
        "params_digest": params_digest({"to": "a@b.c"}),
        "outcome": "parked for approval",
        "created_at": f"2026-09-25T00:00:{seq:02d}+00:00",
        "prev_hash": prev_hash,
    }
    fields.update(overrides)
    fields["entry_hash"] = compute_entry_hash(canonical_entry(**fields))
    return fields


def _chain(n: int = 3) -> list[dict]:
    rows: list[dict] = []
    prev = GENESIS_HASH
    for seq in range(1, n + 1):
        row = _make_entry(prev, seq)
        rows.append(row)
        prev = row["entry_hash"]
    return rows


def test_canonical_entry_pins_field_order() -> None:
    a = canonical_entry(
        seq=1,
        actor="a",
        tool_name="t",
        decision="d",
        rules_matched="r",
        params_digest="p",
        outcome="o",
        created_at="c",
        prev_hash="x",
    )
    assert a.startswith('{"actor":"a"')  # sorted keys
    assert "seq" in a and "prev_hash" in a


def test_every_field_is_committed_to_the_hash() -> None:
    base = _make_entry(GENESIS_HASH, 1)
    for field in (
        "actor",
        "tool_name",
        "decision",
        "rules_matched",
        "params_digest",
        "outcome",
        "created_at",
    ):
        # change the field but keep the stored hash — verification must fail
        row = dict(base)
        row[field] = "tampered"
        result = verify_entries([row])
        assert result.ok is False, field


def test_good_chain_verifies() -> None:
    result = verify_entries(_chain(3))
    assert result.ok is True
    assert result.entries == 3
    assert result.first_bad_seq is None


def test_empty_chain_verifies() -> None:
    result = verify_entries([])
    assert result.ok is True
    assert result.entries == 0


def test_tampered_row_is_detected() -> None:
    rows = _chain(3)
    rows[1]["decision"] = "allow"  # someone rewrote history
    result = verify_entries(rows)
    assert result.ok is False
    assert result.first_bad_seq == 2
    assert "seq 2" in (result.problem or "")


def test_broken_link_is_detected() -> None:
    rows = _chain(3)
    rows[2]["prev_hash"] = GENESIS_HASH  # row spliced out of the chain
    result = verify_entries(rows)
    assert result.ok is False
    assert result.first_bad_seq == 3


def test_deleted_middle_entry_is_detected() -> None:
    rows = _chain(4)
    del rows[1]  # seq 2 removed — seq 3 no longer chains
    result = verify_entries(rows)
    assert result.ok is False
    assert result.first_bad_seq == 3


def test_params_digest_is_stable_and_content_bound() -> None:
    assert params_digest({"a": 1, "b": 2}) == params_digest({"b": 2, "a": 1})
    assert params_digest({"a": 1}) != params_digest({"a": 2})


# ---------------------------------------------------------------------------
# verification_text — the chain check as it reads in chat
# ---------------------------------------------------------------------------


def test_verification_text_intact_with_a_recent_act() -> None:
    two_minutes_ago = datetime.now(UTC) - timedelta(minutes=2)
    text = verification_text(ChainVerification(ok=True, entries=1247), two_minutes_ago)
    assert text == (
        "decision record: 1,247 decisions, chain intact — every entry "
        "still hashes to the one before it, last act 2 minutes ago."
    )


def test_verification_text_counts_and_ages_in_plain_units() -> None:
    one = verification_text(ChainVerification(ok=True, entries=1))
    assert one == (
        "decision record: 1 decision, chain intact — every entry still hashes to the one before it."
    )
    hours_old = datetime.now(UTC) - timedelta(minutes=90)
    assert "last act 1 hour ago" in verification_text(
        ChainVerification(ok=True, entries=5), hours_old
    )
    naive_then = (datetime.now(UTC) - timedelta(days=2)).replace(tzinfo=None)
    assert "last act 2 days ago" in verification_text(
        ChainVerification(ok=True, entries=5), naive_then
    )


def test_verification_text_broken_and_empty_states() -> None:
    broken = verification_text(
        ChainVerification(
            ok=False,
            entries=892,
            first_bad_seq=892,
            problem="seq 892: stored hash does not match the entry contents",
        )
    )
    assert broken == (
        "decision record: BROKEN at entry #892 — seq 892: stored hash "
        "does not match the entry contents. Everything from there on "
        "can't be trusted."
    )
    assert (
        verification_text(ChainVerification(ok=True, entries=0))
        == "decision record: empty — nothing recorded yet."
    )


# ---------------------------------------------------------------------------
# Integration: real Postgres round-trip + tamper detection in the database
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_db_round_trip_verifies(db) -> None:
    audit = AuditLog(db)
    for _ in range(3):
        await audit.append(
            actor="agent",
            tool_name="mail__send_email",
            decision="approve",
            params={"to": "a@b.c"},
            outcome="parked for approval",
        )
    result = await audit.verify_chain()
    assert result.ok is True, result.problem
    assert result.entries == 3
    recent = await audit.recent(limit=2)
    assert len(recent) == 2
    assert recent[0]["seq"] == 3


@pytest.mark.integration
async def test_db_tampering_flips_verification(db) -> None:
    audit = AuditLog(db)
    for _ in range(3):
        await audit.append(actor="agent", tool_name="t", decision="allow", params={"x": 1})
    assert (await audit.verify_chain()).ok is True

    # rewrite history directly in SQL, as a database-level attacker would
    await db.execute("UPDATE audit_log SET decision = 'deny' WHERE seq = 2")

    result = await audit.verify_chain()
    assert result.ok is False
    assert result.first_bad_seq == 2


@pytest.mark.integration
async def test_db_appends_chain_after_restart_like_state(db) -> None:
    # a second AuditLog instance over the same pool (e.g. after a restart)
    # continues the chain rather than forking it
    first = AuditLog(db)
    await first.append(actor="agent", tool_name="t", decision="allow")
    second = AuditLog(db)
    await second.append(actor="agent", tool_name="t", decision="allow")
    result = await second.verify_chain()
    assert result.ok is True
    assert result.entries == 2
