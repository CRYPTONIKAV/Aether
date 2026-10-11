"""Routines tests.

Unit: the pure matcher (the whole trigger decision table), the loop's fire
path (a match runs through the gate; a risky taught action parks; cooldowns
hold; non-matches are silent), and the native tools round-tripping into
the fake store. Integration: the real store — encrypted at rest, decrypted
on read, and fire bookkeeping surviving a fresh pool, which is the point of
persisting routines at all.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fakes import FakeRoutines
from test_agent_loop import LoopKit
from test_native_tools import NativeKit

from aether.authz.audit import AuditLog
from aether.llm.types import ToolSpec
from aether.memory.crypto import Cipher, generate_key_b64
from aether.memory.events import Event
from aether.routines import Routines, trigger_matches


def _obs(
    event_id: int,
    source: str = "mail",
    kind: str = "poll:unread",
    payload: dict | None = None,
    sender: dict | None = None,
) -> Event:
    payload = dict(payload or {"result": "nothing new"})
    if sender is not None:
        payload["_sender"] = sender  # the shape ingest() stores
    return Event(
        id=event_id,
        source=source,
        kind=kind,
        occurred_at=datetime.now(UTC),
        payload=payload,
        salience_score=0.0,
        memorable=False,
        meta={},
    )


# ---------------------------------------------------------------------------
# unit: the pure matcher
# ---------------------------------------------------------------------------


def test_empty_trigger_matches_everything() -> None:
    assert trigger_matches({}, _obs(1))
    assert trigger_matches({"source": "*"}, _obs(1))


def test_source_and_kind_conditions() -> None:
    event = _obs(1, source="mail", kind="poll:unread")
    assert trigger_matches({"source": "mail"}, event)
    assert trigger_matches({"kind": "poll:unread"}, event)
    assert not trigger_matches({"source": "telegram"}, event)
    assert not trigger_matches({"kind": "chat_message"}, event)


def test_from_condition_normalizes_like_the_allowlist() -> None:
    event = _obs(1, sender={"platform": "mail", "handle": "Alice@Example.com"})
    assert trigger_matches({"from": "alice@example.com"}, event)
    assert trigger_matches({"from": "mailto:alice@example.com"}, event)

    tg = _obs(
        2,
        source="telegram",
        kind="chat_message",
        sender={"platform": "telegram", "handle": "@alice"},
    )
    assert trigger_matches({"from": "@alice"}, tg)
    assert trigger_matches({"from": "alice"}, tg)
    assert not trigger_matches({"from": "@bob"}, tg)


def test_from_condition_requires_a_sender() -> None:
    assert not trigger_matches({"from": "alice"}, _obs(1))  # poll: no _sender


def test_contains_is_case_insensitive_and_ands_with_source() -> None:
    event = _obs(1, source="mail", payload={"result": "INVOICE from Acme attached"})
    assert trigger_matches({"contains": "invoice"}, event)
    assert trigger_matches({"source": "mail", "contains": "acme"}, event)
    assert not trigger_matches({"source": "telegram", "contains": "acme"}, event)
    assert not trigger_matches({"source": "mail", "contains": "rent"}, event)


# ---------------------------------------------------------------------------
# unit: the loop's fire path
# ---------------------------------------------------------------------------


async def test_matching_event_fires_the_action_through_the_gate() -> None:
    kit = LoopKit(None)  # no provider needed: routines fire before the turn
    kit.add_tool("note_entity")
    kit.routines.add(
        label="billing",
        trigger={"source": "mail"},
        action={"type": "tool", "tool": "note_entity", "params": {"note": "invoice"}},
    )
    kit.events.events[1] = _obs(1, source="mail")

    await kit.loop._tick()

    assert kit.executed == [("note_entity", {"note": "invoice"})]
    assert kit.routines.fired == [1]
    # the gate's own row, then the provenance row pointing at the routine
    assert kit.audit.entries[-2]["decision"] == "allow"
    provenance = kit.audit.entries[-1]
    assert provenance["actor"] == "routine"
    assert provenance["rules_matched"] == "routine:1"
    assert provenance["decision"] == "info"
    # the user sees that it fired
    assert any("routine" in s and "billing" in s for s in kit.connector.sent)


async def test_risky_routine_action_parks_instead_of_running() -> None:
    kit = LoopKit(None)
    kit.add_tool("mail__send_message")
    kit.routines.add(
        label="landlord",
        trigger={"contains": "rent"},
        action={"type": "tool", "tool": "mail__send_message", "params": {"body": "on it"}},
    )
    kit.events.events[1] = _obs(1, payload={"result": "rent is due"})

    await kit.loop._tick()

    assert kit.executed == []  # teaching is not a way around the gate
    assert len(kit.approvals.created) == 1
    assert kit.routines.fired == [1]  # fired counts even when parked
    assert any("held for approval" in s for s in kit.connector.sent)


async def test_cooldown_window_holds_a_refire() -> None:
    kit = LoopKit(None)
    kit.add_tool("note_entity")
    kit.routines.add(
        trigger={"source": "mail"},
        action={"type": "tool", "tool": "note_entity", "params": {}},
        cooldown_seconds=300,
        last_fired_at=datetime.now(UTC),
    )
    kit.events.events[1] = _obs(1, source="mail")

    await kit.loop._tick()

    assert kit.executed == []
    assert kit.routines.fired == []


async def test_non_matching_event_is_silent() -> None:
    kit = LoopKit(None)
    kit.add_tool("note_entity")
    kit.routines.add(
        trigger={"source": "mail"}, action={"type": "tool", "tool": "note_entity", "params": {}}
    )
    kit.events.events[1] = _obs(1, source="telegram")

    await kit.loop._tick()

    assert kit.executed == []
    assert kit.routines.fired == []
    assert kit.audit.entries == []
    assert kit.connector.sent == []


async def test_two_matching_routines_each_fire_once() -> None:
    kit = LoopKit(None)
    kit.add_tool("note_entity")
    kit.add_tool("memory_search", result="found it")
    kit.routines.add(
        label="one",
        trigger={"contains": "invoice"},
        action={"type": "tool", "tool": "note_entity", "params": {}},
    )
    kit.routines.add(
        label="two",
        trigger={"source": "mail"},
        action={"type": "tool", "tool": "memory_search", "params": {"query": "invoice"}},
    )
    kit.events.events[1] = _obs(1, source="mail", payload={"result": "invoice arrived"})

    await kit.loop._tick()

    assert sorted(kit.routines.fired) == [1, 2]
    assert {name for name, _ in kit.executed} == {"note_entity", "memory_search"}


async def test_a_failing_routine_does_not_stop_the_others() -> None:
    kit = LoopKit(None)

    async def boom(params: dict) -> str:
        raise RuntimeError("executor exploded")

    # read-only name, so the gate lets it through and the handler raises —
    # the kind of failure _fire_routine has to survive
    kit.tools.add_native(ToolSpec(name="mail__get_boom", description="test"), boom)
    kit.add_tool("note_entity")
    kit.routines.add(
        label="broken",
        trigger={"source": "mail"},
        action={"type": "tool", "tool": "mail__get_boom", "params": {}},
    )
    kit.routines.add(
        label="fine",
        trigger={"source": "mail"},
        action={"type": "tool", "tool": "note_entity", "params": {}},
    )
    kit.events.events[1] = _obs(1, source="mail")

    await kit.loop._tick()

    assert kit.routines.fired == [
        1,
        2,
    ]  # both attempted — the failure counts, so its cooldown gates the retry
    assert kit.executed == [("note_entity", {})]  # only the working one ran
    assert any("routine fire failed" in s for s in kit.connector.sent)


# ---------------------------------------------------------------------------
# unit: the native tools
# ---------------------------------------------------------------------------


async def test_eleven_native_tools_register_when_routines_are_wired() -> None:
    kit = NativeKit(routines=FakeRoutines())
    assert kit.count == 11
    assert len(kit.registry) == 11


async def test_create_routine_validates_its_inputs() -> None:
    kit = NativeKit(routines=FakeRoutines())
    assert "needs a label" in await kit.run(
        "create_routine", {"tool_name": "note_entity", "when_source": "mail"}
    )
    assert "needs a tool_name" in await kit.run(
        "create_routine", {"label": "x", "when_source": "mail"}
    )
    no_condition = await kit.run("create_routine", {"label": "x", "tool_name": "note_entity"})
    assert "at least one condition" in no_condition
    assert kit.routines.created == []
    bad_cooldown = await kit.run(
        "create_routine",
        {
            "label": "x",
            "tool_name": "note_entity",
            "when_source": "mail",
            "cooldown_seconds": "soon",
        },
    )
    assert "cooldown_seconds must be an integer" in bad_cooldown


async def test_create_routine_arms_and_describes() -> None:
    kit = NativeKit(routines=FakeRoutines())
    out = await kit.run(
        "create_routine",
        {
            "label": "billing",
            "when_from": "billing@acme.com",
            "when_contains": "invoice",
            "tool_name": "note_entity",
            "params": {"handle": "billing@acme.com", "note": "invoice received"},
        },
    )
    assert "armed" in out
    assert "authorization gate" in out  # the tool tells the model the truth about fires
    created = kit.routines.created[0]
    assert created["label"] == "billing"
    assert created["trigger"] == {"from": "billing@acme.com", "contains": "invoice"}
    assert created["action"] == {
        "type": "tool",
        "tool": "note_entity",
        "params": {"handle": "billing@acme.com", "note": "invoice received"},
    }


async def test_list_toggle_and_delete_routines() -> None:
    kit = NativeKit(routines=FakeRoutines())
    assert await kit.run("list_routines", {}) == "No routines armed."
    await kit.run(
        "create_routine",
        {"label": "billing", "when_source": "mail", "tool_name": "note_entity"},
    )
    rid = kit.routines.routines[0].id
    await kit.routines.mark_fired(rid)

    listed = await kit.run("list_routines", {})
    assert "'billing'" in listed
    assert "fired 1x" in listed
    assert "when source mail" in listed

    paused = await kit.run("set_routine_enabled", {"routine_id": rid, "enabled": False})
    assert "paused" in paused
    assert "paused" in await kit.run("list_routines", {})

    assert "deleted" in await kit.run("delete_routine", {"routine_id": rid})
    assert await kit.run("list_routines", {}) == "No routines armed."


async def test_routine_tools_report_missing_or_malformed_ids() -> None:
    kit = NativeKit(routines=FakeRoutines())
    assert "No routine 7" in await kit.run("delete_routine", {"routine_id": 7})
    assert "needs a routine_id" in await kit.run("delete_routine", {})
    assert "needs a routine_id" in await kit.run("set_routine_enabled", {"routine_id": "abc"})
    assert "No routine 7" in await kit.run(
        "set_routine_enabled", {"routine_id": 7, "enabled": False}
    )


# ---------------------------------------------------------------------------
# integration: the real store
# ---------------------------------------------------------------------------


def _make_routines(db) -> Routines:
    return Routines(db, Cipher.from_b64(generate_key_b64()), AuditLog(db))


@pytest.mark.integration
async def test_routine_persists_encrypted_and_decrypts_on_read(db) -> None:
    store = _make_routines(db)
    routine = await store.create(
        label="billing",
        trigger={"from": "billing@acme.com", "contains": "invoice"},
        action={
            "type": "tool",
            "tool": "note_entity",
            "params": {"handle": "billing@acme.com", "note": "invoice received"},
        },
        cooldown_seconds=60,
    )
    assert routine.id > 0
    assert routine.enabled is True
    assert routine.fire_count == 0
    assert routine.last_fired_at is None

    # at rest: ciphertext, not the trigger or the action
    raw = await db.fetchrow(
        "SELECT trigger_enc, action_enc FROM routines WHERE id = $1", routine.id
    )
    assert raw["trigger_enc"] not in (b"", None)
    assert b"billing@acme.com" not in raw["trigger_enc"]
    assert b"invoice received" not in raw["action_enc"]

    # and the read path decrypts both back
    fresh = await store.get(routine.id)
    assert fresh is not None
    assert fresh.trigger == {"from": "billing@acme.com", "contains": "invoice"}
    assert fresh.action["params"]["note"] == "invoice received"
    assert fresh.cooldown_seconds == 60


@pytest.mark.integration
async def test_fire_bookkeeping_and_lifecycle_survive_a_fresh_pool(db) -> None:
    store = _make_routines(db)
    routine = await store.create(
        label="billing",
        trigger={"source": "mail"},
        action={"type": "tool", "tool": "note_entity", "params": {}},
    )
    await store.mark_fired(routine.id)
    await store.set_enabled(routine.id, False)

    # a fresh store over the same database (same cipher) sees the same
    # world — persistence is the whole point
    second = Routines(db, store._cipher, AuditLog(db))
    paused = await second.get(routine.id)
    assert paused is not None
    assert paused.enabled is False
    assert paused.fire_count == 1
    assert paused.last_fired_at is not None
    assert await second.list_enabled() == []

    resumed = await second.set_enabled(routine.id, True)
    assert resumed is not None and resumed.enabled is True
    assert [r.id for r in await second.list_enabled()] == [routine.id]

    assert await second.delete(routine.id) is True
    assert await second.get(routine.id) is None
    assert await second.delete(routine.id) is False


@pytest.mark.integration
async def test_list_enabled_orders_by_taught_order(db) -> None:
    store = _make_routines(db)
    first = await store.create(
        label="first",
        trigger={"source": "mail"},
        action={"type": "tool", "tool": "note_entity", "params": {}},
    )
    second = await store.create(
        label="second",
        trigger={"source": "telegram"},
        action={"type": "tool", "tool": "note_entity", "params": {}},
    )

    enabled = await store.list_enabled()
    assert [r.id for r in enabled] == [first.id, second.id]  # oldest first

    listed = await store.list()
    assert [r.id for r in listed] == [second.id, first.id]  # newest first


# ---------------------------------------------------------------------------
# unit: the loop's cooldown uses wall-clock time from the store row
# ---------------------------------------------------------------------------


async def test_cooldown_expires_after_its_window() -> None:
    kit = LoopKit(None)
    kit.add_tool("note_entity")
    kit.routines.add(
        trigger={"source": "mail"},
        action={"type": "tool", "tool": "note_entity", "params": {}},
        cooldown_seconds=60,
        last_fired_at=datetime.now(UTC) - timedelta(seconds=90),
    )
    kit.events.events[1] = _obs(1, source="mail")

    await kit.loop._tick()

    assert kit.executed == [("note_entity", {})]  # the window elapsed: it fires
