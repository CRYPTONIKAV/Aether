"""Native-tool tests — the internal tools, exercised through the real
ToolRegistry so the wiring (spec name → handler) is what's under test."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fakes import (
    FakeApprovals,
    FakeAudit,
    FakeConfigManager,
    FakeEntities,
    FakeEventStore,
    FakeRoutines,
    FakeScheduler,
    FakeSurfaceConnector,
    FakeTraces,
    FakeVectorMemory,
)

from aether.agent.loop import CaptureRequestBox, SurfaceFanout
from aether.agent.tools import plain_replay, register_native_tools
from aether.authz.audit import ChainVerification
from aether.connectors.registry import ToolRegistry
from aether.memory.events import Event
from aether.memory.vectors import MemoryHit
from aether.workspace import Workspace


def _event(event_id: int) -> Event:
    return Event(
        id=event_id,
        source="telegram",
        kind="chat_message",
        occurred_at=datetime(2026, 9, 24, 10, 30, tzinfo=UTC),
        payload={"text": "alice owes me a reply"},
        salience_score=7.0,
        memorable=True,
        meta={},
    )


class NativeKit:
    def __init__(
        self,
        events: FakeEventStore | None = None,
        routines: FakeRoutines | None = None,
        traces: FakeTraces | None = None,
        audit: FakeAudit | None = None,
        config_manager: FakeConfigManager | None = None,
        workspace: object | None = None,
        vectors: object | None = None,
    ) -> None:
        self.registry = ToolRegistry()
        self.events = events or FakeEventStore()
        self.entities = FakeEntities()
        self.approvals = FakeApprovals()
        self.scheduler = FakeScheduler()
        self.routines = routines or FakeRoutines()
        self.traces = traces or FakeTraces()
        self.audit = audit or FakeAudit()
        self.connector = FakeSurfaceConnector()
        self.box = CaptureRequestBox()
        self.count = register_native_tools(
            registry=self.registry,
            events=self.events,
            entities=self.entities,
            approvals=self.approvals,
            scheduler=self.scheduler,
            surfaces=SurfaceFanout([self.connector]),
            capture_box=self.box,
            routines=routines,  # None keeps the routine tools unregistered
            traces=traces,  # same for the decision-replay tool
            audit=audit,  # and the integrity check
            config_manager=config_manager,  # and the config tools
            workspace=workspace,  # and the workspace tools
            vectors=vectors,  # and the semantic search layer
        )

    async def run(self, name: str, params: dict) -> str:
        return await self.registry.execute(name, params)


async def test_seven_native_tools_register() -> None:
    kit = NativeKit()
    assert kit.count == 7
    assert len(kit.registry) == 7
    names = {
        t.spec.name
        for t in (
            kit.registry.get(n)
            for n in (
                "memory_search",
                "note_entity",
                "get_pending_approvals",
                "schedule_action",
                "request_screen_capture",
                "send_chat_message",
            )
        )
    }
    assert names == {
        "memory_search",
        "note_entity",
        "get_pending_approvals",
        "schedule_action",
        "request_screen_capture",
        "send_chat_message",
    }


async def test_memory_search_formats_hits_and_misses() -> None:
    kit = NativeKit(FakeEventStore(search_results=[_event(3)]))
    out = await kit.run("memory_search", {"query": "alice reply"})
    assert "[3] 2026-09-24 10:30 telegram/chat_message" in out
    assert "alice owes me a reply" in out

    empty = NativeKit()
    assert await empty.run("memory_search", {"query": "nothing"}) == "No matching memories."
    assert await empty.run("memory_search", {"query": "  "}) == "memory_search needs a query."


async def test_memory_search_rejects_a_non_numeric_limit() -> None:
    kit = NativeKit(FakeEventStore(search_results=[_event(3)]))
    out = await kit.run("memory_search", {"query": "alice reply", "limit": "many"})
    assert out == "memory_search needs limit to be a number."


async def test_memory_search_reads_a_shared_document_back_not_its_json() -> None:
    # a shared doc's payload carries the full extracted text — the JSON
    # dump would clip it away to nothing, so the hit renders the file's own
    # words instead
    doc = Event(
        id=7,
        source="telegram",
        kind="shared_doc",
        occurred_at=datetime(2026, 9, 24, 10, 30, tzinfo=UTC),
        payload={"filename": "report.pdf", "text": "quarterly revenue " + "detail " * 100},
        salience_score=7.0,
        memorable=True,
        meta={},
    )
    kit = NativeKit(FakeEventStore(search_results=[doc]))
    out = await kit.run("memory_search", {"query": "revenue"})
    assert "[7] 2026-09-24 10:30 telegram/shared_doc — report.pdf:" in out
    assert "quarterly revenue" in out
    assert '"text"' not in out  # the JSON dump shape is gone


def _chunk_hit(event_id: int, filename: str = "", location: str = "") -> MemoryHit:
    return MemoryHit(
        event_id=event_id,
        chunk_id=event_id * 10,
        source="telegram",
        kind="shared_doc" if filename else "chat_message",
        filename=filename,
        location=location,
        text=f"chunk text from {filename or 'chat'} {location}".strip(),
        score=0.8,
        occurred_at=datetime(2026, 9, 24, 10, 30, tzinfo=UTC),
    )


async def test_memory_search_with_vectors_ranks_chunks_with_their_location() -> None:
    vectors = FakeVectorMemory(hits=[_chunk_hit(4, "OSY.pdf", "page 3")])
    kit = NativeKit(vectors=vectors)
    out = await kit.run("memory_search", {"query": "deadlock prevention"})
    assert "[4] 2026-09-24 10:30 telegram/shared_doc — OSY.pdf, page 3:" in out
    assert "chunk text from OSY.pdf page 3" in out
    assert vectors.queries == ["deadlock prevention"]
    # the fuzzy scan was never spent — the semantic layer answered
    assert kit.events.ingested == []


async def test_memory_search_rereads_a_whole_document_by_id() -> None:
    vectors = FakeVectorMemory(hits=[_chunk_hit(4, "OSY.pdf", "page 3")])
    vectors.documents[4] = "page 1: intro\n\npage 2: context\n\npage 3: conclusion"
    kit = NativeKit(vectors=vectors)
    out = await kit.run("memory_search", {"query": "", "event_id": 4})
    assert out == "page 1: intro\n\npage 2: context\n\npage 3: conclusion"
    assert vectors.queries == []  # a read, not a search

    # without the vector layer the event payload preview is the read-back
    doc = Event(
        id=4,
        source="telegram",
        kind="shared_doc",
        occurred_at=datetime(2026, 9, 24, 10, 30, tzinfo=UTC),
        payload={"filename": "OSY.pdf", "text": "the old preview text"},
        salience_score=7.0,
        memorable=True,
        meta={},
    )
    fallback = NativeKit(FakeEventStore(events={4: doc}))
    out = await fallback.run("memory_search", {"query": "", "event_id": 4})
    assert "OSY.pdf" in out and "the old preview text" in out
    assert await fallback.run("memory_search", {"query": "", "event_id": 99}) == (
        "No stored memory with id 99."
    )


async def test_memory_search_vector_failure_falls_back_to_the_fuzzy_scan() -> None:
    class ExplodingVectors:
        async def search(self, query, limit=5, min_similarity=0.35):
            raise RuntimeError("embedding api down")

    kit = NativeKit(FakeEventStore(search_results=[_event(3)]), vectors=ExplodingVectors())
    out = await kit.run("memory_search", {"query": "alice reply"})
    assert "[3] 2026-09-24 10:30 telegram/chat_message" in out
    assert "alice owes me a reply" in out


async def test_note_entity_resolves_then_records() -> None:
    kit = NativeKit()
    out = await kit.run(
        "note_entity",
        {
            "handle": "alice@example.com",
            "note": "owes me a reply",
            "platform": "mail",
            "display_name": "Alice",
            "kind": "commitment",
        },
    )
    assert "identity 1" in out
    assert kit.entities.resolved == [("mail", "alice@example.com")]
    identity_id, payload = kit.entities.notes[0]
    assert identity_id == 1
    assert payload["note"] == "owes me a reply"
    assert payload["kind"] == "commitment"

    assert (
        await kit.run("note_entity", {"note": "no handle"})
        == "note_entity needs a handle and a note."
    )


async def test_get_pending_approvals_lists_the_queue() -> None:
    kit = NativeKit()
    assert await kit.run("get_pending_approvals", {}) == "No pending approvals."

    await kit.approvals.create(
        tool_name="mail__send_message",
        params={"to": "a@b.c"},
        note="risky",
    )
    out = await kit.run("get_pending_approvals", {})
    assert "#1 mail__send_message" in out
    assert "a@b.c" in out


async def test_schedule_action_persists_a_future_call() -> None:
    kit = NativeKit()
    when = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
    out = await kit.run(
        "schedule_action",
        {
            "label": "evening summary",
            "run_at": when,
            "tool_name": "mail__send_message",
            "params": {"to": "me@example.com", "body": "summary"},
        },
    )
    assert "authorization gate" in out
    created = kit.scheduler.created[0]
    assert created["label"] == "evening summary"
    assert created["payload"] == {
        "type": "tool",
        "tool": "mail__send_message",
        "params": {"to": "me@example.com", "body": "summary"},
    }
    assert created["run_at"].tzinfo is not None


async def test_schedule_action_rejects_bad_times() -> None:
    kit = NativeKit()
    past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    assert (
        await kit.run(
            "schedule_action",
            {
                "label": "x",
                "run_at": past,
                "tool_name": "t",
            },
        )
        == "run_at must be in the future."
    )
    assert (
        await kit.run(
            "schedule_action",
            {
                "label": "x",
                "run_at": "tomorrow please",
                "tool_name": "t",
            },
        )
        == "run_at 'tomorrow please' is not an ISO datetime (e.g. 2026-09-25T18:30:00+05:30)."
    )
    assert kit.scheduler.created == []


async def test_request_screen_capture_round_trips_the_box() -> None:
    kit = NativeKit()
    out = await kit.run("request_screen_capture", {"reason": "check the deploy"})
    assert "companion" in out
    assert kit.box.take() == "check the deploy"
    assert kit.box.take() is None  # one capture per ask


async def test_send_chat_message_fans_out() -> None:
    kit = NativeKit()
    out = await kit.run("send_chat_message", {"text": "the deploy finished"})
    assert out == "Sent to the user's chat surfaces."
    assert kit.connector.sent == ["the deploy finished"]
    assert await kit.run("send_chat_message", {"text": "  "}) == "send_chat_message needs text."


# ---------------------------------------------------------------------------
# verify_integrity — the record proves itself in plain words
# ---------------------------------------------------------------------------


async def test_audit_wiring_adds_the_integrity_tool() -> None:
    audit = FakeAudit()
    await audit.append(actor="agent", tool_name="mail__list_unread", decision="allow")
    kit = NativeKit(audit=audit)
    assert kit.count == 8

    out = await kit.run("verify_integrity", {})
    assert "decision record" in out
    assert "1 decision, chain intact" in out
    assert "moments ago" in out  # the newest row was just written


async def test_verify_integrity_reports_a_broken_chain() -> None:
    kit = NativeKit(
        audit=FakeAudit(
            verification=ChainVerification(
                ok=False,
                entries=892,
                first_bad_seq=892,
                problem="seq 892: stored hash does not match the entry contents",
            )
        )
    )
    out = await kit.run("verify_integrity", {})
    assert "decision record" in out
    assert "BROKEN at entry #892" in out


async def test_verify_integrity_on_an_empty_record_says_so() -> None:
    kit = NativeKit(audit=FakeAudit())
    assert await kit.run("verify_integrity", {}) == "decision record: empty — nothing recorded yet."


# ---------------------------------------------------------------------------
# plain_replay — the trace as the user reads it in chat
# ---------------------------------------------------------------------------


def _turn_trace(payload: dict) -> FakeTraces:
    traces = FakeTraces()
    traces.add(kind="turn", label="replayed turn", payload=payload)
    return traces


def test_plain_replay_speaks_in_plain_words() -> None:
    traces = _turn_trace(
        {
            "trigger": {
                "messages": [
                    {
                        "surface": "telegram",
                        "handle": "@vedant",
                        "text": "email alice the invoice",
                        "reply_to_id": "",
                        "reply_to_text": "",
                    },
                ],
                "observations": [],
            },
            "calls": [
                {
                    "id": "c1",
                    "name": "mail__send_email",
                    "params": {"to": "alice@x.com"},
                    "decision": "require_approval",
                    "matched_rule": "send",
                    "reason": "sends are risky",
                    "audit_seq": None,
                    "approval_id": 12,
                    "result": "held for approval (#12) — the user has been asked",
                    "is_error": False,
                }
            ],
            "reply": "on it — sent once you approve",
            "chat_refs": {"telegram": "101"},
        }
    )
    out = plain_replay(traces.traces[0])
    assert out.startswith("that message, from the record (trace #1):")
    assert 'You asked: "email alice the invoice"' in out
    assert "I proposed sending an email; the gate held it for your one-tap approval (#12)." in out
    assert 'I replied: "on it — sent once you approve"' in out
    assert "trace #1 in the panel" in out
    # the plain-words rule: never the raw name, never the parameters
    assert "mail__send_email" not in out
    assert "alice@x.com" not in out
    assert "__" not in out


def test_plain_replay_of_a_failed_carry_out_never_leaks_the_error() -> None:
    traces = FakeTraces()
    traces.add(
        kind="carry_out",
        label="approval #5 — gmail__send_message",
        payload={
            "approval_id": 5,
            "tool": "gmail__send_message",
            "params": {},
            "decided_by": "user",
            "result": "unknown tool: gmail__send_message",
            "is_error": True,
            "chat_refs": {"telegram": "102"},
        },
    )
    out = plain_replay(traces.traces[0])
    assert "Approval #5, decided by user." in out
    assert "It couldn't run — I said so at the time." in out
    assert "gmail__send_message" not in out
    assert "unknown tool" not in out


def test_plain_replay_words_allowed_and_denied_calls() -> None:
    traces = _turn_trace(
        {
            "trigger": {
                "messages": [],
                "observations": [
                    {"id": 1, "source": "mail", "kind": "poll:unread", "line": "3 unread"},
                ],
            },
            "calls": [
                {
                    "id": "c1",
                    "name": "note_entity",
                    "params": {},
                    "decision": "allow",
                    "matched_rule": "builtin:internal",
                    "reason": "",
                    "audit_seq": 9,
                    "approval_id": None,
                    "result": "Noted on Alice (identity 1).",
                    "is_error": False,
                },
                {
                    "id": "c2",
                    "name": "telegram__send_message",
                    "params": {"text": "hi"},
                    "decision": "deny",
                    "matched_rule": "quiet mode",
                    "reason": "quiet mode",
                    "audit_seq": 10,
                    "approval_id": None,
                    "result": "denied by policy: quiet mode",
                    "is_error": True,
                },
            ],
            "reply": "noted, and I held off on the ping",
        }
    )
    out = plain_replay(traces.traces[0])
    assert "I noticed 1 new event." in out
    assert "I went ahead with keeping a note — the gate let it through (builtin:internal)." in out
    assert "I proposed sending a message; the gate blocked it (quiet mode)." in out
    assert "telegram__send_message" not in out


def test_plain_replay_of_routine_and_scheduled_fires() -> None:
    traces = FakeTraces()
    traces.add(
        kind="routine",
        label="billing",
        payload={
            "routine": {"id": 1, "label": "billing", "trigger": {"source": "mail"}},
            "event": {"id": 9, "source": "mail", "kind": "poll:unread", "line": "an invoice"},
            "calls": [
                {
                    "id": "c1",
                    "name": "note_entity",
                    "params": {},
                    "decision": "allow",
                    "matched_rule": "builtin:internal",
                    "reason": "",
                    "audit_seq": 11,
                    "approval_id": None,
                    "result": "noted",
                    "is_error": False,
                }
            ],
            "audit_seq": 11,
            "chat_refs": {"telegram": "103"},
        },
    )
    traces.add(
        kind="scheduled",
        label="evening summary",
        payload={
            "action": {"id": 7, "label": "evening summary", "run_at": "2026-09-29 18:30:00+0000"},
            "calls": [
                {
                    "id": "sched-7",
                    "name": "mail__list_messages",
                    "params": {"limit": 1},
                    "decision": "allow",
                    "matched_rule": "builtin:read-only",
                    "reason": "",
                    "audit_seq": 12,
                    "approval_id": None,
                    "result": "3 unread: alice, bob",
                    "is_error": False,
                }
            ],
            "result": "3 unread: alice, bob",
            "is_error": False,
        },
    )
    routine_out = plain_replay(traces.traces[0])
    assert "Your routine 'billing' fired." in routine_out
    assert "I went ahead with keeping a note" in routine_out

    scheduled_out = plain_replay(traces.traces[1])
    assert "The scheduled action 'evening summary' came due." in scheduled_out
    # a read is said as a read, never as a send
    assert (
        "I went ahead with checking mail — the gate let it through (builtin:read-only)."
        in scheduled_out
    )
    assert "sending an email" not in scheduled_out


# ---------------------------------------------------------------------------
# get_config / set_config — the configuration vocabulary as native tools
# ---------------------------------------------------------------------------


async def test_config_tools_register_only_when_wired() -> None:
    """14 with the manager wired, 12 without — the count is the contract
    main.py logs at boot (the workspace tools are counted separately,
    below)."""
    bare = NativeKit(routines=FakeRoutines(), traces=FakeTraces(), audit=FakeAudit())
    assert bare.count == 13
    assert bare.registry.get("get_config") is None
    assert bare.registry.get("set_config") is None

    wired = NativeKit(
        routines=FakeRoutines(),
        traces=FakeTraces(),
        audit=FakeAudit(),
        config_manager=FakeConfigManager(),
    )
    assert wired.count == 15
    assert wired.registry.get("get_config") is not None
    assert wired.registry.get("set_config") is not None


async def test_get_config_delegates_show() -> None:
    manager = FakeConfigManager(show_reply="the whole config")
    kit = NativeKit(config_manager=manager)
    assert await kit.run("get_config", {}) == "the whole config"
    assert await kit.run("get_config", {"path": "agent.tick_seconds"}) == "the whole config"
    assert manager.show_calls == [None, "agent.tick_seconds"]


async def test_set_config_delegates_and_passes_explicit_null_through() -> None:
    manager = FakeConfigManager(set_reply="set")
    kit = NativeKit(config_manager=manager)
    out = await kit.run("set_config", {"op": "set", "path": "agent.tick_seconds", "value": 10})
    assert out == "set"
    assert manager.set_calls == [
        {"op": "set", "path": "agent.tick_seconds", "value": 10, "source": "tool"}
    ]
    # explicit null is a legal value (clearing llm.vision_model) — it must
    # reach the manager, not be mistaken for a missing one
    await kit.run("set_config", {"op": "set", "path": "llm.vision_model", "value": None})
    assert manager.set_calls[-1] == {
        "op": "set",
        "path": "llm.vision_model",
        "value": None,
        "source": "tool",
    }


async def test_set_config_refuses_incomplete_calls_in_plain_words() -> None:
    manager = FakeConfigManager()
    kit = NativeKit(config_manager=manager)
    assert (
        await kit.run("set_config", {"path": "llm.model"})
        == "set_config needs an op: set, add, remove, or reset."
    )
    assert (
        await kit.run("set_config", {"op": "set"})
        == "set_config needs a config path — e.g. agent.tick_seconds."
    )
    assert (
        await kit.run("set_config", {"op": "set", "path": "llm.model"})
        == 'set needs a value — pass "value": null to clear one.'
    )
    assert manager.set_calls == []  # nothing incomplete was ever delegated


# ---------------------------------------------------------------------------
# workspace_* — the human-editable Markdown identity/memory layer
# ---------------------------------------------------------------------------


def _workspace(tmp_path) -> Workspace:
    ws = Workspace(tmp_path / "ws")
    ws.ensure_scaffold()
    return ws


async def test_workspace_tools_register_only_when_wired(tmp_path) -> None:
    bare = NativeKit()
    assert bare.count == 7
    assert bare.registry.get("workspace_read") is None

    wired = NativeKit(workspace=_workspace(tmp_path))
    assert wired.count == 13
    for name in (
        "workspace_read",
        "workspace_search",
        "workspace_remember",
        "workspace_rewrite",
        "workspace_promote",
        "workspace_finish_bootstrap",
    ):
        assert wired.registry.get(name) is not None, name


async def test_workspace_read_returns_scaffolded_files(tmp_path) -> None:
    kit = NativeKit(workspace=_workspace(tmp_path))
    out = await kit.run("workspace_read", {"file": "identity"})
    assert "Name: Aether" in out

    assert "hasn't been introduced" in await kit.run("workspace_read", {"file": "bootstrap"})
    assert await kit.run("workspace_read", {"file": ""}) == (
        "workspace_read needs a file: identity, soul, agents, user, memory, bootstrap, or daily."
    )


async def test_workspace_remember_and_search_round_trip(tmp_path) -> None:
    kit = NativeKit(workspace=_workspace(tmp_path))
    out = await kit.run(
        "workspace_remember", {"text": "prefers concise notifications", "kind": "preference"}
    )
    assert out.startswith("Saved to the workspace as entry ")

    found = await kit.run("workspace_search", {"query": "concise notifications"})
    assert "prefers concise notifications" in found

    assert await kit.run("workspace_remember", {"text": "x"}) == (
        "workspace_remember needs text and a kind: preference, fact, or daily."
    )


async def test_workspace_search_rejects_a_non_numeric_limit(tmp_path) -> None:
    kit = NativeKit(workspace=_workspace(tmp_path))
    await kit.run("workspace_remember", {"text": "note about coffee", "kind": "fact"})
    out = await kit.run("workspace_search", {"query": "coffee", "limit": "many"})
    assert out == "workspace_search needs limit to be a number."


async def test_workspace_remember_refuses_secrets(tmp_path) -> None:
    kit = NativeKit(workspace=_workspace(tmp_path))
    out = await kit.run(
        "workspace_remember",
        {"text": "api_key: sk-abcdefghijklmnopqrstuvwxyz", "kind": "fact"},
    )
    assert out.startswith("not saved:")
    assert await kit.run("workspace_search", {"query": "sk-abcdefghijklmnopqrstuvwxyz"}) == (
        "No matching workspace memory."
    )


async def test_workspace_promote_moves_a_daily_entry_into_memory(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    kit = NativeKit(workspace=workspace)
    today = datetime.now(UTC).date().isoformat()
    saved = await kit.run("workspace_remember", {"text": "deploy went out at 5pm", "kind": "daily"})
    entry_id = saved.rsplit(" ", 1)[-1].rstrip(".")

    promoted = await kit.run("workspace_promote", {"date": today, "entry_id": entry_id})
    assert promoted == f"Promoted {entry_id} into MEMORY.md."
    assert "deploy went out at 5pm" in await kit.run("workspace_read", {"file": "memory"})

    missing = await kit.run("workspace_promote", {"date": today, "entry_id": "doesnotexist"})
    assert missing == f"No active entry doesnotexist on {today}."


async def test_workspace_remember_records_user_vs_agent_origin(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    kit = NativeKit(workspace=workspace)
    await kit.run(
        "workspace_remember", {"text": "told me directly", "kind": "fact", "source": "user"}
    )
    await kit.run("workspace_remember", {"text": "inferred this", "kind": "fact"})
    sources = {e.text: e.source for e in workspace.entries("memory")}
    assert sources["told me directly"] == "user"
    assert sources["inferred this"] == "agent"  # the default when the tool call omits it


async def test_workspace_rewrite_replaces_identity_or_soul(tmp_path) -> None:
    kit = NativeKit(workspace=_workspace(tmp_path))
    out = await kit.run("workspace_rewrite", {"file": "soul", "text": "# SOUL.md\n\n- Be terse.\n"})
    assert out == "soul.md replaced."
    # the tool trims surrounding whitespace from model output before saving
    assert await kit.run("workspace_read", {"file": "soul"}) == "# SOUL.md\n\n- Be terse."


async def test_workspace_rewrite_refuses_other_files(tmp_path) -> None:
    kit = NativeKit(workspace=_workspace(tmp_path))
    out = await kit.run("workspace_rewrite", {"file": "memory", "text": "whatever"})
    assert out.startswith("workspace_rewrite can only replace identity or soul")
    assert "whatever" not in await kit.run("workspace_read", {"file": "memory"})


async def test_workspace_rewrite_refuses_secrets(tmp_path) -> None:
    kit = NativeKit(workspace=_workspace(tmp_path))
    out = await kit.run(
        "workspace_rewrite",
        {"file": "identity", "text": "api_key: sk-abcdefghijklmnopqrstuvwxyz0123456789"},
    )
    assert out.startswith("not saved:")


async def test_workspace_finish_bootstrap_removes_the_marker(tmp_path) -> None:
    kit = NativeKit(workspace=_workspace(tmp_path))
    assert await kit.run("workspace_finish_bootstrap", {}) == "First-run setup marked complete."
    assert await kit.run("workspace_finish_bootstrap", {}) == "No first-run setup was pending."
    assert await kit.run("workspace_read", {"file": "bootstrap"}) == "bootstrap is empty."
