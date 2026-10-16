"""json_utils + memory unit tests: hash, allowlist, normalization, heuristics."""

from __future__ import annotations

import pytest
from fakes import FakeEventStore, FakeJudge, FakeProvider

from aether.config import AppContacts, ContactRule, ContactsConfig, SalienceConfig
from aether.llm.json_utils import JsonParseError, extract_json
from aether.llm.types import Turn
from aether.memory.context import apply_daily_cap, bound_events, collect_senders
from aether.memory.entities import PersonRef, Sender, candidate_score, normalize_handle
from aether.memory.events import (
    Event,
    content_hash,
    event_text,
    is_allowlisted,
    start_of_today,
)
from aether.memory.salience import (
    DEFAULT_JUDGE_SCORE,
    JudgeError,
    Judgment,
    LLMJudge,
    Salience,
    heuristic_gate,
)

# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


def test_extract_json_plain() -> None:
    assert extract_json('{"salience": 7}') == {"salience": 7}


def test_extract_json_fenced_and_wrapped_in_prose() -> None:
    text = 'Sure!\n```json\n{"salience": 7, "category": "message"}\n```\nHope that helps.'
    assert extract_json(text) == {"salience": 7, "category": "message"}


def test_extract_json_with_nested_object() -> None:
    text = 'prefix {"a": {"b": [1, 2]}, "c": true} suffix'
    assert extract_json(text) == {"a": {"b": [1, 2]}, "c": True}


def test_extract_json_rejects_garbage() -> None:
    for bad in ("", "   ", "no braces at all", '{"salience": ', "[1, 2]"):
        with pytest.raises(JsonParseError):
            extract_json(bad)


# ---------------------------------------------------------------------------
# handle normalization + contact allowlist
# ---------------------------------------------------------------------------


def test_normalize_handle_forms() -> None:
    assert normalize_handle("gmail", "Friend@Example.com") == "email:friend@example.com"
    assert normalize_handle("*", "mailto:X@Y.com") == "email:x@y.com"
    # phones are platform-agnostic and key on the last 10 digits
    assert normalize_handle("phone", "+1 (555) 010-1234") == normalize_handle("sms", "5550101234")
    assert normalize_handle("whatsapp", "+91 98765 43210") == "tel:9876543210"
    # platform handles embed their platform
    assert normalize_handle("telegram", "@vedant") == "telegram:vedant"
    assert normalize_handle("discord", "@vedant") == "discord:vedant"
    assert normalize_handle("slack", "U123ABC") == "slack:u123abc"
    assert normalize_handle("telegram", "") == ""


def _contacts(mode: str, *rules: dict) -> ContactsConfig:
    return ContactsConfig(mode=mode, allowlist=[ContactRule(**r) for r in rules])


def _seen(contacts: ContactsConfig, sender: Sender | None) -> bool:
    """The ingestion question: is this sender allowlisted where they arrived?
    The app key is the sender's own platform."""
    return is_allowlisted(contacts, sender.platform if sender else "", sender)


def test_allowlist_mode_off_sees_everything() -> None:
    contacts = _contacts("off")
    assert _seen(contacts, Sender("telegram", "@stranger")) is True


def test_allowlist_senderless_events_pass() -> None:
    contacts = _contacts("enforce", {"platform": "telegram", "handle": "@vedant"})
    assert is_allowlisted(contacts, "telegram", None) is True


def test_allowlist_matches_platform_rules_and_star_rules() -> None:
    contacts = _contacts(
        "enforce",
        {"platform": "telegram", "handle": "@vedant"},
        {"platform": "*", "handle": "friend@example.com"},
    )
    assert _seen(contacts, Sender("telegram", "@vedant")) is True
    assert _seen(contacts, Sender("gmail", "friend@example.com")) is True  # * email rule
    assert _seen(contacts, Sender("outlook", "Friend@Example.com")) is True  # normalized match
    assert _seen(contacts, Sender("telegram", "@rando")) is False


def test_allowlist_star_rule_matches_bare_handles_across_platforms() -> None:
    contacts = _contacts("enforce", {"platform": "*", "handle": "@vedant"})
    assert _seen(contacts, Sender("discord", "@vedant")) is True
    assert _seen(contacts, Sender("discord", "@not-vedant")) is False


def test_allowlist_phone_rule() -> None:
    contacts = _contacts("enforce", {"platform": "phone", "handle": "+15550101234"})
    assert _seen(contacts, Sender("sms", "5550101234")) is True
    assert _seen(contacts, Sender("sms", "5559999999")) is False


def test_per_app_list_only_counts_on_its_app() -> None:
    # gmail's own people never open discord — the app key picks the slice
    contacts = ContactsConfig(
        mode="enforce",
        gmail=AppContacts(allowlist=[ContactRule(platform="gmail", handle="friend@example.com")]),
    )
    assert is_allowlisted(contacts, "gmail", Sender("email", "friend@example.com")) is True
    assert is_allowlisted(contacts, "discord", Sender("discord", "@friend")) is False


def test_per_app_switch_overrides_the_default() -> None:
    # the default says enforce; discord's own switch lets everyone in
    contacts = ContactsConfig(mode="enforce", discord=AppContacts(mode="off"))
    assert is_allowlisted(contacts, "discord", Sender("discord", "@stranger")) is True
    assert is_allowlisted(contacts, "telegram", Sender("telegram", "@stranger")) is False
    # and the mirror: default off, one app enforcing its own list
    enforcing = ContactsConfig(
        mode="off",
        telegram=AppContacts(
            mode="enforce", allowlist=[ContactRule(platform="telegram", handle="@sam")]
        ),
    )
    assert is_allowlisted(enforcing, "telegram", Sender("telegram", "@sam")) is True
    assert is_allowlisted(enforcing, "telegram", Sender("telegram", "@stranger")) is False
    assert is_allowlisted(enforcing, "discord", Sender("discord", "@stranger")) is True


def test_content_hash_is_canonical_and_kind_bound() -> None:
    assert content_hash("message", {"a": 1, "b": 2}) == content_hash("message", {"b": 2, "a": 1})
    assert content_hash("message", {"a": 1}) != content_hash("file_event", {"a": 1})
    assert content_hash("message", {"a": 1}) != content_hash("message", {"a": 2})


# ---------------------------------------------------------------------------
# entity candidate scoring
# ---------------------------------------------------------------------------


def test_candidate_score_signals() -> None:
    # same phone digits -> perfect
    assert (
        candidate_score(PersonRef("sms", "+15550101234"), PersonRef("phone", "555-010-1234"))
        == 100.0
    )
    # same email local part -> strong
    assert (
        candidate_score(
            PersonRef("email", "vedant@work.com", ""), PersonRef("gmail", "vedant@gmail.com", "")
        )
        > 80
    )
    # same display names -> strong
    assert (
        candidate_score(
            PersonRef("telegram", "@v", "Vedant Mehta"), PersonRef("discord", "@vm", "Vedant Mehta")
        )
        >= 80
    )
    # unrelated -> weak
    assert candidate_score(PersonRef("telegram", "@alpha"), PersonRef("slack", "U999")) < 50


# ---------------------------------------------------------------------------
# salience heuristics + pipeline (FakeStore/FakeJudge — no DB)
# ---------------------------------------------------------------------------


def test_heuristic_gate_catches_obvious_noise() -> None:
    assert heuristic_gate("Click here to unsubscribe from our newsletter") is not None
    assert heuristic_gate("hey, are we still on for dinner friday?") is None


def _event(event_id: int, payload: dict) -> Event:
    return Event(
        id=event_id,
        source="mail",
        kind="message",
        occurred_at=start_of_today(),
        payload=payload,
        salience_score=0.0,
        memorable=False,
        meta={},
    )


async def test_noise_event_scores_zero_without_judge_call() -> None:
    store = FakeEventStore(events={1: _event(1, {"text": "unsubscribe from our newsletter"})})
    judge = FakeJudge([])
    score = await Salience(store, judge, SalienceConfig()).score_event(1)
    assert score == 0.0
    assert judge.calls == []  # the LLM was never spent
    assert store.updates == [(1, 0.0, False, {"category": "noise", "actionability": ""})]


async def test_rate_capped_source_is_pre_gated() -> None:
    store = FakeEventStore(
        events={1: _event(1, {"text": "hi from a flood"})},
        recent_counts={"mail": 21},  # above the default cap of 20/hour
    )
    judge = FakeJudge([])
    score = await Salience(store, judge, SalienceConfig()).score_event(1)
    assert score == 0.0
    assert judge.calls == []


async def test_judged_event_scores_and_marks_memorable() -> None:
    store = FakeEventStore(events={1: _event(1, {"text": "mom: dinner friday?"})})
    judge = FakeJudge(
        [Judgment(salience=7.5, category="family", actionability="reply", one_line="dinner plan")]
    )
    score = await Salience(store, judge, SalienceConfig()).score_event(1)
    assert score == 7.5
    assert judge.calls == [event_text(store.events[1])]
    assert store.updates == [(1, 7.5, True, {"category": "family", "actionability": "reply"})]


async def test_judge_scores_are_clamped() -> None:
    store = FakeEventStore(events={1: _event(1, {"text": "x"})})
    judge = FakeJudge([Judgment(salience=99.0, category="", actionability="", one_line="")])
    assert await Salience(store, judge, SalienceConfig()).score_event(1) == 10.0

    store2 = FakeEventStore(events={2: _event(2, {"text": "x"})})
    judge2 = FakeJudge([Judgment(salience=-5.0, category="", actionability="", one_line="")])
    assert await Salience(store2, judge2, SalienceConfig()).score_event(2) == 0.0


async def test_judge_failure_defaults_conservatively() -> None:
    store = FakeEventStore(events={1: _event(1, {"text": "important thing"})})
    judge = FakeJudge([RuntimeError("provider down")])
    score = await Salience(store, judge, SalienceConfig()).score_event(1)
    assert score == DEFAULT_JUDGE_SCORE  # 5.0 — below threshold, not memorable
    assert store.updates[0][2] is False  # memorable=False


async def test_missing_event_scores_zero() -> None:
    store = FakeEventStore()
    score = await Salience(store, FakeJudge([]), SalienceConfig()).score_event(404)
    assert score == 0.0
    assert store.updates == []


async def test_llm_judge_parses_and_validates() -> None:
    provider = FakeProvider(
        [
            Turn(
                text='{"salience": 6, "category": "work", "actionability": "review", "one_line": "invoice arrived"}'
            )
        ]
    )
    judgment = await LLMJudge(provider).judge("anything")
    assert judgment.salience == 6.0
    assert judgment.category == "work"

    # missing required key -> JudgeError, not a crash in the caller
    bad = FakeProvider([Turn(text='{"category": "work"}')])
    with pytest.raises(JudgeError):
        await LLMJudge(bad).judge("anything")


# ---------------------------------------------------------------------------
# context building (pure parts)
# ---------------------------------------------------------------------------


def _ctx_event(event_id: int, days_ago: int, score: float, payload_size: int = 50) -> Event:
    from datetime import timedelta

    return Event(
        id=event_id,
        source="s",
        kind="k",
        occurred_at=start_of_today() - timedelta(days=days_ago) + timedelta(hours=event_id % 12),
        payload={"text": "x" * payload_size},
        salience_score=score,
        memorable=True,
        meta={},
    )


def test_daily_cap_keeps_top_scoring_todays_events() -> None:
    today = [_ctx_event(i, 0, score=10 - i) for i in range(5)]  # scores 10,9,8,7,6
    yesterday = [_ctx_event(100, 1, score=1.0)]
    events = today + yesterday
    capped = apply_daily_cap(events, cap=2)
    assert len([e for e in capped if e.occurred_at.date() == start_of_today().date()]) == 2
    assert capped[0].id in (0, 1)  # the two highest-scoring survive
    assert capped[-1].id == 100  # older history untouched
    assert capped == sorted(capped, key=lambda e: e.occurred_at, reverse=True)


def test_daily_cap_noop_when_under_cap() -> None:
    events = [_ctx_event(0, 0, 5.0), _ctx_event(1, 0, 6.0)]
    assert apply_daily_cap(events, cap=20) == events


def test_bound_events_stops_at_char_budget() -> None:
    events = [_ctx_event(i, 0, 5.0, payload_size=100) for i in range(10)]
    bounded = bound_events(events, budget_chars=250)
    assert len(bounded) == 2  # ~3 fit, budget stops before the third


def test_collect_senders_finds_nested_sender_refs() -> None:
    e1 = _ctx_event(1, 0, 5.0)
    e1.payload = {
        "_sender": {"platform": "telegram", "handle": "@vedant"},
        "forwarded": {"_sender": {"platform": "email", "handle": "x@y.z"}},
    }
    e2 = _ctx_event(2, 0, 5.0)
    e2.payload = {"text": "no senders here"}
    senders = collect_senders([e1, e2])
    assert (Sender("telegram", "@vedant") in senders) is True
    assert (Sender("email", "x@y.z") in senders) is True
    assert len(senders) == 2
