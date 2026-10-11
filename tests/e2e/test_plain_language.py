"""The plain-language contract, machine-checked: no raw tool names
(x__y), no raw rule ids (builtin:, user:), no raw event kinds
(poll:unread), no raw ISO timestamps, no JSON blobs and no internal
scoring words ever reach visible text — on any page, including every
expanded trace detail. The dev fixture deliberately carries the
backend's real raw shapes (the _event_line log format, UPPER_SNAKE
composio actions, "approval #N — <tool>" labels) so a leak here fails
loudly instead of shipping.
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.e2e

PAGES = [
    "activity",
    "attention",
    "tasks",
    "memory",
    "audit",
    "traces",
    "policy",
    "apps",
    "settings",
]

RAW_PATTERNS = [
    # composio names are UPPER_SNAKE after the __, so the suffix can't be lowercase-only
    (re.compile(r"\b[a-z][a-z0-9]*__[A-Za-z0-9_]+\b"), "raw tool name"),
    (re.compile(r"\b(builtin|default|user|routine|config):[a-z0-9_-]+", re.I), "raw rule id"),
    (re.compile(r"\bpoll:[a-z_]+\b"), "raw event kind"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"), "raw ISO timestamp"),
    (re.compile(r"\{[^{}]*\"[a-z_]+\"[^{}]*:"), "JSON blob"),
    (re.compile(r"\bsalience\b", re.I), "internal scoring word"),
]


def _scan(text: str, where: str) -> None:
    for pattern, what in RAW_PATTERNS:
        hits = pattern.findall(text)
        assert not hits, f"{where}: {what} visible — {hits[:3]!r}"


def test_no_raw_tokens_on_any_page(page, live_server, token) -> None:
    page.add_init_script(f"localStorage.setItem('aether_token', '{token}')")
    page.goto(f"{live_server}/#/activity")
    page.wait_for_selector('[data-testid="page-title"]')

    for name in PAGES:
        page.goto(f"{live_server}/#/{name}")
        page.wait_for_selector('[data-testid="page-title"]')
        page.wait_for_timeout(700)  # let the pollers land
        _scan(page.locator("body").inner_text(), name)

    # expand every trace so the lazy detail bodies are scanned too
    page.goto(f"{live_server}/#/traces")
    page.wait_for_selector('[data-testid="trace-row"]')
    heads = page.locator('[data-testid="trace-head"]')
    for i in range(heads.count()):
        heads.nth(i).click()
        page.wait_for_timeout(400)
    page.wait_for_selector('[data-testid="trace-row"][data-open="1"] [data-testid="trace-call"]')
    page.wait_for_timeout(400)
    _scan(page.locator("body").inner_text(), "traces (all expanded)")


def test_plain_renderings_against_the_fixture(page, live_server, token) -> None:
    page.add_init_script(f"localStorage.setItem('aether_token', '{token}')")

    page.goto(f"{live_server}/#/traces")
    page.wait_for_selector('[data-testid="trace-row"]')
    heads = page.locator('[data-testid="trace-head"]')
    for i in range(heads.count()):
        heads.nth(i).click()
        page.wait_for_timeout(400)
    page.wait_for_selector('[data-testid="trace-row"][data-open="1"] [data-testid="trace-call"]')
    text = page.locator("body").inner_text()

    # the backend's "approval #N — <raw tool>" label reads plainly
    assert "Approval #1 — Sending" in text
    # the model's own `plain` sentence wins where the loop recorded one
    assert "Emailing Sam the Thursday confirmation" in text
    # a call without `plain` (legacy row) is backfilled by the real route
    # from the backend vocabulary — "telegram__send_message" never shows
    assert "Sending a message (to @sam)" in text
    # a composio UPPER_SNAKE action reads as the app, never "composio"
    assert "Sending an email" in text
    assert "composio" not in text.lower()
    # a parked call's result is a model instruction — never quoted
    assert "held for approval (#" not in text
    # observations render like live events, not the raw log line
    assert "Also seen: Checked your email" in text
    assert "Because: Checked your email" in text
    # a line the loop repeated across steps renders once
    assert text.count("Sam asked for a confirmation") == 1
    # model-authored text: raw tool tokens transcribe, pasted JSON collapses
    assert "I will run sending email once approved" in text
    assert "msg-9" not in text
    # decided_by "user" reads as "you"
    assert "decided by you" in text

    page.goto(f"{live_server}/#/activity")
    page.wait_for_selector('[data-testid="event-row"]')
    page.wait_for_timeout(700)
    text = page.locator("body").inner_text()
    # the chat handle rides under _sender — and resolves to a name
    assert "Chat with Sam on Telegram" in text
    # an agent-outcome event transcribes the composio tool and unwraps the
    # envelope — the raw JSON (log_id, successful) never reaches the screen
    assert "Aether · Checking the connected Email account" in text
    assert "sam@example.com" in text
    assert "log_id" not in text
    assert "successful" not in text
    # a legacy agent event without `plain` is backfilled by the real route
    assert "Aether · Sent the message (to @sam)" in text

    page.goto(f"{live_server}/#/attention")
    page.wait_for_selector('[data-testid="approval-card"]')
    page.wait_for_timeout(500)
    text = page.locator("body").inner_text()
    assert "Completed after your approval" in text
    assert "Approved by you" in text
    assert "on You" not in text

    page.goto(f"{live_server}/#/memory")
    page.wait_for_timeout(800)
    assert "Relationship — coworker on the infra team" in page.locator("body").inner_text()

    page.goto(f"{live_server}/#/audit")
    page.wait_for_selector("table")
    page.wait_for_timeout(500)
    text = page.locator("body").inner_text()
    assert "Read-only; nothing was changed" in text
    actors = page.locator("tbody tr td:nth-child(3)").all_inner_texts()
    assert all(a in ("Aether", "You", "Scheduled check", "Routine") for a in actors)

    page.goto(f"{live_server}/#/settings")
    page.wait_for_timeout(800)
    text = page.locator("body").inner_text()
    assert '"warm and a little informal"' not in text
    assert "warm and a little informal" in text
