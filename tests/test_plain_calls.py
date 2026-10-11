"""The backend plain-language vocabulary for tool calls: `describe_call`
and the read-time backfill helpers the API uses for rows that predate the
`plain` field. The panel's contract — no raw tool name, no JSON, ever —
rests on these sentences."""

from __future__ import annotations

from aether.agent.tools import (
    describe_call,
    describe_dismissed,
    describe_outcome,
    plain_outcome,
    with_plain_calls,
    with_plain_event,
)


class TestDescribeCall:
    def test_read_like_mcp_tool_reads_as_a_check(self) -> None:
        assert describe_call("mail__list_messages") == "Checking mail"

    def test_send_words_carry_the_salient_context_never_a_dump(self) -> None:
        assert (
            describe_call(
                "mail__send_message",
                {"to": "sam@example.com", "subject": "Re: Thursday", "body": "x" * 200},
            )
            == "Sending an email (to sam@example.com, subject 'Re: Thursday')"
        )

    def test_composio_reads_run_the_verb_check_on_the_action(self) -> None:
        # regression: the read-like check must see "list_messages", not
        # "gmail_list_messages" — or every composio read reads as a send
        assert describe_call("composio__GMAIL_LIST_MESSAGES") == "Checking gmail"

    def test_composio_who_am_i_idiom(self) -> None:
        assert describe_call("composio__GMAIL_WHO_AM_I") == "Checking the connected gmail account"

    def test_chat_ref_reads_as_the_recipient(self) -> None:
        assert describe_call("telegram__send_message", {"chat_ref": "@sam"}) == (
            "Sending a message (to @sam)"
        )

    def test_native_tool_words(self) -> None:
        assert describe_call("memory_search", {"query": "sam"}) == "Searching my memory (query sam)"

    def test_unknown_app_falls_back_without_the_raw_name(self) -> None:
        text = describe_call("composio__HORIZON_SEND_PING")
        assert text == "An action in horizon"
        assert "HORIZON" not in text and "__" not in text


class TestDescribeOutcome:
    """The chat message after an approved call runs — the result stated
    plainly, with no "Done — tool executed" ceremony."""

    def test_success_is_the_plain_done_sentence(self) -> None:
        assert (
            describe_outcome("mail__send_message", {"to": "sam@example.com"})
            == "Sent the email (to sam@example.com)."
        )

    def test_no_context_variant(self) -> None:
        assert describe_outcome("mail__send_message") == "Sent the email."


class TestDescribeDismissed:
    """The chat acknowledgment when the user taps Deny — a plain okay."""

    def test_denial_acknowledgment(self) -> None:
        assert (
            describe_dismissed("mail__send_message", {"to": "sam@example.com"})
            == "Okay — not sending an email (to sam@example.com)."
        )

    def test_no_context_variant(self) -> None:
        assert describe_dismissed("mail__send_message") == "Okay — not sending an email."


class TestPlainOutcome:
    def test_done_is_the_past_tense_sentence(self) -> None:
        # a completed act must read as finished — the doing form
        # ("Sending an email…") on a done event is what a later turn
        # re-drafts as something still left to do
        assert plain_outcome("mail__list_messages", {}, "action_done") == "Checked mail"
        assert (
            plain_outcome("mail__send_message", {"to": "sam@example.com"}, "action_done")
            == "Sent the email (to sam@example.com)"
        )

    def test_failed_is_framed(self) -> None:
        assert plain_outcome("mail__send_message", {"to": "sam@example.com"}, "action_failed") == (
            "Failed — sending an email (to sam@example.com)"
        )

    def test_denied_reads_as_a_denial(self) -> None:
        assert plain_outcome("mail__send_message", {}, "action_denied") == (
            "Not run — you denied sending an email"
        )


class TestWithPlainCalls:
    def test_fills_missing_plain(self) -> None:
        out = with_plain_calls({"calls": [{"name": "mail__list_messages", "params": {}}]})
        assert out["calls"][0]["plain"] == "Checking mail"

    def test_never_overwrites_an_existing_plain(self) -> None:
        out = with_plain_calls(
            {
                "calls": [
                    {"name": "mail__list_messages", "params": {}, "plain": "Checking mail for Sam"}
                ]
            }
        )
        assert out["calls"][0]["plain"] == "Checking mail for Sam"

    def test_does_not_mutate_the_stored_payload(self) -> None:
        payload = {"calls": [{"name": "mail__list_messages", "params": {}}]}
        with_plain_calls(payload)
        assert "plain" not in payload["calls"][0]

    def test_passes_through_payloads_without_calls(self) -> None:
        assert with_plain_calls({"result": "ok"}) == {"result": "ok"}
        assert with_plain_calls(None) is None


class TestWithPlainEvent:
    def test_fills_a_legacy_agent_outcome(self) -> None:
        out = with_plain_event(
            {"tool": "composio__GMAIL_WHO_AM_I", "params": {}}, kind="action_done"
        )
        assert out["plain"] == "Checked the connected gmail account"

    def test_never_overwrites_an_existing_plain(self) -> None:
        out = with_plain_event(
            {"tool": "mail__list_messages", "params": {}, "plain": "Checking mail for Sam"},
            kind="action_done",
        )
        assert out["plain"] == "Checking mail for Sam"

    def test_leaves_non_agent_payloads_alone(self) -> None:
        assert with_plain_event({"text": "hi"}, kind="chat_message") == {"text": "hi"}
