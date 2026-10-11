"""Adapter mapping tests — pure functions, no network, no SDK calls.

These pin the exact wire shapes each provider produces, so a change in a
request body is caught even though the real API is never touched.
"""

from types import SimpleNamespace

import pytest

from aether.llm.anthropic_provider import (
    messages_to_anthropic,
    tools_to_anthropic,
    turn_from_anthropic,
)
from aether.llm.base import ProviderError
from aether.llm.gemini_provider import (
    _schema_to_gemini,
    messages_to_gemini,
    tools_to_gemini,
    turn_from_gemini,
)
from aether.llm.openai_provider import (
    messages_to_openai,
    tools_to_openai,
    turn_from_openai,
)
from aether.llm.types import (
    ImageBlock,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
)

# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_user_text_and_image() -> None:
    msg = Message.user("hello", images=[ImageBlock(data_b64="abc", media_type="image/png")])
    out = messages_to_anthropic([msg])
    assert out == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
                },
            ],
        }
    ]


def test_anthropic_assistant_tool_use_round_trips_thinking() -> None:
    thinking = {"type": "thinking", "thinking": "hm", "signature": "sig"}
    msg = Message.assistant(
        "plan",
        [ToolCall(id="t1", name="search", arguments={"q": "x"})],
        provider_extra=[thinking],
    )
    out = messages_to_anthropic([msg])
    assert out[0]["role"] == "assistant"
    assert out[0]["content"] == [
        thinking,
        {"type": "text", "text": "plan"},
        {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
    ]


def test_anthropic_tool_results_ride_in_user_message() -> None:
    msg = Message.tool_results(
        [ToolResult(tool_call_id="t1", name="search", content="found", is_error=True)]
    )
    out = messages_to_anthropic([msg])
    assert out == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "found"}],
                    "is_error": True,
                }
            ],
        }
    ]


def test_anthropic_empty_messages_are_dropped() -> None:
    out = messages_to_anthropic([Message.assistant(""), Message.user("")])
    assert out == []


def test_anthropic_tools_shape() -> None:
    spec = ToolSpec(name="search", description="Search memory", input_schema={"type": "object"})
    assert tools_to_anthropic([spec]) == [
        {"name": "search", "description": "Search memory", "input_schema": {"type": "object"}}
    ]


def test_anthropic_turn_parsing_skips_thinking_blocks() -> None:
    class FakeThinking:
        type = "thinking"
        thinking = "deep"
        signature = "s"

        def model_dump(self, exclude_none: bool = False) -> dict:
            return {"type": self.type, "thinking": self.thinking, "signature": self.signature}

    response = SimpleNamespace(
        content=[
            FakeThinking(),
            SimpleNamespace(type="text", text="hi"),
            SimpleNamespace(type="tool_use", id="t1", name="search", input={"q": "x"}),
        ],
        stop_reason="tool_use",
    )
    turn = turn_from_anthropic(response)
    assert turn.text == "hi"
    assert turn.tool_calls == [ToolCall(id="t1", name="search", arguments={"q": "x"})]
    assert turn.stop_reason == "tool_use"
    # thinking blocks are preserved for the next request's round-trip
    assert turn.provider_extra == [{"type": "thinking", "thinking": "deep", "signature": "s"}]


def test_anthropic_turn_stop_reason_mapping() -> None:
    for api_reason, expected in [
        ("end_turn", "end_turn"),
        ("max_tokens", "max_output_tokens"),
        (None, "end_turn"),
    ]:
        response = SimpleNamespace(content=[], stop_reason=api_reason)
        assert turn_from_anthropic(response).stop_reason == expected


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def test_openai_user_plain_text_stays_string() -> None:
    out = messages_to_openai("sys", [Message.user("hi")])
    assert out == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_openai_image_becomes_data_url() -> None:
    msg = Message.user("look", images=[ImageBlock(data_b64="abc", media_type="image/png")])
    out = messages_to_openai("sys", [msg])
    assert out[1]["content"] == [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]


def test_openai_assistant_tool_calls_serialize_arguments() -> None:
    msg = Message.assistant(tool_calls=[ToolCall(id="c1", name="f", arguments={"a": 1})])
    out = messages_to_openai("sys", [msg])
    entry = out[1]
    assert entry["role"] == "assistant"
    assert entry["content"] is None
    assert entry["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
    ]


def test_openai_tool_result_shape() -> None:
    msg = Message.tool_results([ToolResult(tool_call_id="c1", name="f", content="res")])
    out = messages_to_openai("sys", [msg])
    assert out[1] == {"role": "tool", "tool_call_id": "c1", "content": "res"}


def test_openai_tools_shape() -> None:
    spec = ToolSpec(name="f", description="d", input_schema={"type": "object"})
    assert tools_to_openai([spec]) == [
        {
            "type": "function",
            "function": {"name": "f", "description": "d", "parameters": {"type": "object"}},
        }
    ]


def test_openai_turn_parsing_tolerates_bad_arguments() -> None:
    good = SimpleNamespace(
        content="hi there",
        tool_calls=[
            SimpleNamespace(id="c1", function=SimpleNamespace(name="f", arguments='{"a": 1}')),
            SimpleNamespace(id="c2", function=SimpleNamespace(name="g", arguments="not json")),
        ],
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=good, finish_reason="tool_calls")])
    turn = turn_from_openai(response)
    assert turn.text == "hi there"
    assert turn.tool_calls[0] == ToolCall(id="c1", name="f", arguments={"a": 1})
    assert turn.tool_calls[1] == ToolCall(id="c2", name="g", arguments={})
    assert turn.stop_reason == "tool_use"


def test_openai_turn_without_choices_raises() -> None:
    with pytest.raises(ProviderError):
        turn_from_openai(SimpleNamespace(choices=[]))


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def test_gemini_user_text_and_image() -> None:
    msg = Message.user("hello", images=[ImageBlock(data_b64="abc", media_type="image/png")])
    out = messages_to_gemini([msg])
    assert out == [
        {
            "role": "user",
            "parts": [
                {"text": "hello"},
                {"inline_data": {"mime_type": "image/png", "data": "abc"}},
            ],
        }
    ]


def test_gemini_assistant_function_call() -> None:
    msg = Message.assistant(tool_calls=[ToolCall(id="x", name="search", arguments={"q": "a"})])
    out = messages_to_gemini([msg])
    assert out == [
        {"role": "model", "parts": [{"function_call": {"name": "search", "args": {"q": "a"}}}]}
    ]


def test_gemini_function_response_pairs_by_name() -> None:
    msg = Message.tool_results(
        [ToolResult(tool_call_id="whatever", name="search", content="found it")]
    )
    out = messages_to_gemini([msg])
    assert out == [
        {
            "role": "user",
            "parts": [
                {"function_response": {"name": "search", "response": {"result": "found it"}}}
            ],
        }
    ]


def test_gemini_schema_uppercases_type_names() -> None:
    schema = {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["q"],
    }
    out = _schema_to_gemini(schema)
    assert out["type"] == "OBJECT"
    assert out["properties"]["q"]["type"] == "STRING"
    assert out["properties"]["tags"]["type"] == "ARRAY"
    assert out["properties"]["tags"]["items"]["type"] == "STRING"
    assert out["required"] == ["q"]


def test_gemini_schema_strips_what_the_sdk_forbids() -> None:
    # Composio tools ship draft-07 schemas — $schema, $ref and friends make
    # the genai SDK's Schema model reject the whole request
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "q": {"type": "string"},
            "meta": {"$ref": "#/$defs/Meta"},
        },
        "$defs": {"Meta": {"type": "object"}},
    }
    out = _schema_to_gemini(schema)
    assert "$schema" not in out
    assert "additionalProperties" not in out
    assert "$defs" not in out
    assert out["properties"]["q"]["type"] == "STRING"  # property names survive
    assert out["properties"]["meta"] == {}  # a $ref degrades, never crashes


def test_gemini_schema_never_rewrites_payload_values() -> None:
    # default/enum/example hold literals, not subschemas — a "type" key in a
    # default must not get uppercased like a schema keyword
    schema = {
        "type": "object",
        "properties": {
            "cfg": {"type": "object", "default": {"type": "text", "value": ""}},
            "mode": {"type": "string", "enum": ["a", "b"], "example": "a"},
        },
    }
    out = _schema_to_gemini(schema)
    assert out["properties"]["cfg"]["default"] == {"type": "text", "value": ""}
    assert out["properties"]["mode"]["enum"] == ["a", "b"]
    assert out["properties"]["mode"]["example"] == "a"


def test_gemini_tools_shape() -> None:
    spec = ToolSpec(name="f", description="d", input_schema={"type": "object"})
    assert tools_to_gemini([spec]) == [
        {
            "function_declarations": [
                {"name": "f", "description": "d", "parameters": {"type": "OBJECT"}}
            ]
        }
    ]


def test_gemini_turn_parsing_synthesizes_ids() -> None:
    part_text = SimpleNamespace(text="hello", function_call=None)
    part_call = SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name="search", args={"q": "x"}),
    )
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[part_text, part_call]),
        finish_reason="STOP",
    )
    turn = turn_from_gemini(SimpleNamespace(candidates=[candidate]))
    assert turn.text == "hello"
    assert turn.tool_calls == [ToolCall(id="gemini_1", name="search", arguments={"q": "x"})]
    assert turn.stop_reason == "tool_use"
    assert turn.provider_extra is None  # nothing signed -> nothing round-tripped


def test_gemini_turn_captures_thought_signatures() -> None:
    # Gemini 3 signs function calls; with parallel calls only the first
    # part carries the signature, and the rest echo back bare.
    part_signed = SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name="f", args={"a": 1}),
        thought_signature="sig-abc",
    )
    part_bare = SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name="g", args={}),
        thought_signature=None,
    )
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[part_signed, part_bare]),
        finish_reason="STOP",
    )
    turn = turn_from_gemini(SimpleNamespace(candidates=[candidate]))
    assert [c.id for c in turn.tool_calls] == ["gemini_0", "gemini_1"]
    assert turn.provider_extra == {"gemini_0": "sig-abc"}


def test_gemini_function_call_parts_replay_their_signatures() -> None:
    msg = Message.assistant(
        "checking",
        [
            ToolCall(id="gemini_0", name="f", arguments={"a": 1}),
            ToolCall(id="gemini_1", name="g", arguments={}),
        ],
        provider_extra={"gemini_0": "sig-abc"},
    )
    out = messages_to_gemini([msg])
    assert out[0]["parts"] == [
        {"text": "checking"},
        {"function_call": {"name": "f", "args": {"a": 1}}, "thought_signature": "sig-abc"},
        {"function_call": {"name": "g", "args": {}}},
    ]


def test_gemini_signature_survives_the_tool_loop_round_trip() -> None:
    # run_tool_loop's exact wiring: parse a response, replay it as
    # Message.assistant(turn.text, turn.tool_calls, turn.provider_extra) —
    # the signature must come back on the function-call part or the next
    # request 400s ("missing a thought_signature").
    candidate = SimpleNamespace(
        content=SimpleNamespace(
            parts=[
                SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(name="memory_search", args={"query": "bob"}),
                    thought_signature="sig-xyz",
                )
            ]
        ),
        finish_reason="STOP",
    )
    turn = turn_from_gemini(SimpleNamespace(candidates=[candidate]))
    history = [Message.assistant(turn.text, turn.tool_calls, turn.provider_extra)]

    parts = messages_to_gemini(history)[0]["parts"]
    assert parts == [
        {
            "function_call": {"name": "memory_search", "args": {"query": "bob"}},
            "thought_signature": "sig-xyz",
        }
    ]


def test_gemini_turn_max_tokens_and_no_candidates() -> None:
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[]), finish_reason="FinishReason.MAX_TOKENS"
    )
    turn = turn_from_gemini(SimpleNamespace(candidates=[candidate]))
    assert turn.text == ""
    assert turn.stop_reason == "max_output_tokens"

    with pytest.raises(ProviderError):
        turn_from_gemini(SimpleNamespace(candidates=[]))
