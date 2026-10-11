"""Gemini adapter — official `google-genai` SDK, async client."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from google import genai
from google.genai import types as gtypes

from .base import ProviderError
from .types import ImageBlock, Message, TextBlock, ToolCall, ToolResult, ToolSpec, Turn

# JSON Schema spellings -> the casing the genai SDK's Schema type expects.
_TYPE_NAMES = {
    "object": "OBJECT",
    "array": "ARRAY",
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "any": "ANY",
}


# the keys the genai Schema model declares (a subset — $schema and the
# draft-07 combiners Composio ships are rejected outright). Module-level:
# the recursion below touches every node of every tool on every call.
_SCHEMA_KEYS = frozenset(
    {
        "type",
        "format",
        "description",
        "properties",
        "items",
        "anyOf",
        "default",
        "enum",
        "example",
        "required",
        "title",
        "nullable",
        "pattern",
    }
)

# these hold literal payloads, not subschemas — recursing into them would
# rewrite a default like {"type": "text"} into {"type": "TEXT"}
_SCHEMA_PAYLOAD_KEYS = frozenset({"default", "enum", "example"})


def _schema_to_gemini(schema: Any) -> Any:
    """Recursively uppercase `type` names and drop keys the SDK's Schema
    model doesn't declare — Composio's draft-07 schemas carry $schema and
    other combiners the genai SDK rejects outright."""
    if isinstance(schema, dict):
        out: dict[str, Any] = {}
        for key, value in schema.items():
            if key not in _SCHEMA_KEYS:
                continue
            if key in _SCHEMA_PAYLOAD_KEYS:
                out[key] = value
            elif key == "type" and isinstance(value, str):
                out[key] = _TYPE_NAMES.get(value.lower(), value.upper())
            elif key == "properties" and isinstance(value, dict):
                # keys here are property names, not schema keywords —
                # keep them and sanitize each subschema
                out[key] = {name: _schema_to_gemini(sub) for name, sub in value.items()}
            elif isinstance(value, (dict, list)):
                out[key] = _schema_to_gemini(value)
            else:
                out[key] = value
        return out
    if isinstance(schema, list):
        return [_schema_to_gemini(item) for item in schema]
    return schema


def messages_to_gemini(messages: list[Message]) -> list[dict[str, Any]]:
    """Internal messages -> Gemini `contents` bodies."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "user":
            parts: list[dict[str, Any]] = []
            for b in msg.blocks:
                if isinstance(b, TextBlock) and b.text:
                    parts.append({"text": b.text})
                elif isinstance(b, ImageBlock):
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": b.media_type,
                                "data": b.data_b64,
                            }
                        }
                    )
            if parts:
                out.append({"role": "user", "parts": parts})
        elif msg.role == "assistant":
            parts = []
            # provider_extra maps call id -> the thought_signature Gemini 3
            # signed that call with; it must ride back on the replayed part
            # or the next request 400s ("missing a thought_signature").
            signatures = msg.provider_extra if isinstance(msg.provider_extra, dict) else {}
            for b in msg.blocks:
                if isinstance(b, TextBlock) and b.text:
                    parts.append({"text": b.text})
                elif isinstance(b, ToolCall):
                    part: dict[str, Any] = {"function_call": {"name": b.name, "args": b.arguments}}
                    if b.id in signatures:
                        part["thought_signature"] = signatures[b.id]
                    parts.append(part)
            if parts:
                out.append({"role": "model", "parts": parts})
        elif msg.role == "tool":
            # Gemini pairs function responses by name, not id, and carries
            # them in a user turn.
            parts = []
            for b in msg.blocks:
                if isinstance(b, ToolResult):
                    parts.append(
                        {
                            "function_response": {
                                "name": b.name or "tool",
                                "response": {"result": b.content},
                            }
                        }
                    )
            if parts:
                out.append({"role": "user", "parts": parts})
    return out


def tools_to_gemini(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "function_declarations": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": _schema_to_gemini(t.input_schema),
                }
                for t in tools
            ]
        }
    ]


def turn_from_gemini(response: Any) -> Turn:
    """SDK response -> internal Turn. Gemini function calls carry no id, so
    one is synthesized per position; results pair by name anyway. Gemini 3
    signs each function call with a thought_signature — captured into
    provider_extra keyed by call id, for the request mapper to echo back."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise ProviderError("gemini: response contained no candidates")
    candidate = candidates[0]
    parts = getattr(getattr(candidate, "content", None), "parts", None) or []
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    signatures: dict[str, str] = {}
    for i, part in enumerate(parts):
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            call_id = f"gemini_{i}"
            calls.append(
                ToolCall(
                    id=call_id,
                    name=getattr(function_call, "name", ""),
                    arguments=dict(getattr(function_call, "args", None) or {}),
                )
            )
            signature = getattr(part, "thought_signature", None)
            if signature:
                signatures[call_id] = signature
            continue
        text = getattr(part, "text", None)
        if isinstance(text, str):
            text_parts.append(text)
    finish = str(getattr(candidate, "finish_reason", "") or "")
    if finish.endswith("MAX_TOKENS"):
        reason = "max_output_tokens"
    else:
        reason = "tool_use" if calls else "end_turn"
    return Turn(
        text="".join(text_parts),
        tool_calls=calls,
        stop_reason=reason,
        provider_extra=signatures or None,
    )


def _retry_delay_seconds(exc: Exception) -> float | None:
    """Pull the server's suggested delay out of a 429; None = not retryable."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status != 429:
        return None
    match = re.search(r"retry in ([\d.]+)s", str(exc), re.IGNORECASE)
    return min(float(match.group(1)) + 1.0, 120.0) if match else 5.0


class GeminiProvider:
    name = "gemini"
    supports_tools = True

    def __init__(
        self,
        client: genai.Client,
        model: str,
        max_tokens: int = 16000,
        supports_vision: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.supports_vision = supports_vision

    @classmethod
    def build(cls, api_key: str, model: str, max_tokens: int = 16000) -> GeminiProvider:
        if not api_key:
            raise ProviderError(
                "llm.provider is gemini but GEMINI_API_KEY is not set — "
                "add it to .env, or point llm.provider at a different provider in config.yaml"
            )
        return cls(genai.Client(api_key=api_key), model, max_tokens)

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> Turn:
        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": self.max_tokens,
        }
        if tools:
            config_kwargs["tools"] = tools_to_gemini(tools)
        # free-tier 429s carry a "Please retry in Xs" hint — wait it out a
        # couple of times rather than tanking the whole agent tick
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=messages_to_gemini(messages),
                    config=gtypes.GenerateContentConfig(**config_kwargs),
                )
                return turn_from_gemini(response)
            except Exception as exc:
                last_exc = exc
                delay = _retry_delay_seconds(exc)
                if delay is None or attempt == 2:
                    break
                await asyncio.sleep(delay)
        raise ProviderError(f"gemini call failed: {last_exc}") from last_exc
