"""NVIDIA NIM provider tests — client construction only, no network."""

import pytest

from aether.config import LLMConfig, Settings
from aether.llm.base import ProviderError
from aether.llm.nvidia_provider import DEFAULT_BASE_URL, NvidiaNimProvider
from aether.llm.openai_provider import turn_from_openai
from aether.llm.registry import ProviderRegistry
from aether.llm.types import Message, TextBlock


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_registry_dispatches_nvidia_nim() -> None:
    registry = ProviderRegistry.from_config(
        _settings(nvidia_api_key="ngc-test"),
        LLMConfig(provider="nvidia_nim", model="meta/llama-3.3-70b-instruct"),
    )
    provider = registry.for_role("reasoning")
    assert isinstance(provider, NvidiaNimProvider)
    assert provider.name == "nvidia_nim"
    assert provider.model == "meta/llama-3.3-70b-instruct"


def test_nvidia_nim_points_at_nim_endpoint_by_default() -> None:
    provider = NvidiaNimProvider.build("ngc-test", "m1")
    base_url = str(provider.client.base_url)
    assert base_url.startswith(DEFAULT_BASE_URL)
    assert DEFAULT_BASE_URL == "https://integrate.api.nvidia.com/v1"


def test_nvidia_nim_base_url_is_configurable() -> None:
    provider = NvidiaNimProvider.build("ngc-test", "m1", base_url="https://custom/v1")
    assert str(provider.client.base_url).startswith("https://custom/v1")
    registry = ProviderRegistry.from_config(
        _settings(nvidia_api_key="ngc-test", nvidia_base_url="https://custom/v1"),
        LLMConfig(provider="nvidia_nim", model="m1"),
    )
    custom = registry.for_role("reasoning")
    assert str(custom.client.base_url).startswith("https://custom/v1")


def test_missing_nvidia_key_error_names_the_variable() -> None:
    with pytest.raises(ProviderError, match="NVIDIA_API_KEY"):
        ProviderRegistry.from_config(_settings(), LLMConfig(provider="nvidia_nim", model="m1"))


def test_nvidia_nim_supports_all_three_roles_with_overrides() -> None:
    registry = ProviderRegistry.from_config(
        _settings(nvidia_api_key="ngc-test"),
        LLMConfig(
            provider="nvidia_nim",
            model="meta/llama-3.3-70b-instruct",
            salience_model="meta/llama-3.1-8b-instruct",
            vision_model="nvidia/llama-3.2-11b-vision-instruct",
        ),
    )
    assert registry.for_role("reasoning").model == "meta/llama-3.3-70b-instruct"
    assert registry.for_role("salience").model == "meta/llama-3.1-8b-instruct"
    vision = registry.for_role("vision")
    assert vision is not None
    assert vision.model == "nvidia/llama-3.2-11b-vision-instruct"
    assert vision.supports_vision is True


def test_nvidia_nim_response_mapping_tolerates_bad_tool_json() -> None:
    class _Fn:
        name = "lookup"
        arguments = "{not-json"

    class _Tc:
        id = "call_1"
        function = _Fn()

    class _Msg:
        content = "hi"
        tool_calls = [_Tc()]

    class _Choice:
        message = _Msg()
        finish_reason = "tool_calls"

    class _Resp:
        choices = [_Choice()]

    turn = turn_from_openai(_Resp())
    assert turn.text == "hi"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].arguments == {}
    assert turn.stop_reason == "tool_use"


def test_nvidia_nim_request_mapping_keeps_text_shape() -> None:
    from aether.llm.openai_provider import messages_to_openai

    out = messages_to_openai("sys", [Message(role="user", blocks=[TextBlock(text="hello")])])
    assert out[1] == {"role": "user", "content": "hello"}


async def test_nvidia_nim_complete_uses_openai_max_tokens_spelling() -> None:
    # NIM is an OpenAI-compatible hosted endpoint: like OpenAI (and unlike
    # Ollama) it receives max_completion_tokens. Pinned so a spelling change
    # is deliberate, not silent.
    captured = {}

    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)

            class _Fn:
                name = "lookup"
                arguments = "{}"

            class _Tc:
                id = "call_1"
                function = _Fn()

            class _Msg:
                content = "done"
                tool_calls = [_Tc()]

            class _Choice:
                message = _Msg()
                finish_reason = "stop"

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    provider = NvidiaNimProvider(_Client(), "meta/llama-3.3-70b-instruct", 16000)
    turn = await provider.complete("sys", [Message(role="user", blocks=[TextBlock(text="hi")])], [])
    assert turn.text == "done"
    assert captured["model"] == "meta/llama-3.3-70b-instruct"
    assert captured["max_completion_tokens"] == 16000
    assert "max_tokens" not in captured


def test_nvidia_settings_read_env_names(monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "ngc-env")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://custom/v1")
    s = _settings()
    assert s.nvidia_api_key == "ngc-env"
    assert s.nvidia_base_url == "https://custom/v1"


def test_whitespace_api_key_is_rejected() -> None:
    with pytest.raises(ProviderError, match="NVIDIA_API_KEY"):
        NvidiaNimProvider.build("   ", "m1")
