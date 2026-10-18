"""NVIDIA NIM adapter — the NIM OpenAI-compatible endpoint.

NIM speaks the chat-completions wire format, so this reuses the OpenAI
adapter wholesale and only swaps the endpoint and the credentials, following
the same pattern `OllamaProvider` uses for its `base_url`.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from .base import ProviderError
from .openai_provider import OpenAIProvider

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaNimProvider(OpenAIProvider):
    name = "nvidia_nim"

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        max_tokens: int = 16000,
        supports_vision: bool = True,
    ) -> None:
        super().__init__(client, model, max_tokens, supports_vision)

    @classmethod
    def build(
        cls,
        api_key: str,
        model: str,
        max_tokens: int = 16000,
        base_url: str = DEFAULT_BASE_URL,
    ) -> NvidiaNimProvider:
        if not (api_key or "").strip():
            raise ProviderError(
                "llm.provider is nvidia_nim but NVIDIA_API_KEY is not set — "
                "add it to .env, or point llm.provider at a different provider in config.yaml"
            )
        return cls(
            AsyncOpenAI(base_url=base_url or DEFAULT_BASE_URL, api_key=api_key),
            model,
            max_tokens,
        )
