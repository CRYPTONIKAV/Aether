"""Resolve configuration into Provider instances, one per role.

A single provider backs the agent, with optional per-role model overrides
(`salience_model`, `vision_model`) that build extra instances of the same
provider class with a different model id — one provider choice governs
everything, while each job can run on a smaller/cheaper model where it
makes sense.
"""

from __future__ import annotations

from ..config import LLMConfig, Settings
from .anthropic_provider import AnthropicProvider
from .base import Provider, ProviderError
from .gemini_provider import GeminiProvider
from .nvidia_provider import NvidiaNimProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider

ROLES = ("reasoning", "salience", "vision")


def _build(settings: Settings, llm: LLMConfig, model: str, vision_capable: bool) -> Provider:
    if llm.provider == "anthropic":
        return AnthropicProvider.build(settings.anthropic_api_key, model, llm.max_tokens)
    if llm.provider == "openai":
        return OpenAIProvider.build(settings.openai_api_key, model, llm.max_tokens)
    if llm.provider == "gemini":
        return GeminiProvider.build(settings.gemini_api_key, model, llm.max_tokens)
    if llm.provider == "nvidia_nim":
        return NvidiaNimProvider.build(
            settings.nvidia_api_key, model, llm.max_tokens, settings.nvidia_base_url
        )
    if llm.provider == "ollama":
        return OllamaProvider.build(
            settings.ollama_base_url, model, llm.max_tokens, supports_vision=vision_capable
        )
    raise ProviderError(f"unknown llm.provider: {llm.provider!r}")


class ProviderRegistry:
    """The resolved provider for each role; `vision` may be None."""

    def __init__(self, providers: dict[str, Provider | None]) -> None:
        self._providers = providers

    def for_role(self, role: str) -> Provider | None:
        if role in self._providers:
            return self._providers[role]
        return self._providers["reasoning"]  # unknown role -> default provider

    @classmethod
    def from_config(cls, settings: Settings, llm: LLMConfig) -> ProviderRegistry:
        # Whether the default instance can see images. For hosted providers
        # that's inherent; for Ollama it depends entirely on the model the
        # user pulled, so it's an explicit config flag.
        default_vision = llm.ollama_vision if llm.provider == "ollama" else True
        default = _build(settings, llm, llm.model, default_vision)

        providers: dict[str, Provider | None] = {
            "reasoning": default,
            "salience": default,
            "vision": default if default.supports_vision else None,
        }
        if llm.salience_model and llm.salience_model != llm.model:
            providers["salience"] = _build(settings, llm, llm.salience_model, default_vision)
        if llm.vision_model and llm.vision_model != llm.model:
            # An explicit vision_model is a deliberate choice — trust that
            # it accepts images even though a local Ollama can't be probed.
            vision = _build(settings, llm, llm.vision_model, vision_capable=True)
            providers["vision"] = vision if vision.supports_vision else None
        return cls(providers)
