"""Embeddings: the free model that makes memory semantically searchable.

One protocol, one adapter today. `GeminiEmbedder` rides the same AI Studio
key and google-genai SDK as the chat provider, on `gemini-embedding-001`
(free tier) stored at 768 Matryoshka dims — Google's recommended
quality/size point. A future adapter (e.g. Jina's free tier) only has to
implement `Embedder`.

Failures here must never cost the caller anything: indexing skips a batch
and retries via backfill, retrieval falls back to keyword search.
"""

from __future__ import annotations

import asyncio
import math
from typing import Literal, Protocol, runtime_checkable

from google import genai
from google.genai import types as gtypes

from .base import ProviderError

# embed_content takes a list, but one giant request is a single 429 away
# from losing the whole batch — keep requests small and retryable
_BATCH = 50
# gemini-embedding-001 is Matryoshka: 3072 by default, 768 recommended
_DEFAULT_DIMS = 768


@runtime_checkable
class Embedder(Protocol):
    """Text in, vectors out — the whole contract the memory layer needs."""

    model: str
    dims: int

    async def embed(
        self, texts: list[str], task: Literal["query", "document"] = "document"
    ) -> list[list[float]]: ...


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _retry_delay(exc: Exception) -> float | None:
    """The server's own suggested delay out of a 429; None = give up."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status != 429:
        return None
    import re

    match = re.search(r"retry in ([\d.]+)s", str(exc), re.IGNORECASE)
    return min(float(match.group(1)) + 1.0, 120.0) if match else 5.0


class GeminiEmbedder:
    """gemini-embedding-001 via the google-genai SDK — the free tier."""

    model = "gemini-embedding-001"
    dims = _DEFAULT_DIMS

    def __init__(self, client: genai.Client, model: str, dims: int = _DEFAULT_DIMS) -> None:
        self.client = client
        self.model = model
        self.dims = dims

    @classmethod
    def build(cls, api_key: str, model: str = "gemini-embedding-001") -> GeminiEmbedder:
        if not api_key:
            raise ProviderError(
                "no GEMINI_API_KEY — vector memory stays off and retrieval "
                "falls back to keyword search"
            )
        return cls(genai.Client(api_key=api_key), model)

    async def _embed_batch(
        self, texts: list[str], task: Literal["query", "document"]
    ) -> list[list[float]]:
        # genai 2.x takes task_type as a plain string
        task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"
        # free-tier 429s carry a "Please retry in Xs" hint — wait it out a
        # couple of times, same as the chat provider
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = await self.client.aio.models.embed_content(
                    model=self.model,
                    contents=texts,
                    config=gtypes.EmbedContentConfig(
                        output_dimensionality=self.dims, task_type=task_type
                    ),
                )
                return [list(e.values) for e in response.embeddings]
            except Exception as exc:
                last_exc = exc
                delay = _retry_delay(exc)
                if delay is None or attempt == 2:
                    break
                await asyncio.sleep(delay)
        raise ProviderError(f"gemini embedding failed: {last_exc}") from last_exc

    async def embed(
        self, texts: list[str], task: Literal["query", "document"] = "document"
    ) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            out.extend(await self._embed_batch(texts[start : start + _BATCH], task))
        return out
