"""Embeddings: the free model that makes memory semantically searchable.

One protocol, one adapter today. `GeminiEmbedder` rides the same AI Studio
key and google-genai SDK as the chat provider, on `gemini-embedding-001`
(free tier) stored at 768 Matryoshka dims — Google's recommended
quality/size point. A future adapter (e.g. Jina's free tier) only has to
implement `Embedder`.

Failures here must never cost the caller anything: indexing skips a batch
and retries via backfill, retrieval reports the failure plainly.
"""

from __future__ import annotations

import asyncio
import math
import struct
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


def pack_vector(values: list[float]) -> bytes:
    """Float32 little-endian — the on-disk width REAL[] had, so encrypting
    the vector costs the same storage the plain column did."""
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes) -> list[float]:
    count, remainder = divmod(len(blob), 4)
    if remainder:
        raise ValueError(f"packed vector has a partial float: {len(blob)} bytes")
    return list(struct.unpack(f"<{count}f", blob))


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
                "no GEMINI_API_KEY — vector memory stays off and memory search is unavailable"
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
        # batches fly concurrently — a 200-page pdf is four requests, not
        # four round trips. gather preserves the input order, so callers
        # can still zip results against their parts
        batches = [texts[start : start + _BATCH] for start in range(0, len(texts), _BATCH)]
        results = await asyncio.gather(*(self._embed_batch(batch, task) for batch in batches))
        return [vector for batch in results for vector in batch]
