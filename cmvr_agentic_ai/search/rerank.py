"""Voyage AI reranking wrapper for rerank-2.5."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import config


@lru_cache(maxsize=1)
def _client() -> Any:
    import voyageai

    if not config.VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is not set")
    return voyageai.Client(api_key=config.VOYAGE_API_KEY)


@dataclass(frozen=True)
class RerankHit:
    index: int
    relevance_score: float


def rerank(query: str, documents: list[str], *, top_k: int) -> list[RerankHit]:
    """Rerank ``documents`` against ``query``; returns hits sorted best-first.

    ``index`` maps back to the position in the input ``documents`` list.
    """
    if not documents:
        return []
    response = _client().rerank(
        query,
        list(documents),
        model=config.RERANK_MODEL,
        top_k=min(top_k, len(documents)),
        truncation=True,
    )
    return [
        RerankHit(index=int(result.index), relevance_score=float(result.relevance_score))
        for result in response.results
    ]
