"""Voyage AI embedding wrapper for voyage-4-large (query vs document input types).

The Voyage client is created lazily so importing this module never requires a
key (index-only or offline flows stay importable).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import config


@lru_cache(maxsize=1)
def _client() -> Any:
    import voyageai

    if not config.VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is not set")
    return voyageai.Client(api_key=config.VOYAGE_API_KEY)


def embed_query(text: str) -> list[float]:
    """Embed a single search query (input_type='query')."""
    response = _client().embed(
        [text],
        model=config.EMBEDDING_MODEL,
        input_type="query",
        truncation=True,
        output_dtype="float",
        output_dimension=config.EMBEDDING_DIMENSION,
    )
    return [float(value) for value in response.embeddings[0]]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents (input_type='document') in one request.

    Callers are responsible for keeping each request within Voyage's token
    budget (see db/embed_ais.py for token-aware batching).
    """
    if not texts:
        return []
    response = _client().embed(
        list(texts),
        model=config.EMBEDDING_MODEL,
        input_type="document",
        truncation=True,
        output_dtype="float",
        output_dimension=config.EMBEDDING_DIMENSION,
    )
    return [[float(value) for value in vector] for vector in response.embeddings]


def count_tokens(texts: list[str]) -> list[int]:
    """Token counts per text, used for request batching."""
    tokenized = _client().tokenize(list(texts), model=config.EMBEDDING_MODEL)
    return [len(tokens) for tokens in tokenized]
