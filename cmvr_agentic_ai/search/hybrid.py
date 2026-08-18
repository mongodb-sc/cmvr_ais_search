"""Shared hybrid retrieval: MongoDB ``$rankFusion`` (vector + text) then Voyage rerank.

Both tools (``cmvr_search`` and ``ais_search``) reuse ``hybrid_rerank`` so the
``$rankFusion`` aggregation is defined exactly once.
"""

from __future__ import annotations

from typing import Any, Callable

import config
from search import embeddings, rerank


def snippet(text: str, limit: int = config.SNIPPET_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def _vector_pipeline(
    query_vector: list[float],
    *,
    index: str,
    path: str,
    limit: int,
    vector_filter: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    stage: dict[str, Any] = {
        "index": index,
        "path": path,
        "queryVector": query_vector,
        "numCandidates": min(max(limit * 10, 100), 10_000),
        "limit": limit,
    }
    if vector_filter:
        stage["filter"] = vector_filter
    return [{"$vectorSearch": stage}]


def _text_pipeline(search_stage: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    return [{"$search": search_stage}, {"$limit": limit}]


def rank_fusion(
    collection: Any,
    *,
    vector_pipeline: list[dict[str, Any]],
    text_pipeline: list[dict[str, Any]],
    limit: int,
    projection: dict[str, Any] | None,
    vector_weight: float = config.VECTOR_WEIGHT,
    text_weight: float = config.LEXICAL_WEIGHT,
) -> list[dict[str, Any]]:
    """Fuse a vector and a full-text pipeline with a single ``$rankFusion`` stage."""
    pipeline: list[dict[str, Any]] = [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "vector": vector_pipeline,
                        "lexical": text_pipeline,
                    }
                },
                "combination": {
                    "weights": {"vector": vector_weight, "lexical": text_weight}
                },
            }
        },
        {"$limit": limit},
    ]
    if projection:
        pipeline.append({"$project": projection})
    return list(collection.aggregate(pipeline))


def hybrid_rerank(
    collection: Any,
    *,
    query: str,
    query_vector: list[float],
    vector_index: str,
    vector_path: str,
    text_search_stage: dict[str, Any],
    projection: dict[str, Any],
    rerank_text: Callable[[dict[str, Any]], str],
    candidate_limit: int = config.CANDIDATE_LIMIT,
    rerank_top_k: int,
    vector_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run hybrid fusion, rerank the fused candidates, return top docs.

    Each returned document carries a ``_rerank_score`` float.
    """
    candidates = rank_fusion(
        collection,
        vector_pipeline=_vector_pipeline(
            query_vector,
            index=vector_index,
            path=vector_path,
            limit=candidate_limit,
            vector_filter=vector_filter,
        ),
        text_pipeline=_text_pipeline(text_search_stage, limit=candidate_limit),
        limit=candidate_limit,
        projection=projection,
    )
    if not candidates:
        return []

    documents = [rerank_text(doc)[: config.MAX_RERANK_DOCUMENT_CHARS] for doc in candidates]
    hits = rerank.rerank(query, documents, top_k=rerank_top_k)

    ranked: list[dict[str, Any]] = []
    for hit in hits:
        doc = dict(candidates[hit.index])
        doc["_rerank_score"] = hit.relevance_score
        ranked.append(doc)
    return ranked
