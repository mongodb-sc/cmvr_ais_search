"""cmvr_search tool: hybrid search + rerank over cmvr_rules with AIS-code extraction."""

from __future__ import annotations

from typing import Any

import config
from search import embeddings
from search.hybrid import hybrid_rerank, snippet

_PROJECTION = {
    "_id": 1,
    "ruleNumber": 1,
    "canonicalTitle": 1,
    "chapterId": 1,
    "ruleText": 1,
    "AIS": 1,
}


def _rerank_text(doc: dict[str, Any]) -> str:
    return f"{doc.get('canonicalTitle', '')}\n{doc.get('ruleText', '')}"


def _ais_codes(doc: dict[str, Any]) -> list[str]:
    ais = doc.get("AIS")
    if isinstance(ais, dict):
        return sorted(ais.keys())
    return []


def cmvr_search(
    query: str,
    vehicle_category: str | None = None,
    *,
    top_k: int = config.CMVR_TOP_K,
    candidate_limit: int = config.CANDIDATE_LIMIT,
) -> dict[str, Any]:
    """Hybrid ($rankFusion vector + text) search over cmvr_rules, reranked.

    Returns matched rules and the union of AIS codes cross-referenced by them.
    ``vehicle_category`` (e.g. "M3", "N2") softly biases the query and lexical
    match toward that category rather than hard-filtering it out.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query is required")

    effective_query = (
        f"{query} (vehicle category {vehicle_category})" if vehicle_category else query
    )
    query_vector = embeddings.embed_query(effective_query)

    should: list[dict[str, Any]] = [
        {"text": {"query": effective_query, "path": "canonicalTitle", "score": {"boost": {"value": 3}}}},
        {"text": {"query": effective_query, "path": "ruleText"}},
    ]
    if vehicle_category:
        should.append(
            {"text": {"query": vehicle_category, "path": "ruleText", "score": {"boost": {"value": 2}}}}
        )

    docs = hybrid_rerank(
        config.cmvr_collection(),
        query=effective_query,
        query_vector=query_vector,
        vector_index=config.CMVR_TEXT_VECTOR_INDEX,
        vector_path=config.CMVR_TEXT_VECTOR_FIELD,
        text_search_stage={
            "index": config.CMVR_LEXICAL_INDEX,
            "compound": {"should": should, "minimumShouldMatch": 1},
        },
        projection=_PROJECTION,
        rerank_text=_rerank_text,
        candidate_limit=candidate_limit,
        rerank_top_k=top_k,
    )

    rules: list[dict[str, Any]] = []
    all_codes: set[str] = set()
    for doc in docs:
        codes = _ais_codes(doc)
        all_codes.update(codes)
        rules.append(
            {
                "ruleNumber": doc.get("ruleNumber", ""),
                "canonicalTitle": doc.get("canonicalTitle", ""),
                "ruleText_snippet": snippet(doc.get("ruleText", "")),
                "ais_codes": codes,
                "rerank_score": round(doc["_rerank_score"], 4),
            }
        )

    return {"rules": rules, "ais_codes": sorted(all_codes)}
