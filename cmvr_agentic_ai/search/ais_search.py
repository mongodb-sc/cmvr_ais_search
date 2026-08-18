"""ais_search tool: hybrid search + rerank over ais_rules, code-aware."""

from __future__ import annotations

from typing import Any

import config
from search import embeddings
from search.hybrid import hybrid_rerank, snippet

_PROJECTION = {
    "_id": 1,
    "AIS_id": 1,
    "AIS": 1,
    "rule": 1,
    "heading": 1,
    "subheading": 1,
    "description": 1,
    "source_file": 1,
}


def _rerank_text(doc: dict[str, Any]) -> str:
    header = " ".join(
        part
        for part in (doc.get("AIS_id", ""), doc.get("heading", ""), doc.get("subheading") or "")
        if part
    )
    return f"{header}\n{doc.get('description', '')}"


def ais_search(
    query: str,
    ais_codes: list[str] | None = None,
    *,
    top_k: int = config.AIS_TOP_K,
    candidate_limit: int = config.CANDIDATE_LIMIT,
) -> dict[str, Any]:
    """Hybrid ($rankFusion vector + text) search over ais_rules, reranked.

    Restricts results to clauses whose own standard (``AIS_id``) is in
    ``ais_codes`` (a hard pre-filter on both the vector and text pipelines) and
    ranks them by semantic + lexical relevance to ``query``. When ``ais_codes``
    is empty, no filter is applied and the whole collection is searched.
    ``further_ais_refs`` lists newly discovered cross-references (from the ``AIS``
    array) so the agent can decide whether to loop again.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query is required")
    codes = [code.strip() for code in (ais_codes or []) if code and code.strip()]

    query_vector = embeddings.embed_query(query)

    should: list[dict[str, Any]] = [
        {"text": {"query": query, "path": "description"}},
        {"text": {"query": query, "path": "heading", "score": {"boost": {"value": 2}}}},
        {"text": {"query": query, "path": "subheading", "score": {"boost": {"value": 2}}}},
    ]
    compound: dict[str, Any] = {"should": should, "minimumShouldMatch": 1}
    vector_filter: dict[str, Any] | None = None
    if codes:
        # Hard pre-filter both retrieval pipelines to AIS_id in the given codes.
        compound["filter"] = [{"in": {"path": "AIS_id", "value": codes}}]
        vector_filter = {"AIS_id": {"$in": codes}}

    docs = hybrid_rerank(
        config.ais_collection(),
        query=query,
        query_vector=query_vector,
        vector_index=config.AIS_TEXT_VECTOR_INDEX,
        vector_path=config.AIS_TEXT_VECTOR_FIELD,
        text_search_stage={
            "index": config.AIS_LEXICAL_INDEX,
            "compound": compound,
        },
        projection=_PROJECTION,
        rerank_text=_rerank_text,
        candidate_limit=candidate_limit,
        rerank_top_k=top_k,
        vector_filter=vector_filter,
    )

    seen = set(codes)
    clauses: list[dict[str, Any]] = []
    further: set[str] = set()
    for doc in docs:
        clauses.append(
            {
                "AIS_id": doc.get("AIS_id", ""),
                "rule": doc.get("rule", ""),
                "heading": doc.get("heading", ""),
                "subheading": doc.get("subheading"),
                "description_snippet": snippet(doc.get("description", "")),
                "rerank_score": round(doc["_rerank_score"], 4),
            }
        )
        for ref in doc.get("AIS") or []:
            if ref and ref not in seen:
                further.add(ref)

    return {"clauses": clauses, "further_ais_refs": sorted(further)}
