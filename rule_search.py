"""Embedding, Atlas hybrid retrieval, and Voyage reranking for CMVR rules."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import voyageai
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel


TITLE_VECTOR_FIELD = "canonicalTitleEmbedding"
TEXT_VECTOR_FIELD = "ruleTextEmbedding"
TITLE_VECTOR_INDEX = "cmvr_title_vector"
TEXT_VECTOR_INDEX = "cmvr_rule_text_vector"
LEXICAL_INDEX = "cmvr_rules_lexical"
DEFAULT_EMBEDDING_MODEL = "voyage-4-large"
SUPPORTED_EMBEDDING_MODELS = (
    "voyage-4-large",
    "voyage-4",
    "voyage-4-lite",
)
EMBEDDING_REQUEST_TOKEN_LIMITS = {
    "voyage-4-large": 120_000,
    "voyage-4": 320_000,
    "voyage-4-lite": 1_000_000,
}
EMBEDDING_DIMENSION = 1_024
RERANK_MODEL = "rerank-2.5"
RRF_K = 60
MAX_EMBEDDING_INPUT_TOKENS = 32_000
MAX_RERANK_DOCUMENT_CHARS = 24_000


@dataclass(frozen=True)
class EmbeddingSummary:
    embedding_model: str
    total_documents: int
    selected_documents: int
    updated_documents: int
    skipped_documents: int
    title_vectors: int
    text_vectors: int
    indexes_created: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    rank: int
    canonical_key: str
    rule_number: str
    canonical_title: str
    chapter_id: str
    status: str
    rerank_score: float
    fused_score: float
    title_vector_rank: int | None
    text_vector_rank: int | None
    lexical_rank: int | None
    rule_text: str
    ais: dict[str, str]


class RuleSearchService:
    """Own CMVR rule vectors, Atlas Search indexes, hybrid fusion, and reranking."""

    def __init__(
        self,
        *,
        mongo_uri: str,
        database_name: str,
        collection_name: str,
        voyage_api_key: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        timeout_seconds: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        if not mongo_uri:
            raise ValueError("MONGODB_URI is required")
        if not voyage_api_key:
            raise ValueError("VOYAGE_API_KEY is required")
        if embedding_model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                "embedding_model must be one of: "
                + ", ".join(SUPPORTED_EMBEDDING_MODELS)
            )

        self.mongo_client: MongoClient[dict[str, Any]] = MongoClient(
            mongo_uri,
            appname="cmvr-rule-hybrid-search",
            serverSelectionTimeoutMS=10_000,
            tz_aware=True,
        )
        self.mongo_client.admin.command("ping")
        self.collection: Collection[dict[str, Any]] = self.mongo_client[
            database_name
        ][collection_name]
        self.embedding_model = embedding_model
        self.embedding_request_token_limit = EMBEDDING_REQUEST_TOKEN_LIMITS[
            embedding_model
        ]
        self.voyage_client = voyageai.Client(
            api_key=voyage_api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def close(self) -> None:
        self.mongo_client.close()

    def collection_stats(self) -> dict[str, int]:
        return {
            "total": self.collection.count_documents({}),
            "with_rule_text": self.collection.count_documents(
                {"ruleText": {"$type": "string", "$ne": ""}}
            ),
            "with_title_embedding": self.collection.count_documents(
                {f"{TITLE_VECTOR_FIELD}.0": {"$exists": True}}
            ),
            "with_text_embedding": self.collection.count_documents(
                {f"{TEXT_VECTOR_FIELD}.0": {"$exists": True}}
            ),
        }

    def list_search_indexes(self) -> list[dict[str, Any]]:
        return list(self.collection.list_search_indexes())

    def ensure_search_indexes(
        self,
        fields: Sequence[str] = ("canonicalTitle", "ruleText"),
    ) -> tuple[str, ...]:
        existing = {
            index.get("name"): index for index in self.collection.list_search_indexes()
        }
        models: list[SearchIndexModel] = []

        if "canonicalTitle" in fields and TITLE_VECTOR_INDEX not in existing:
            models.append(
                SearchIndexModel(
                    name=TITLE_VECTOR_INDEX,
                    type="vectorSearch",
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": TITLE_VECTOR_FIELD,
                                "numDimensions": EMBEDDING_DIMENSION,
                                "similarity": "cosine",
                            },
                            {"type": "filter", "path": "documentId"},
                        ]
                    },
                )
            )
        if "ruleText" in fields and TEXT_VECTOR_INDEX not in existing:
            models.append(
                SearchIndexModel(
                    name=TEXT_VECTOR_INDEX,
                    type="vectorSearch",
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": TEXT_VECTOR_FIELD,
                                "numDimensions": EMBEDDING_DIMENSION,
                                "similarity": "cosine",
                            },
                            {"type": "filter", "path": "documentId"},
                        ]
                    },
                )
            )
        if LEXICAL_INDEX not in existing:
            models.append(
                SearchIndexModel(
                    name=LEXICAL_INDEX,
                    definition={
                        "mappings": {
                            "dynamic": False,
                            "fields": {
                                "canonicalTitle": {
                                    "type": "string",
                                    "analyzer": "lucene.english",
                                },
                                "ruleText": {
                                    "type": "string",
                                    "analyzer": "lucene.english",
                                },
                                "documentId": {"type": "token"},
                            },
                        }
                    },
                )
            )

        if not models:
            return ()
        return tuple(self.collection.create_search_indexes(models))

    def search_index_status(self) -> list[dict[str, Any]]:
        return [
            {
                "name": index.get("name", ""),
                "type": index.get("type", "search"),
                "status": index.get("status", "UNKNOWN"),
                "queryable": bool(index.get("queryable", False)),
            }
            for index in self.collection.list_search_indexes()
        ]

    def wait_for_search_indexes(
        self,
        names: Sequence[str],
        *,
        timeout_seconds: float = 300,
        poll_seconds: float = 2,
    ) -> list[dict[str, Any]]:
        required = set(names)
        deadline = time.monotonic() + timeout_seconds
        while True:
            statuses = self.search_index_status()
            ready = {
                item["name"]
                for item in statuses
                if item["queryable"] and item["status"] in {"READY", "STEADY"}
            }
            if required <= ready:
                return statuses
            failed = [
                item for item in statuses if item["name"] in required and item["status"] == "FAILED"
            ]
            if failed:
                raise RuntimeError(f"Atlas Search index build failed: {failed}")
            if time.monotonic() >= deadline:
                return statuses
            time.sleep(poll_seconds)

    def create_embeddings(
        self,
        *,
        overwrite: bool = False,
        create_indexes: bool = True,
        batch_size: int = 32,
        document_id: str | None = "cmvr-1989",
        fields: Sequence[str] = ("canonicalTitle", "ruleText"),
        progress: Callable[[int, int], None] | None = None,
    ) -> EmbeddingSummary:
        if batch_size < 1 or batch_size > 1_000:
            raise ValueError("batch_size must be between 1 and 1000")
        selected_fields = tuple(dict.fromkeys(fields))
        allowed_fields = {"canonicalTitle", "ruleText"}
        if not selected_fields or not set(selected_fields) <= allowed_fields:
            raise ValueError("Select canonicalTitle, ruleText, or both")

        query: dict[str, Any] = {
            "canonicalTitle": {"$type": "string", "$ne": ""},
            "ruleText": {"$type": "string", "$ne": ""},
        }
        if document_id:
            query["documentId"] = document_id
        if not overwrite:
            stale_conditions: list[dict[str, Any]] = []
            if "canonicalTitle" in selected_fields:
                stale_conditions.append(
                    {f"{TITLE_VECTOR_FIELD}.0": {"$exists": False}}
                )
            if "ruleText" in selected_fields:
                stale_conditions.append(
                    {f"{TEXT_VECTOR_FIELD}.0": {"$exists": False}}
                )
            stale_conditions.extend(
                [
                    {"embeddingMetadata.model": {"$ne": self.embedding_model}},
                    {"embeddingMetadata.dimensions": {"$ne": EMBEDDING_DIMENSION}},
                ]
            )
            query["$or"] = stale_conditions

        total_documents = self.collection.count_documents(
            {"documentId": document_id} if document_id else {}
        )
        selected_documents = self.collection.count_documents(query)
        updated_documents = 0
        title_vectors = 0
        text_vectors = 0

        projection = {
            "canonicalTitle": 1,
            "ruleText": 1,
            "canonicalKey": 1,
        }
        cursor = self.collection.find(query, projection).sort("_id", 1)
        for batch in _batched(cursor, batch_size):
            title_embeddings = (
                self._embed_documents(
                    [document["canonicalTitle"] for document in batch]
                )
                if "canonicalTitle" in selected_fields
                else [None] * len(batch)
            )
            text_embeddings = (
                self._embed_documents([document["ruleText"] for document in batch])
                if "ruleText" in selected_fields
                else [None] * len(batch)
            )
            now = datetime.now(timezone.utc)
            operations: list[UpdateOne] = []
            for document, title_embedding, text_embedding in zip(
                batch, title_embeddings, text_embeddings, strict=True
            ):
                set_fields: dict[str, Any] = {
                    "embeddingMetadata.model": self.embedding_model,
                    "embeddingMetadata.dimensions": EMBEDDING_DIMENSION,
                    "embeddingMetadata.inputType": "document",
                    "embeddingMetadata.updatedAt": now,
                }
                if title_embedding is not None:
                    set_fields[TITLE_VECTOR_FIELD] = title_embedding
                    set_fields["embeddingMetadata.fields.canonicalTitle"] = now
                if text_embedding is not None:
                    set_fields[TEXT_VECTOR_FIELD] = text_embedding
                    set_fields["embeddingMetadata.fields.ruleText"] = now
                operations.append(
                    UpdateOne({"_id": document["_id"]}, {"$set": set_fields})
                )
            result = self.collection.bulk_write(operations, ordered=False)
            updated_documents += result.modified_count
            title_vectors += sum(vector is not None for vector in title_embeddings)
            text_vectors += sum(vector is not None for vector in text_embeddings)
            if progress:
                processed = max(title_vectors, text_vectors)
                progress(processed, selected_documents)

        created = self.ensure_search_indexes(selected_fields) if create_indexes else ()
        return EmbeddingSummary(
            embedding_model=self.embedding_model,
            total_documents=total_documents,
            selected_documents=selected_documents,
            updated_documents=updated_documents,
            skipped_documents=max(0, total_documents - selected_documents),
            title_vectors=title_vectors,
            text_vectors=text_vectors,
            indexes_created=created,
        )

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_limit: int = 30,
        vector_weight: float = 0.70,
        document_id: str | None = "cmvr-1989",
    ) -> list[SearchResult]:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Enter a search query")
        if top_k < 1 or top_k > 50:
            raise ValueError("top_k must be between 1 and 50")
        if candidate_limit < top_k or candidate_limit > 200:
            raise ValueError("candidate_limit must be between top_k and 200")

        if not math.isfinite(vector_weight) or not 0 <= vector_weight <= 1:
            raise ValueError("vector_weight must be between 0 and 1")
        lexical_weight = 1 - vector_weight
        text_hits: list[dict[str, Any]] = []
        lexical_hits: list[dict[str, Any]] = []
        if vector_weight > 0:
            query_vector = self._embed_query(clean_query)
            text_hits = self._vector_search(
                query_vector,
                path=TEXT_VECTOR_FIELD,
                index=TEXT_VECTOR_INDEX,
                limit=candidate_limit,
                document_id=document_id,
            )
        if lexical_weight > 0:
            lexical_hits = self._lexical_search(
                clean_query,
                limit=candidate_limit,
                document_id=document_id,
            )

        result_sets: dict[str, Sequence[dict[str, Any]]] = {}
        source_weights: dict[str, float] = {}
        if vector_weight > 0:
            result_sets["vector"] = text_hits
            source_weights["vector"] = vector_weight
        if lexical_weight > 0:
            result_sets["lexical"] = lexical_hits
            source_weights["lexical"] = lexical_weight

        fused = reciprocal_rank_fusion(result_sets, source_weights, rrf_k=RRF_K)
        candidates = fused[:candidate_limit]
        if not candidates:
            return []

        rerank_documents = [
            _rerank_document(item["document"]) for item in candidates
        ]
        reranked = self.voyage_client.rerank(
            clean_query,
            rerank_documents,
            model=RERANK_MODEL,
            top_k=min(top_k, len(candidates)),
            truncation=True,
        )

        results: list[SearchResult] = []
        for rank, rerank_result in enumerate(reranked.results, start=1):
            fused_item = candidates[rerank_result.index]
            document = fused_item["document"]
            ranks = fused_item["ranks"]
            results.append(
                SearchResult(
                    rank=rank,
                    canonical_key=document.get("canonicalKey", str(document["_id"])),
                    rule_number=document.get("ruleNumber", ""),
                    canonical_title=document.get("canonicalTitle", ""),
                    chapter_id=document.get("chapterId", ""),
                    status=document.get("status", ""),
                    rerank_score=float(rerank_result.relevance_score),
                    fused_score=float(fused_item["score"]),
                    title_vector_rank=None,
                    text_vector_rank=ranks.get("vector"),
                    lexical_rank=ranks.get("lexical"),
                    rule_text=document.get("ruleText", ""),
                    ais=document.get("AIS") or {},
                )
            )
        return results

    def _embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        tokenized = self.voyage_client.tokenize(
            list(texts), model=self.embedding_model
        )
        if len(tokenized) != len(texts):
            raise ValueError("Voyage tokenizer result count does not match input count")

        embeddings: list[list[float]] = []
        batch: list[str] = []
        batch_tokens = 0
        for text, tokens in zip(texts, tokenized, strict=True):
            token_count = len(tokens)
            if token_count > MAX_EMBEDDING_INPUT_TOKENS:
                raise ValueError(
                    f"Embedding input has {token_count} tokens; maximum is "
                    f"{MAX_EMBEDDING_INPUT_TOKENS}"
                )
            if (
                batch
                and batch_tokens + token_count > self.embedding_request_token_limit
            ):
                embeddings.extend(self._embed_document_batch(batch))
                batch = []
                batch_tokens = 0
            batch.append(text)
            batch_tokens += token_count
        if batch:
            embeddings.extend(self._embed_document_batch(batch))
        return embeddings

    def _embed_document_batch(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.voyage_client.embed(
            list(texts),
            model=self.embedding_model,
            input_type="document",
            truncation=False,
            output_dtype="float",
            output_dimension=EMBEDDING_DIMENSION,
        )
        return [[float(value) for value in vector] for vector in response.embeddings]

    def _embed_query(self, query: str) -> list[float]:
        response = self.voyage_client.embed(
            [query],
            model=self.embedding_model,
            input_type="query",
            truncation=False,
            output_dtype="float",
            output_dimension=EMBEDDING_DIMENSION,
        )
        return [float(value) for value in response.embeddings[0]]

    def _vector_search(
        self,
        query_vector: list[float],
        *,
        path: str,
        index: str,
        limit: int,
        document_id: str | None,
    ) -> list[dict[str, Any]]:
        vector_stage: dict[str, Any] = {
            "index": index,
            "path": path,
            "queryVector": query_vector,
            "numCandidates": min(max(limit * 10, 100), 10_000),
            "limit": limit,
        }
        if document_id:
            vector_stage["filter"] = {"documentId": document_id}
        pipeline = [
            {"$vectorSearch": vector_stage},
            {"$set": {"sourceScore": {"$meta": "vectorSearchScore"}}},
            {"$project": _search_projection()},
        ]
        return list(self.collection.aggregate(pipeline))

    def _lexical_search(
        self,
        query: str,
        *,
        limit: int,
        document_id: str | None,
    ) -> list[dict[str, Any]]:
        compound: dict[str, Any] = {
            "should": [
                {
                    "text": {
                        "query": query,
                        "path": "canonicalTitle",
                        "score": {"boost": {"value": 3}},
                    }
                },
                {"text": {"query": query, "path": "ruleText"}},
            ],
            "minimumShouldMatch": 1,
        }
        if document_id:
            compound["filter"] = [
                {"equals": {"path": "documentId", "value": document_id}}
            ]
        pipeline = [
            {"$search": {"index": LEXICAL_INDEX, "compound": compound}},
            {"$limit": limit},
            {"$set": {"sourceScore": {"$meta": "searchScore"}}},
            {"$project": _search_projection()},
        ]
        return list(self.collection.aggregate(pipeline))


def reciprocal_rank_fusion(
    result_sets: dict[str, Sequence[dict[str, Any]]],
    weights: dict[str, float],
    *,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked result lists while retaining source ranks."""

    fused: dict[Any, dict[str, Any]] = {}
    for source_name, documents in result_sets.items():
        weight = weights.get(source_name, 0.0)
        for rank, document in enumerate(documents, start=1):
            document_id = document["_id"]
            item = fused.setdefault(
                document_id,
                {"document": document, "score": 0.0, "ranks": {}},
            )
            item["score"] += weight / (rrf_k + rank)
            item["ranks"][source_name] = rank
            if len(document.get("ruleText", "")) > len(
                item["document"].get("ruleText", "")
            ):
                item["document"] = document
    return sorted(fused.values(), key=lambda item: item["score"], reverse=True)


def results_as_rows(results: Sequence[SearchResult]) -> list[list[Any]]:
    return [
        [
            result.rank,
            result.rule_number,
            result.canonical_title,
            result.chapter_id,
            result.status,
            _excerpt(result.rule_text),
            ", ".join(sorted(result.ais)),
        ]
        for result in results
    ]


def _batched(cursor: Any, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for document in cursor:
        batch.append(document)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _normalize_weights(*weights: float) -> tuple[float, ...]:
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Search weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("At least one search weight must be greater than zero")
    return tuple(weight / total for weight in weights)


def _search_projection() -> dict[str, Any]:
    return {
        "_id": 1,
        "canonicalKey": 1,
        "canonicalTitle": 1,
        "chapterId": 1,
        "status": 1,
        "documentId": 1,
        "ruleNumber": 1,
        "ruleText": 1,
        "AIS": 1,
        "sourceScore": 1,
    }


def _rerank_document(document: dict[str, Any]) -> str:
    rule_text = document.get("ruleText", "")[:MAX_RERANK_DOCUMENT_CHARS]
    return "\n".join(
        (
            f"Rule {document.get('ruleNumber', '')}: {document.get('canonicalTitle', '')}",
            f"Chapter: {document.get('chapterId', '')}",
            "Rule text:",
            rule_text,
        )
    )


def _excerpt(text: str, limit: int = 700) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."
