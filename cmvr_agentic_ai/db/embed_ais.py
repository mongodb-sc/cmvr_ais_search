"""Generate voyage-4-large embeddings for ais_rules.description.

Run:  python cmvr_agentic_ai/db/embed_ais.py

Writes a 1024-dim ``descriptionEmbedding`` plus ``embeddingMetadata`` to every
ais_rules document that has a non-empty ``description``. Idempotent by default:
documents that already have an embedding are skipped unless ``--overwrite``.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from typing import Any, Iterator

from pymongo import ASCENDING, UpdateOne

import config
from search import embeddings

_PAGE_SIZE = 256


def _token_batches(texts: list[str]) -> Iterator[tuple[list[int], list[str]]]:
    """Yield (indices, texts) batches that respect Voyage's request token budget."""
    token_counts = embeddings.count_tokens(texts)
    indices: list[int] = []
    batch: list[str] = []
    batch_tokens = 0
    for i, (text, tokens) in enumerate(zip(texts, token_counts)):
        capped = min(tokens, config.EMBEDDING_PER_DOCUMENT_TOKEN_LIMIT)
        if batch and batch_tokens + capped > config.EMBEDDING_REQUEST_TOKEN_LIMIT:
            yield indices, batch
            indices, batch, batch_tokens = [], [], 0
        indices.append(i)
        batch.append(text)
        batch_tokens += capped
    if batch:
        yield indices, batch


def _pages(cursor: Any, size: int) -> Iterator[list[dict[str, Any]]]:
    page: list[dict[str, Any]] = []
    for doc in cursor:
        page.append(doc)
        if len(page) >= size:
            yield page
            page = []
    if page:
        yield page


def generate(overwrite: bool = False) -> dict[str, int]:
    collection = config.ais_collection()
    query: dict[str, Any] = {"description": {"$type": "string", "$ne": ""}}
    if not overwrite:
        query["descriptionEmbedding"] = {"$exists": False}

    total = collection.count_documents(query)
    print(f"ais_rules to embed: {total} (overwrite={overwrite})")
    if total == 0:
        return {"total": 0, "embedded": 0}

    cursor = collection.find(query, {"description": 1}).sort("_id", ASCENDING)
    embedded = 0
    for page in _pages(cursor, _PAGE_SIZE):
        texts = [doc["description"] for doc in page]
        vectors: list[list[float] | None] = [None] * len(page)
        for indices, batch in _token_batches(texts):
            batch_vectors = embeddings.embed_documents(batch)
            for position, vector in zip(indices, batch_vectors):
                vectors[position] = vector

        now = datetime.now(timezone.utc)
        operations = [
            UpdateOne(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        config.AIS_TEXT_VECTOR_FIELD: vector,
                        "embeddingMetadata": {
                            "model": config.EMBEDDING_MODEL,
                            "dimensions": config.EMBEDDING_DIMENSION,
                            "inputType": "document",
                            "field": "description",
                            "updatedAt": now,
                        },
                    }
                },
            )
            for doc, vector in zip(page, vectors)
            if vector is not None
        ]
        if operations:
            result = collection.bulk_write(operations, ordered=False)
            embedded += result.modified_count
        print(f"  embedded {embedded}/{total}")

    return {"total": total, "embedded": embedded}


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed ais_rules.description with voyage-4-large")
    parser.add_argument("--overwrite", action="store_true", help="Re-embed even if present")
    args = parser.parse_args()
    summary = generate(overwrite=args.overwrite)
    print(f"Done: {summary}")


if __name__ == "__main__":
    main()
