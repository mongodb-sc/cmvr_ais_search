"""Idempotent Atlas Search index setup for both collections.

Run:  python cmvr_agentic_ai/db/indexes.py

- cmvr_rules indexes already exist on the live cluster; they are verified/skipped.
- ais_rules indexes (vector on descriptionEmbedding + lexical) are created here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Any

from pymongo.operations import SearchIndexModel

import config


def _existing(collection: Any) -> dict[str, dict[str, Any]]:
    return {index.get("name"): index for index in collection.list_search_indexes()}


def ensure_cmvr_indexes() -> list[str]:
    """CMVR indexes already exist; report status, create only if missing."""
    collection = config.cmvr_collection()
    existing = _existing(collection)
    models: list[SearchIndexModel] = []

    if config.CMVR_TEXT_VECTOR_INDEX not in existing:
        models.append(
            SearchIndexModel(
                name=config.CMVR_TEXT_VECTOR_INDEX,
                type="vectorSearch",
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": config.CMVR_TEXT_VECTOR_FIELD,
                            "numDimensions": config.EMBEDDING_DIMENSION,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "documentId"},
                    ]
                },
            )
        )
    if config.CMVR_LEXICAL_INDEX not in existing:
        models.append(
            SearchIndexModel(
                name=config.CMVR_LEXICAL_INDEX,
                definition={
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "canonicalTitle": {"type": "string", "analyzer": "lucene.english"},
                            "ruleText": {"type": "string", "analyzer": "lucene.english"},
                            "documentId": {"type": "token"},
                        },
                    }
                },
            )
        )
    return list(collection.create_search_indexes(models)) if models else []


def ensure_ais_indexes() -> list[str]:
    """Create the AIS vector + lexical indexes if they do not exist yet."""
    collection = config.ais_collection()
    existing = _existing(collection)
    models: list[SearchIndexModel] = []

    if config.AIS_TEXT_VECTOR_INDEX not in existing:
        models.append(
            SearchIndexModel(
                name=config.AIS_TEXT_VECTOR_INDEX,
                type="vectorSearch",
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": config.AIS_TEXT_VECTOR_FIELD,
                            "numDimensions": config.EMBEDDING_DIMENSION,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "AIS_id"},
                    ]
                },
            )
        )
    if config.AIS_LEXICAL_INDEX not in existing:
        models.append(
            SearchIndexModel(
                name=config.AIS_LEXICAL_INDEX,
                definition={
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "description": {"type": "string", "analyzer": "lucene.english"},
                            "heading": {"type": "string", "analyzer": "lucene.english"},
                            "subheading": {"type": "string", "analyzer": "lucene.english"},
                            # token fields power exact code matching via `equals`.
                            "AIS_id": {"type": "token"},
                            "AIS": {"type": "token"},
                        },
                    }
                },
            )
        )
    return list(collection.create_search_indexes(models)) if models else []


def main() -> None:
    cmvr_created = ensure_cmvr_indexes()
    ais_created = ensure_ais_indexes()
    print(f"cmvr_rules: created {cmvr_created or 'nothing (already present)'}")
    print(f"ais_rules:  created {ais_created or 'nothing (already present)'}")
    print("\nCurrent search indexes:")
    for label, collection in (
        ("cmvr_rules", config.cmvr_collection()),
        ("ais_rules", config.ais_collection()),
    ):
        for index in collection.list_search_indexes():
            print(
                f"  [{label}] {index.get('name')}: "
                f"type={index.get('type', 'search')} "
                f"status={index.get('status')} queryable={index.get('queryable')}"
            )


if __name__ == "__main__":
    main()
