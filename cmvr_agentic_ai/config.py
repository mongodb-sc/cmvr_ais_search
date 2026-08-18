"""Central configuration and lazy MongoDB/Voyage handles for the POC.

Every runnable entry point (``app.py``, ``db/*.py``) adds the package root to
``sys.path`` so these flat imports (``import config``) resolve regardless of the
working directory Streamlit or ``python`` is launched from.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# --- MongoDB -----------------------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "automotive_regulations")
CMVR_COLLECTION = os.getenv("CMVR_RULE_COLLECTION", "cmvr_rules")
AIS_COLLECTION = os.getenv("AIS_RULE_COLLECTION", "ais_rules")
HISTORY_COLLECTION = os.getenv("RESEARCH_HISTORY_COLLECTION", "research_history")

# CMVR indexes already exist and are READY on the live cluster; we reuse them.
CMVR_TEXT_VECTOR_INDEX = os.getenv("CMVR_TEXT_VECTOR_INDEX", "cmvr_rule_text_vector")
CMVR_LEXICAL_INDEX = os.getenv("CMVR_LEXICAL_INDEX", "cmvr_rules_lexical")
CMVR_TEXT_VECTOR_FIELD = "ruleTextEmbedding"

# AIS indexes are created by db/indexes.py (they do not exist yet).
AIS_TEXT_VECTOR_INDEX = os.getenv("AIS_TEXT_VECTOR_INDEX", "ais_rule_text_vector")
AIS_LEXICAL_INDEX = os.getenv("AIS_LEXICAL_INDEX", "ais_rules_lexical")
AIS_TEXT_VECTOR_FIELD = "descriptionEmbedding"

# --- Voyage AI ---------------------------------------------------------------
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-4-large")
RERANK_MODEL = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5")
# Confirmed against the live cmvr_rules vectors: 1024 dims, cosine similarity.
EMBEDDING_DIMENSION = 1_024

# Voyage per-request / per-document token budgets for voyage-4-large.
EMBEDDING_REQUEST_TOKEN_LIMIT = 120_000
EMBEDDING_PER_DOCUMENT_TOKEN_LIMIT = 32_000

# --- Grove gateway (Anthropic-compatible) ------------------------------------
GROVE_BASE_URL = os.getenv(
    "GROVE_BASE_URL",
    "https://grove-gateway-prod.azure-api.net/grove-foundry-prod/anthropic/v1/messages",
)
GROVE_MODEL = os.getenv("GROVE_MODEL", "claude-fable-5")

# --- Hybrid search tuning ----------------------------------------------------
VECTOR_WEIGHT = 0.7
LEXICAL_WEIGHT = 0.3
CANDIDATE_LIMIT = 25
CMVR_TOP_K = 6
AIS_TOP_K = 8
MAX_RERANK_DOCUMENT_CHARS = 8_000
SNIPPET_CHARS = 480


@lru_cache(maxsize=1)
def mongo_client() -> Any:
    """Return a process-wide MongoClient, verifying connectivity once."""
    from pymongo import MongoClient

    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is not set")
    client: Any = MongoClient(
        MONGODB_URI,
        appname="cmvr-agentic-ai",
        serverSelectionTimeoutMS=10_000,
        tz_aware=True,
    )
    client.admin.command("ping")
    return client


def cmvr_collection() -> Any:
    return mongo_client()[MONGODB_DATABASE][CMVR_COLLECTION]


def ais_collection() -> Any:
    return mongo_client()[MONGODB_DATABASE][AIS_COLLECTION]


def history_collection() -> Any:
    return mongo_client()[MONGODB_DATABASE][HISTORY_COLLECTION]
