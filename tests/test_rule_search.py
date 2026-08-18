from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from rule_search import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    LEXICAL_INDEX,
    RERANK_MODEL,
    TEXT_VECTOR_FIELD,
    TEXT_VECTOR_INDEX,
    TITLE_VECTOR_FIELD,
    TITLE_VECTOR_INDEX,
    RuleSearchService,
    reciprocal_rank_fusion,
    results_as_rows,
)


class FakeCursor:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents

    def sort(self, *_args: object) -> "FakeCursor":
        return self

    def __iter__(self):
        return iter(self.documents)


class RuleSearchFusionTests(unittest.TestCase):
    def test_weighted_rrf_combines_source_ranks(self) -> None:
        first = {"_id": "rule-1", "canonicalTitle": "One", "ruleText": "text"}
        second = {"_id": "rule-2", "canonicalTitle": "Two", "ruleText": "text"}

        fused = reciprocal_rank_fusion(
            {
                "vector": [second, first],
                "lexical": [second],
            },
            {"vector": 0.6, "lexical": 0.4},
        )

        self.assertEqual(fused[0]["document"]["_id"], "rule-2")
        self.assertEqual(fused[0]["ranks"], {
            "vector": 1,
            "lexical": 1,
        })


class RuleEmbeddingTests(unittest.TestCase):
    def test_embeds_selected_fields_and_creates_filter_indexes(self) -> None:
        service = RuleSearchService.__new__(RuleSearchService)
        documents = [
            {
                "_id": "cmvr-1989:rule:1",
                "canonicalTitle": "Short title",
                "ruleText": "Complete rule text",
            }
        ]
        collection = MagicMock()
        collection.count_documents.side_effect = [1, 1]
        collection.find.return_value = FakeCursor(documents)
        collection.bulk_write.return_value = SimpleNamespace(modified_count=1)
        collection.list_search_indexes.return_value = []
        collection.create_search_indexes.return_value = [
            TITLE_VECTOR_INDEX,
            TEXT_VECTOR_INDEX,
            LEXICAL_INDEX,
        ]
        service.collection = collection
        service.embedding_model = DEFAULT_EMBEDDING_MODEL
        service._embed_documents = MagicMock(
            side_effect=[[[0.1] * EMBEDDING_DIMENSION], [[0.2] * EMBEDDING_DIMENSION]]
        )

        summary = service.create_embeddings(
            fields=("canonicalTitle", "ruleText"),
            batch_size=8,
        )

        self.assertEqual(summary.title_vectors, 1)
        self.assertEqual(summary.text_vectors, 1)
        self.assertEqual(service._embed_documents.call_args_list[0].args[0], ["Short title"])
        self.assertEqual(service._embed_documents.call_args_list[1].args[0], ["Complete rule text"])
        models = collection.create_search_indexes.call_args.args[0]
        definitions = {model.document["name"]: model.document for model in models}
        self.assertEqual(
            definitions[TITLE_VECTOR_INDEX]["definition"]["fields"][1],
            {"type": "filter", "path": "documentId"},
        )
        self.assertEqual(
            definitions[TEXT_VECTOR_INDEX]["definition"]["fields"][1],
            {"type": "filter", "path": "documentId"},
        )

    def test_title_only_does_not_embed_rule_text(self) -> None:
        service = RuleSearchService.__new__(RuleSearchService)
        collection = MagicMock()
        collection.count_documents.side_effect = [1, 1]
        collection.find.return_value = FakeCursor([
            {"_id": "rule-1", "canonicalTitle": "Title", "ruleText": "Text"}
        ])
        collection.bulk_write.return_value = SimpleNamespace(modified_count=1)
        service.collection = collection
        service.embedding_model = DEFAULT_EMBEDDING_MODEL
        service._embed_documents = MagicMock(return_value=[[0.1] * EMBEDDING_DIMENSION])

        summary = service.create_embeddings(
            fields=("canonicalTitle",),
            create_indexes=False,
        )

        service._embed_documents.assert_called_once_with(["Title"])
        self.assertEqual(summary.title_vectors, 1)
        self.assertEqual(summary.text_vectors, 0)


class RuleHybridSearchTests(unittest.TestCase):
    def test_hybrid_search_fuses_then_uses_voyage_rerank(self) -> None:
        service = RuleSearchService.__new__(RuleSearchService)
        service._embed_query = MagicMock(return_value=[0.1] * EMBEDDING_DIMENSION)
        rule_one = {
            "_id": "cmvr-1989:rule:1",
            "canonicalKey": "cmvr-1989:rule:1",
            "ruleNumber": "1",
            "canonicalTitle": "Short title",
            "chapterId": "cmvr-1989:chapter:I",
            "status": "active",
            "ruleText": "Rule one text",
            "AIS": {},
        }
        rule_two = {
            "_id": "cmvr-1989:rule:115",
            "canonicalKey": "cmvr-1989:rule:115",
            "ruleNumber": "115",
            "canonicalTitle": "Emission of smoke",
            "chapterId": "cmvr-1989:chapter:V",
            "status": "active",
            "ruleText": "Emission requirements",
            "AIS": {"AIS-025": "description"},
        }
        service._vector_search = MagicMock(return_value=[rule_two, rule_one])
        service._lexical_search = MagicMock(return_value=[rule_two])
        service.voyage_client = MagicMock()
        service.voyage_client.rerank.return_value = SimpleNamespace(
            results=[SimpleNamespace(index=0, relevance_score=0.95)]
        )

        results = service.hybrid_search(
            "smoke emission", top_k=1, candidate_limit=2, vector_weight=0.8
        )

        self.assertEqual(results[0].rule_number, "115")
        self.assertEqual(results[0].status, "active")
        service._vector_search.assert_called_once()
        self.assertEqual(service._vector_search.call_args.kwargs["path"], TEXT_VECTOR_FIELD)
        service.voyage_client.rerank.assert_called_once()
        call = service.voyage_client.rerank.call_args
        self.assertEqual(call.args[0], "smoke emission")
        self.assertEqual(call.kwargs["model"], RERANK_MODEL)
        self.assertEqual(results_as_rows(results)[0][1], "115")

    def test_lexical_only_skips_query_embedding_and_vector_search(self) -> None:
        service = RuleSearchService.__new__(RuleSearchService)
        document = {
            "_id": "rule-115",
            "canonicalKey": "rule-115",
            "ruleNumber": "115",
            "canonicalTitle": "Emission",
            "chapterId": "chapter-V",
            "status": "active",
            "ruleText": "Smoke emission",
            "AIS": {},
        }
        service._embed_query = MagicMock()
        service._vector_search = MagicMock()
        service._lexical_search = MagicMock(return_value=[document])
        service.voyage_client = MagicMock()
        service.voyage_client.rerank.return_value = SimpleNamespace(
            results=[SimpleNamespace(index=0, relevance_score=0.9)]
        )

        results = service.hybrid_search(
            "smoke", top_k=1, candidate_limit=1, vector_weight=0
        )

        self.assertEqual(results[0].rule_number, "115")
        service._embed_query.assert_not_called()
        service._vector_search.assert_not_called()
        service._lexical_search.assert_called_once()

    def test_vector_only_skips_lexical_search(self) -> None:
        service = RuleSearchService.__new__(RuleSearchService)
        document = {
            "_id": "rule-115",
            "canonicalKey": "rule-115",
            "ruleNumber": "115",
            "canonicalTitle": "Emission",
            "chapterId": "chapter-V",
            "status": "active",
            "ruleText": "Smoke emission",
            "AIS": {},
        }
        service._embed_query = MagicMock(return_value=[0.1] * EMBEDDING_DIMENSION)
        service._vector_search = MagicMock(return_value=[document])
        service._lexical_search = MagicMock()
        service.voyage_client = MagicMock()
        service.voyage_client.rerank.return_value = SimpleNamespace(
            results=[SimpleNamespace(index=0, relevance_score=0.9)]
        )

        results = service.hybrid_search(
            "smoke", top_k=1, candidate_limit=1, vector_weight=1
        )

        self.assertEqual(results[0].rule_number, "115")
        service._vector_search.assert_called_once()
        service._lexical_search.assert_not_called()

    def test_embedding_input_types_are_query_and_document(self) -> None:
        service = RuleSearchService.__new__(RuleSearchService)
        service.embedding_model = "voyage-4-lite"
        service.embedding_request_token_limit = 1_000_000
        service.voyage_client = MagicMock()
        service.voyage_client.tokenize.return_value = [[1, 2]]
        service.voyage_client.embed.side_effect = [
            SimpleNamespace(embeddings=[[0.1] * EMBEDDING_DIMENSION]),
            SimpleNamespace(embeddings=[[0.2] * EMBEDDING_DIMENSION]),
        ]

        service._embed_documents(["document"])
        service._embed_query("query")

        calls = service.voyage_client.embed.call_args_list
        self.assertEqual(calls[0].kwargs["input_type"], "document")
        self.assertEqual(calls[1].kwargs["input_type"], "query")
        self.assertEqual(calls[0].kwargs["model"], "voyage-4-lite")
        self.assertEqual(calls[1].kwargs["model"], "voyage-4-lite")


if __name__ == "__main__":
    unittest.main()
