from __future__ import annotations

import gc
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import gradio as gr

from gradio_app import (
    MODE_DRY_RUN,
    MODE_INGEST,
    MODE_INSPECT,
    _redact,
    build_app,
    build_pipeline_arguments,
    create_rule_embeddings_ui,
    hybrid_rule_search_ui,
    run_from_ui,
)
from rule_search import EmbeddingSummary, SearchResult


class GradioArgumentTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temporary_file.close()
        self.pdf_path = Path(temporary_file.name)

    def tearDown(self) -> None:
        self.pdf_path.unlink(missing_ok=True)

    def test_inspect_mode_needs_no_credentials(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "",
                "VOYAGE_API_KEY": "",
                "MONGODB_URI": "",
            },
            clear=False,
        ):
            args = build_pipeline_arguments(
                uploaded_file=str(self.pdf_path),
                mode=MODE_INSPECT,
                document_name="CMVR 1989",
                document_kind="cmvr",
                pages="95-98",
                ocr=False,
                max_chunks=12,
                continue_on_error=False,
                openai_api_key="",
                voyage_api_key="",
                mongo_uri="",
                database_name="automotive_regulations",
                transactions=False,
            )

        self.assertTrue(args.inspect_only)
        self.assertFalse(args.dry_run)
        self.assertEqual(args.pages, (95, 98))
        self.assertEqual(args.max_chunks, 12)
        self.assertEqual(args.embedding_model, "voyage-4-large")
        self.assertEqual(args.embedding_dimension, 1_024)

    def test_ingest_mode_maps_credentials_without_cli_exposure(self) -> None:
        args = build_pipeline_arguments(
            uploaded_file=self.pdf_path,
            mode=MODE_INGEST,
            document_name="",
            document_kind="auto",
            pages="",
            ocr=True,
            max_chunks=None,
            continue_on_error=True,
            openai_api_key="openai-secret",
            voyage_api_key="voyage-secret",
            mongo_uri="mongodb://user:password@localhost:27017",
            database_name="regulations",
            transactions=True,
        )

        self.assertFalse(args.inspect_only)
        self.assertFalse(args.dry_run)
        self.assertEqual(args.openai_api_key, "openai-secret")
        self.assertEqual(args.voyage_api_key, "voyage-secret")
        self.assertEqual(args.database, "regulations")
        self.assertTrue(args.transactions)

    def test_dry_run_mode_disables_mongo_writes(self) -> None:
        args = build_pipeline_arguments(
            uploaded_file=self.pdf_path,
            mode=MODE_DRY_RUN,
            document_name="AIS-057",
            document_kind="standard",
            pages="20",
            ocr=False,
            max_chunks=2,
            continue_on_error=False,
            openai_api_key="openai-secret",
            voyage_api_key="voyage-secret",
            mongo_uri="",
            database_name="",
            transactions=False,
        )

        self.assertTrue(args.dry_run)
        self.assertFalse(args.inspect_only)


class GradioExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temporary_file.close()
        self.pdf_path = Path(temporary_file.name)

    def tearDown(self) -> None:
        self.pdf_path.unlink(missing_ok=True)

    def _run(self) -> tuple[str, str]:
        return run_from_ui(
            str(self.pdf_path),
            MODE_INSPECT,
            "CMVR 1989",
            "cmvr",
            "1",
            False,
            1,
            False,
            "openai-secret",
            "voyage-secret",
            "mongodb://user:password@localhost:27017",
            "regulations",
            False,
            progress=MagicMock(),
        )

    def test_successful_run_returns_status_and_output(self) -> None:
        with patch("gradio_app.run_pipeline", return_value=0):
            status, output = self._run()

        self.assertIn("Completed", status)
        self.assertNotIn("openai-secret", output)

    def test_failure_redacts_secrets_and_mongo_credentials(self) -> None:
        error = RuntimeError(
            "openai-secret mongodb://user:password@localhost:27017"
        )
        with patch("gradio_app.run_pipeline", side_effect=error):
            status, output = self._run()

        self.assertIn("Failed", status)
        self.assertNotIn("openai-secret", status + output)
        self.assertNotIn("user:password", status + output)
        self.assertIn("[REDACTED]", status + output)

    def test_redact_handles_mongodb_srv_credentials(self) -> None:
        value = _redact("mongodb+srv://user:pass@example.net/db", [])
        self.assertEqual(value, "mongodb+srv://[REDACTED]@example.net/db")


class GradioAppTests(unittest.TestCase):
    def test_app_builds_as_blocks(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            app = build_app()
            try:
                self.assertIsInstance(app, gr.Blocks)
            finally:
                app.close()
                del app
                gc.collect()


class RuleSearchGradioTests(unittest.TestCase):
    def test_embedding_callback_returns_summary_and_indexes(self) -> None:
        service = MagicMock()
        service.create_embeddings.return_value = EmbeddingSummary(
            embedding_model="voyage-4",
            total_documents=177,
            selected_documents=177,
            updated_documents=177,
            skipped_documents=0,
            title_vectors=177,
            text_vectors=177,
            indexes_created=("cmvr_title_vector", "cmvr_rule_text_vector"),
        )
        service.collection_stats.return_value = {
            "total": 177,
            "with_rule_text": 177,
            "with_title_embedding": 177,
            "with_text_embedding": 177,
        }
        service.search_index_status.return_value = []
        service.wait_for_search_indexes.return_value = [
            {
                "name": "cmvr_title_vector",
                "type": "vectorSearch",
                "status": "READY",
                "queryable": True,
            }
        ]

        with patch("gradio_app.RuleSearchService", return_value=service):
            status, summary, indexes = create_rule_embeddings_ui(
                ["canonicalTitle", "ruleText"],
                "voyage-4",
                False,
                True,
                True,
                16,
                "cmvr-1989",
                "mongodb://localhost",
                "db",
                "rules",
                "voyage-secret",
                progress=MagicMock(),
            )

        self.assertIn("Completed", status)
        self.assertEqual(summary["titleVectorsGenerated"], 177)
        self.assertEqual(summary["embeddingModel"], "voyage-4")
        self.assertIn("cmvr_title_vector", indexes)
        service.close.assert_called_once()

    def test_hybrid_search_callback_returns_reranked_rows(self) -> None:
        service = MagicMock()
        service.collection_stats.return_value = {
            "total": 177,
            "with_rule_text": 177,
            "with_title_embedding": 177,
            "with_text_embedding": 177,
        }
        service.search_index_status.return_value = [
            {"name": name, "status": "READY", "queryable": True}
            for name in ("cmvr_title_vector", "cmvr_rule_text_vector", "cmvr_rules_lexical")
        ]
        service.hybrid_search.return_value = [
            SearchResult(
                rank=1,
                canonical_key="cmvr-1989:rule:115",
                rule_number="115",
                canonical_title="Emission of smoke",
                chapter_id="cmvr-1989:chapter:V",
                status="active",
                rerank_score=0.99,
                fused_score=0.03,
                title_vector_rank=2,
                text_vector_rank=1,
                lexical_rank=1,
                rule_text="Every motor vehicle shall comply.",
                ais={"AIS-025": "description"},
            )
        ]

        with patch("gradio_app.RuleSearchService", return_value=service):
            status, rows, details = hybrid_rule_search_ui(
                "diesel smoke emissions",
                "voyage-4-lite",
                5,
                30,
                70,
                "cmvr-1989",
                "mongodb://localhost",
                "db",
                "rules",
                "voyage-secret",
                progress=MagicMock(),
            )

        self.assertIn("Completed", status)
        self.assertIn("115", rows)
        self.assertIn("active", rows)
        self.assertIn("Rule 115", details)
        self.assertNotIn("Rerank", rows)
        self.assertNotIn("Fused", rows)
        self.assertNotIn("Title rank", rows)
        self.assertNotIn("Rerank score", details)
        service.close.assert_called_once()

    def test_search_failure_redacts_credentials(self) -> None:
        with patch(
            "gradio_app.RuleSearchService",
            side_effect=RuntimeError("mongodb://user:pass@host failed"),
        ):
            status, rows, details = hybrid_rule_search_ui(
                "query",
                "voyage-4-large",
                5,
                30,
                70,
                "cmvr-1989",
                "mongodb://user:pass@host",
                "db",
                "rules",
                "voyage-secret",
                progress=MagicMock(),
            )

        self.assertIn("Failed", status)
        self.assertNotIn("user:pass", status)
        self.assertEqual(rows, "")
        self.assertEqual(details, "")


if __name__ == "__main__":
    unittest.main()