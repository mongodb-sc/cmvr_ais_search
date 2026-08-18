#!/usr/bin/env python3
"""Gradio interface for the automotive regulatory graph pipeline."""

from __future__ import annotations

import argparse
import contextlib
import html
import io
import logging
import os
import re
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from ingest_regulatory_graph import (
    LOGGER,
    build_argument_parser,
    parse_page_range,
    run_pipeline,
)
from rule_search import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    LEXICAL_INDEX,
    RERANK_MODEL,
    SUPPORTED_EMBEDDING_MODELS,
    TEXT_VECTOR_INDEX,
    TITLE_VECTOR_INDEX,
    RuleSearchService,
    results_as_rows,
)


MODE_INSPECT = "Inspect structure"
MODE_DRY_RUN = "Extract without MongoDB"
MODE_INGEST = "Ingest into MongoDB"

APP_CSS = """
:root {
  --surface: #ffffff;
  --canvas: #f2f5f4;
  --ink: #17211f;
  --muted: #60706c;
  --line: #cad4d1;
  --teal: #176b63;
  --coral: #c7503e;
  --amber: #d69a2d;
}

body,
.gradio-container {
  background-color: var(--canvas) !important;
  background-image:
    linear-gradient(rgba(23, 107, 99, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(23, 107, 99, 0.035) 1px, transparent 1px) !important;
  background-size: 28px 28px !important;
  color: var(--ink) !important;
  font-family: "IBM Plex Sans", sans-serif !important;
  letter-spacing: 0 !important;
}

.gradio-container {
    --block-background-fill: #ffffff !important;
    --block-border-color: #cad4d1 !important;
    --block-label-background-fill: #e8efed !important;
    --block-label-text-color: #34433f !important;
    --block-title-text-color: #17211f !important;
    --body-text-color: #17211f !important;
    --body-text-color-subdued: #60706c !important;
    --button-secondary-background-fill: #e4e9e7 !important;
    --button-secondary-background-fill-hover: #d7dfdc !important;
    --button-secondary-text-color: #17211f !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #b9c5c1 !important;
    --input-placeholder-color: #788581 !important;
  max-width: 1320px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
  margin: 0 auto !important;
  padding: 24px !important;
}

#masthead {
  align-items: flex-end;
  border-bottom: 3px solid var(--ink);
  display: flex;
  justify-content: space-between;
  margin-bottom: 22px;
  padding: 6px 0 18px;
}

#masthead .eyebrow {
  color: var(--coral);
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
  font-weight: 500;
  margin: 0 0 7px;
  text-transform: uppercase;
}

#masthead h1 {
  color: var(--ink);
  font-size: 29px;
  font-weight: 600;
  line-height: 1.08;
  margin: 0;
}

#model-mark {
  align-items: center;
  background: var(--ink);
  border-left: 5px solid var(--amber);
  color: #ffffff;
  display: flex;
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
  gap: 12px;
  min-height: 38px;
  padding: 0 14px;
  white-space: nowrap;
}

#model-mark span {
  color: #9fd2ca;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
  border-color: var(--line) !important;
  border-radius: 6px !important;
  box-shadow: none !important;
}

.gradio-container label,
.gradio-container .label-wrap {
  font-family: "IBM Plex Sans", sans-serif !important;
  letter-spacing: 0 !important;
}

.gradio-container button.label-wrap {
    color: var(--ink) !important;
}

.gradio-container input[type="checkbox"] {
    background: #ffffff !important;
    border-color: #788581 !important;
}

.gradio-container input[type="checkbox"]:checked {
    background: var(--teal) !important;
    border-color: var(--teal) !important;
}

.gradio-container tr.file {
    background: #eef3f1 !important;
    color: var(--ink) !important;
}

.gradio-container tr.file td.filename {
    color: var(--ink) !important;
}

#run-button {
  background: var(--teal) !important;
  border: 1px solid var(--teal) !important;
  border-radius: 4px !important;
  min-height: 46px;
}

#clear-button {
  border-radius: 4px !important;
  min-height: 46px;
}

#run-status {
  background: var(--surface);
  border-left: 5px solid var(--amber);
  border-radius: 0 4px 4px 0;
  min-height: 52px;
  padding: 12px 16px;
}

#run-status p {
  margin: 0;
}

#pipeline-output textarea,
#pipeline-output pre,
#pipeline-output code {
  font-family: "IBM Plex Mono", monospace !important;
  font-size: 12px !important;
  letter-spacing: 0 !important;
}

.gradio-container .tabs,
.gradio-container .tabitem,
.gradio-container .column {
    box-sizing: border-box !important;
    max-width: 100% !important;
    min-width: 0 !important;
    width: 100% !important;
}

.results-table-wrap {
    border: 1px solid var(--line);
    border-radius: 5px;
    max-width: 100%;
    overflow-x: auto;
}

.results-table {
    border-collapse: collapse;
    color: var(--ink);
    font-family: "IBM Plex Mono", monospace;
    font-size: 12px;
    min-width: 1,500px;
    width: 100%;
}

.results-table th:nth-child(3),
.results-table td:nth-child(3) {
    min-width: 240px;
}

.results-table th:nth-child(4),
.results-table td:nth-child(4) {
    min-width: 180px;
}

.results-table th:nth-child(6),
.results-table td:nth-child(6) {
    min-width: 460px;
}

.results-table th:nth-child(7),
.results-table td:nth-child(7) {
    min-width: 160px;
}

.results-table th,
.results-table td {
    border-bottom: 1px solid var(--line);
    padding: 10px 12px;
    text-align: left;
    vertical-align: top;
}

.results-table th {
    background: #dfe9e6;
    font-weight: 500;
    white-space: nowrap;
}

.results-table td {
    background: #ffffff;
}

.results-table tr:nth-child(even) td {
    background: #f5f8f7;
}

.results-table .numeric {
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}

.tab-nav button {
    font-weight: 600 !important;
}

@media (max-width: 720px) {
  .gradio-container {
    padding: 14px !important;
  }

  #masthead {
    align-items: flex-start;
    flex-direction: column;
    gap: 14px;
  }

  #masthead h1 {
    font-size: 24px;
  }
}
"""

APP_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
"""


def _text(value: str | None) -> str:
    return (value or "").strip()


def _uploaded_path(uploaded_file: Any) -> Path:
    raw_path = (
        uploaded_file
        if isinstance(uploaded_file, (str, Path))
        else getattr(uploaded_file, "name", None)
    )
    if not raw_path:
        raise ValueError("Select a PDF before running the pipeline")

    path = Path(raw_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("The selected file must be a PDF")
    if not path.is_file():
        raise ValueError("The selected PDF is no longer available")
    return path


def build_pipeline_arguments(
    *,
    uploaded_file: Any,
    mode: str,
    document_name: str | None,
    document_kind: str,
    pages: str | None,
    ocr: bool,
    max_chunks: float | int | None,
    continue_on_error: bool,
    openai_api_key: str | None,
    voyage_api_key: str | None,
    mongo_uri: str | None,
    database_name: str | None,
    transactions: bool,
) -> argparse.Namespace:
    """Build the same argument namespace used by the production CLI."""

    path = _uploaded_path(uploaded_file)
    parser = build_argument_parser()
    args = parser.parse_args([str(path)])

    if mode not in {MODE_INSPECT, MODE_DRY_RUN, MODE_INGEST}:
        raise ValueError(f"Unsupported run mode: {mode}")

    page_value = _text(pages)
    if page_value:
        try:
            args.pages = parse_page_range(page_value)
        except argparse.ArgumentTypeError as error:
            raise ValueError(str(error)) from error
    else:
        args.pages = None

    chunk_limit: int | None = None
    if max_chunks is not None:
        chunk_limit = int(max_chunks)
        if chunk_limit < 1:
            raise ValueError("Chunk limit must be at least 1 or left blank")

    args.document_name = _text(document_name) or None
    args.document_kind = document_kind
    args.ocr = bool(ocr)
    args.max_chunks = chunk_limit
    args.inspect_only = mode == MODE_INSPECT
    args.dry_run = mode == MODE_DRY_RUN
    args.continue_on_error = bool(continue_on_error)

    args.openai_api_key = _text(openai_api_key) or os.getenv("OPENAI_API_KEY")
    args.voyage_api_key = _text(voyage_api_key) or os.getenv("VOYAGE_API_KEY")
    args.mongo_uri = _text(mongo_uri) or os.getenv("MONGODB_URI")
    args.database = (
        _text(database_name)
        or os.getenv("MONGODB_DATABASE")
        or "automotive_regulations"
    )
    args.transactions = bool(transactions)

    # Keep the UI on the repository's verified model and vector shape.
    args.embedding_model = "voyage-4-large"
    args.embedding_dimension = 1_024
    return args


def _redact(text: str, secrets: list[str | None]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return re.sub(
        r"(mongodb(?:\+srv)?://)([^@/\s]+)@",
        r"\1[REDACTED]@",
        redacted,
        flags=re.IGNORECASE,
    )


def run_from_ui(
    uploaded_file: Any,
    mode: str,
    document_name: str,
    document_kind: str,
    pages: str,
    ocr: bool,
    max_chunks: float | int | None,
    continue_on_error: bool,
    openai_api_key: str,
    voyage_api_key: str,
    mongo_uri: str,
    database_name: str,
    transactions: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str]:
    """Validate UI input, invoke the pipeline, and return sanitized output."""

    progress(0.02, desc="Validating request")
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    previous_level = LOGGER.level
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(handler)

    supplied_secrets = [
        _text(openai_api_key),
        _text(voyage_api_key),
        _text(mongo_uri),
    ]
    try:
        args = build_pipeline_arguments(
            uploaded_file=uploaded_file,
            mode=mode,
            document_name=document_name,
            document_kind=document_kind,
            pages=pages,
            ocr=ocr,
            max_chunks=max_chunks,
            continue_on_error=continue_on_error,
            openai_api_key=openai_api_key,
            voyage_api_key=voyage_api_key,
            mongo_uri=mongo_uri,
            database_name=database_name,
            transactions=transactions,
        )
        supplied_secrets.extend(
            [args.openai_api_key, args.voyage_api_key, args.mongo_uri]
        )
        progress(0.08, desc="Running regulatory pipeline")
        with contextlib.redirect_stdout(output):
            exit_code = run_pipeline(args)
        if exit_code != 0:
            raise RuntimeError(f"Pipeline exited with status {exit_code}")

        progress(1.0, desc="Complete")
        status = f"**Completed**  \n{mode} finished successfully."
    except Exception as error:
        safe_error = _redact(f"{type(error).__name__}: {error}", supplied_secrets)
        output.write(f"\nERROR: {safe_error}\n")
        status = f"**Failed**  \n{safe_error}"
    finally:
        LOGGER.removeHandler(handler)
        LOGGER.setLevel(previous_level)

    rendered_output = _redact(output.getvalue().strip(), supplied_secrets)
    return status, rendered_output or "No pipeline output was produced."


def create_rule_embeddings_ui(
    fields: list[str],
    embedding_model: str,
    overwrite: bool,
    create_indexes: bool,
    wait_for_indexes: bool,
    batch_size: float | int,
    document_id: str,
    mongo_uri: str,
    database_name: str,
    collection_name: str,
    voyage_api_key: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, dict[str, Any], str]:
    """Embed selected CMVR rule fields and provision Atlas Search indexes."""

    service: RuleSearchService | None = None
    secrets = [_text(mongo_uri), _text(voyage_api_key)]
    try:
        clean_fields = list(fields or [])
        clean_document_id = _text(document_id) or None
        service = RuleSearchService(
            mongo_uri=_text(mongo_uri) or os.getenv("MONGODB_URI", ""),
            database_name=_text(database_name)
            or os.getenv("MONGODB_DATABASE", "automotive_regulations"),
            collection_name=_text(collection_name)
            or os.getenv("CMVR_RULE_COLLECTION", "cmvr_rules"),
            voyage_api_key=_text(voyage_api_key)
            or os.getenv("VOYAGE_API_KEY", ""),
            embedding_model=embedding_model,
        )
        progress(0.03, desc="Checking stored rules")

        def report(completed: int, total: int) -> None:
            fraction = 0.05 + (0.75 * completed / max(total, 1))
            progress(fraction, desc=f"Embedded {completed}/{total} rules")

        summary = service.create_embeddings(
            fields=clean_fields,
            overwrite=bool(overwrite),
            create_indexes=bool(create_indexes),
            batch_size=int(batch_size),
            document_id=clean_document_id,
            progress=report,
        )
        statuses = service.search_index_status()
        if create_indexes and wait_for_indexes:
            required = [LEXICAL_INDEX]
            if "canonicalTitle" in clean_fields:
                required.append(TITLE_VECTOR_INDEX)
            if "ruleText" in clean_fields:
                required.append(TEXT_VECTOR_INDEX)
            progress(0.85, desc="Waiting for Atlas Search indexes")
            statuses = service.wait_for_search_indexes(required)

        stats = service.collection_stats()
        payload = {
            "embeddingModel": summary.embedding_model,
            "dimensions": EMBEDDING_DIMENSION,
            "fields": clean_fields,
            "totalDocuments": summary.total_documents,
            "selectedDocuments": summary.selected_documents,
            "updatedDocuments": summary.updated_documents,
            "skippedDocuments": summary.skipped_documents,
            "titleVectorsGenerated": summary.title_vectors,
            "ruleTextVectorsGenerated": summary.text_vectors,
            "indexesCreated": list(summary.indexes_created),
            "collectionStats": stats,
        }
        status_table = _render_index_table(statuses)
        progress(1.0, desc="Embedding job complete")
        return (
            "**Completed**  \nEmbeddings and index provisioning finished.",
            payload,
            status_table,
        )
    except Exception as error:
        safe_error = _redact(f"{type(error).__name__}: {error}", secrets)
        return f"**Failed**  \n{safe_error}", {"error": safe_error}, ""
    finally:
        if service:
            service.close()


def hybrid_rule_search_ui(
    query: str,
    embedding_model: str,
    top_k: float | int,
    candidate_limit: float | int,
    vector_balance: float,
    document_id: str,
    mongo_uri: str,
    database_name: str,
    collection_name: str,
    voyage_api_key: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str]:
    """Run dual-vector plus lexical retrieval and Voyage reranking."""

    service: RuleSearchService | None = None
    secrets = [_text(mongo_uri), _text(voyage_api_key)]
    try:
        service = RuleSearchService(
            mongo_uri=_text(mongo_uri) or os.getenv("MONGODB_URI", ""),
            database_name=_text(database_name)
            or os.getenv("MONGODB_DATABASE", "automotive_regulations"),
            collection_name=_text(collection_name)
            or os.getenv("CMVR_RULE_COLLECTION", "cmvr_rules"),
            voyage_api_key=_text(voyage_api_key)
            or os.getenv("VOYAGE_API_KEY", ""),
            embedding_model=embedding_model,
        )
        stats = service.collection_stats()
        normalized_vector_weight = float(vector_balance) / 100
        if normalized_vector_weight > 0 and stats["with_text_embedding"] < stats["total"]:
            raise RuntimeError("Rule-text embeddings are incomplete; run Tab 1 first")

        required_indexes: set[str] = set()
        if normalized_vector_weight > 0:
            required_indexes.add(TEXT_VECTOR_INDEX)
        if normalized_vector_weight < 1:
            required_indexes.add(LEXICAL_INDEX)
        ready_indexes = {
            item["name"]
            for item in service.search_index_status()
            if item["queryable"] and item["status"] in {"READY", "STEADY"}
        }
        missing_indexes = required_indexes - ready_indexes
        if missing_indexes:
            raise RuntimeError(
                "Atlas Search indexes are not ready: " + ", ".join(sorted(missing_indexes))
            )

        progress(0.1, desc="Embedding query and retrieving candidates")
        results = service.hybrid_search(
            query,
            top_k=int(top_k),
            candidate_limit=int(candidate_limit),
            vector_weight=normalized_vector_weight,
            document_id=_text(document_id) or None,
        )
        progress(1.0, desc="Reranking complete")
        details = _format_search_details(results)
        return (
            f"**Completed**  \nReturned {len(results)} reranked rules using {RERANK_MODEL}.",
            _render_search_table(results),
            details,
        )
    except Exception as error:
        safe_error = _redact(f"{type(error).__name__}: {error}", secrets)
        return f"**Failed**  \n{safe_error}", "", ""
    finally:
        if service:
            service.close()


def _format_search_details(results: Sequence[Any]) -> str:
    if not results:
        return "No matching rules found."
    sections: list[str] = []
    for result in results:
        ais_codes = ", ".join(sorted(result.ais)) or "None"
        excerpt = html.escape(" ".join(result.rule_text.split())[:1_500])
        sections.append(
            "\n".join(
                (
                    f"### {result.rank}. Rule {html.escape(result.rule_number)}: "
                    f"{html.escape(result.canonical_title)}",
                    f"**Chapter:** {html.escape(result.chapter_id)}  ",
                    f"**Status:** {html.escape(result.status)}  ",
                    f"**AIS:** {html.escape(ais_codes)}",
                    "",
                    excerpt + ("..." if len(result.rule_text) > 1_500 else ""),
                )
            )
        )
    return "\n\n---\n\n".join(sections)


def _render_index_table(statuses: Sequence[dict[str, Any]]) -> str:
    rows = [
        [item["name"], item["type"], item["status"], str(item["queryable"])]
        for item in statuses
    ]
    return _render_html_table(["Index", "Type", "Status", "Queryable"], rows)


def _render_search_table(results: Sequence[Any]) -> str:
    return _render_html_table(
        [
            "Rank",
            "Rule",
            "Title",
            "Chapter",
            "Status",
            "Rule text excerpt",
            "AIS",
        ],
        results_as_rows(results),
        numeric_columns={0},
    )


def _render_html_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    numeric_columns: set[int] | None = None,
) -> str:
    if not rows:
        return '<div class="results-table-wrap"><p style="padding:12px">No rows.</p></div>'
    numeric = numeric_columns or set()
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows: list[str] = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            css_class = ' class="numeric"' if index in numeric else ""
            rendered = "" if value is None else html.escape(str(value))
            cells.append(f"<td{css_class}>{rendered}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="results-table-wrap"><table class="results-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def build_app() -> gr.Blocks:
    """Build the two-tab CMVR embedding and hybrid-search application."""

    with gr.Blocks(
        title="Automotive Regulatory Graph",
        fill_width=True,
        analytics_enabled=False,
    ) as app:
        gr.HTML(
            """
            <header id="masthead">
              <div>
                <p class="eyebrow">UST / Regulatory intelligence</p>
                <h1>Automotive Regulatory Graph</h1>
              </div>
              <div id="model-mark">VOYAGE 4 SERIES <span>1024D</span></div>
            </header>
            """
        )

        with gr.Accordion("MongoDB and Voyage settings", open=False):
            with gr.Row():
                mongo_uri = gr.Textbox(
                    label="MongoDB URI",
                    type="password",
                    placeholder="Uses MONGODB_URI when blank",
                )
                voyage_api_key = gr.Textbox(
                    label="Voyage API key",
                    type="password",
                    placeholder="Uses VOYAGE_API_KEY when blank",
                )
            with gr.Row():
                database_name = gr.Textbox(
                    value=os.getenv("MONGODB_DATABASE", "automotive_regulations"),
                    label="Database",
                )
                collection_name = gr.Textbox(
                    value=os.getenv("CMVR_RULE_COLLECTION", "cmvr_rules"),
                    label="Collection",
                )
                document_id = gr.Textbox(value="cmvr-1989", label="Document ID filter")

        embedding_model = gr.Dropdown(
            choices=[
                ("Voyage 4 Large — highest retrieval quality", "voyage-4-large"),
                ("Voyage 4 — balanced quality and cost", "voyage-4"),
                ("Voyage 4 Lite — lowest latency and cost", "voyage-4-lite"),
            ],
            value=os.getenv("VOYAGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            label="Voyage embedding and query model",
            info="All listed Voyage 4 models are compatible with the shared 1024D Atlas indexes.",
        )

        with gr.Tabs():
            with gr.Tab("1. Embeddings"):
                gr.Markdown(
                    "Create independent Voyage vectors for `canonicalTitle` and "
                    "`ruleText`, then provision the Atlas vector and lexical indexes."
                )
                fields = gr.CheckboxGroup(
                    choices=[
                        ("Canonical title", "canonicalTitle"),
                        ("Rule text", "ruleText"),
                    ],
                    value=["canonicalTitle", "ruleText"],
                    label="Fields to embed",
                )
                with gr.Row():
                    overwrite = gr.Checkbox(value=False, label="Recreate existing vectors")
                    create_indexes = gr.Checkbox(value=True, label="Create missing Atlas indexes")
                    wait_for_indexes = gr.Checkbox(value=True, label="Wait until indexes are queryable")
                    batch_size = gr.Number(value=16, minimum=1, maximum=128, precision=0, label="Batch size")
                embed_button = gr.Button("Create embeddings", variant="primary", elem_id="run-button")
                embed_status = gr.Markdown("**Ready**", elem_id="run-status")
                embed_summary = gr.JSON(label="Embedding summary")
                gr.Markdown("**Atlas Search indexes**")
                index_status = gr.HTML()
                embed_button.click(
                    fn=create_rule_embeddings_ui,
                    inputs=[
                        fields,
                        embedding_model,
                        overwrite,
                        create_indexes,
                        wait_for_indexes,
                        batch_size,
                        document_id,
                        mongo_uri,
                        database_name,
                        collection_name,
                        voyage_api_key,
                    ],
                    outputs=[embed_status, embed_summary, index_status],
                    concurrency_limit=1,
                    api_visibility="private",
                )

            with gr.Tab("2. Hybrid Search"):
                gr.Markdown(
                    "Fuse full `ruleText` vector search with Atlas lexical search, "
                    f"then rerank candidates with `{RERANK_MODEL}`."
                )
                search_query = gr.Textbox(
                    label="Search CMVR rules",
                    placeholder="Example: emission limits for diesel vehicles",
                    lines=2,
                )
                with gr.Row():
                    top_k = gr.Number(value=5, minimum=1, maximum=50, precision=0, label="Results")
                    candidate_limit = gr.Number(value=30, minimum=5, maximum=200, precision=0, label="Candidates before reranking")
                vector_balance = gr.Slider(
                    0,
                    100,
                    value=70,
                    step=5,
                    label="Lexical ↔ vector balance",
                    info="0 = lexical only, 100 = ruleText vector only. The vector branch always uses ruleText.",
                )
                search_button = gr.Button("Search and rerank", variant="primary", elem_id="run-button")
                search_status = gr.Markdown("**Ready**", elem_id="run-status")
                gr.Markdown("**Reranked results**")
                search_results = gr.HTML()
                search_details = gr.Markdown(label="Result details")
                search_button.click(
                    fn=hybrid_rule_search_ui,
                    inputs=[
                        search_query,
                        embedding_model,
                        top_k,
                        candidate_limit,
                        vector_balance,
                        document_id,
                        mongo_uri,
                        database_name,
                        collection_name,
                        voyage_api_key,
                    ],
                    outputs=[search_status, search_results, search_details],
                    concurrency_limit=1,
                    api_visibility="private",
                    scroll_to_output=True,
                )

    return app


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Launch the Gradio pipeline UI")
    parser.add_argument(
        "--server-name",
        default=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
    parser.add_argument("--inbrowser", action="store_true")
    launch_args = parser.parse_args()

    app = build_app().queue(default_concurrency_limit=1, max_size=8)
    app.launch(
        server_name=launch_args.server_name,
        server_port=launch_args.server_port,
        inbrowser=launch_args.inbrowser,
        share=False,
        show_error=False,
        max_file_size="100mb",
        enable_monitoring=False,
        strict_cors=True,
        footer_links=[],
        css=APP_CSS,
        head=APP_HEAD,
    )


if __name__ == "__main__":
    main()