# Automotive Regulatory PDF Structural Findings

## Scope and environment

The analysis was executed on 22 July 2026 with:

- Python 3.12.11
- Docling 2.114.0
- docling-core 2.87.1
- `DocumentConverter` with accurate table structure recognition
- `HierarchicalChunker(always_emit_headings=True)`

The embedded text of all 325 source pages was scanned for representative
features. Detailed Docling conversion was then run on the following ranges:

| Document | Pages | Structural feature |
| --- | ---: | --- |
| `cmvr-1989.pdf` | 66 | Chapter V transition, Rule 91, definitions, list items, and footnote |
| `cmvr-1989.pdf` | 95-98 | Rules 110-115, chained provisos, emission tables, formulas, and footnotes |
| `cmvr-1989.pdf` | 121-122 | Tables containing Rule, AIS, and IS cross-references |
| `AIS-057_Rev_1_with Amd_57fb9704-c742-49c8-8cbc-5637bb2d783d.pdf` | 20-22 | Annexes G/H, nested clauses, formulas, and photometric tables |

## 1. Document structure

### CMVR chapters, rules, sub-rules, and provisos

Docling correctly detected `CHAPTER V CONSTRUCTION, EQUIPMENT AND MAINTENANCE
OF MOTOR VEHICLES` and `Preliminary` as `SectionHeaderItem` objects. It did not
preserve their semantic relationship: both were level-1 headers. Enabling
Docling's legal heading hierarchy inference produced the same flat levels.

Rule titles were usually not headers. For example, `91. Definitions` and
`115. Emission of smoke...` were `TextItem` objects. Lettered definitions were
a mixture of `TextItem` and grouped `ListItem` objects. Sub-rules such as `(2)`
were embedded in text, rather than represented as structural children.

Provisos were plain text. `Provided that`, `Provided further that`, and
`Provided also that` appeared as separate chunks in the Rule 112/115 sample,
but had no explicit parent link to the rule or sub-rule they qualify.

Amendment notes were generally classified as `footnote`, which is useful for
filtering, but they inherited the latest heading metadata just like body text.

One source list group joined the end of Rule 91 and the opening of Rule 92 in a
single chunk. This is why the ingestion script splits embedded rule boundaries
before assigning context.

### AIS annexes and clauses

`ANNEX G`, `ANNEX H`, table titles, and some numbered clauses were detected as
section headers, but every detected header remained level 1. Other clauses were
list items. For example:

- `G-3. CIL values` was a section header.
- `G-3.1`, `G-3.1.1`, and `G-3.2` were list items.
- `H-1.1. Water submersion test` was a section header.
- `H-1.2...` content remained under stale `H-1.1` chunk metadata.

Setting `merge_list_items=False` did not separate `G-3.1.2`, `G.3.1.3`, and
`G-3.2`; the three clauses were already serialized into one source group. The
custom clause-boundary splitter separates them before extraction.

### Tables and formulas

Docling retained tables as `TableItem` objects and exposed them as data frames.
Results depended on header complexity:

- The CMVR petrol/CNG/LPG table became a clean 5-row by 4-column grid. Vehicle
  type, CO percentage, and HC limits remained associated with each row.
- The CMVR diesel table became a 1-row by 3-column grid. Its multi-row header
  was flattened, and OCR/text extraction rendered `Hartridge` as `Mar tidge`.
- AIS-057 Table G1 became a 2-row by 6-column grid.
- AIS-057 Table G-2 became a 3-row by 9-column grid. White, amber, and red values
  remained associated with the corresponding illumination-angle columns.

`HierarchicalChunker` serializes a table into repeated row-and-column statements,
which is suitable for LLM extraction and embeddings. The script marks these
chunks as `content_type=table` in the LLM context.

Formula handling was inconsistent. In CMVR, `RHC=0.5 x HC` was incorrectly
classified as a section header. In AIS-057, one formula was a `FormulaItem` but
serialized as `<!-- formula-not-decoded -->`. Numeric relationships involving
formulas therefore require source-page review when they are safety-critical.

### Cross-reference forms

The PDFs use several equivalent surface forms:

- Rules: `rule 124`, `Rule 126A`, and numbered definitions such as `115.`
- AIS: `AIS-031`, `AIS: 031`, `AIS -031`, `AIS 052`, and `AIS-007(Rev.5)`
- IS: `IS:14557`, `IS 2553`, `IS:2553 (Part-2)`, `IS 13944:1995`, and
  `IS : 11865 - 1992`

Docling preserved most forms in text and table chunks. It occasionally inserted
internal spacing, such as `A IS 025`; the normalizer accepts this variant.

## 2. Hierarchical chunking evaluation

The raw `HierarchicalChunker` results were:

- CMVR page 66: 8 chunks, including a heading-only Chapter V chunk.
- CMVR pages 95-98: 38 chunks, including 2 table chunks.
- AIS-057 pages 20-22: 28 chunks, including 2 table chunks.

Headings were available in `chunk.meta.headings`, but they did not provide the
required paths by themselves:

- After the Chapter V heading chunk, Rule 91 chunks contained only
  `['Preliminary']`, not `CHAPTER V > Rule 91`.
- Rule 115 chunks contained the topical heading `Emission of smoke...`, not a
  normalized Rule 115 parent.
- AIS `H-1.2` content retained `H-1.1. Water submersion test` as its heading.
- CMVR provisos after the misclassified RHC formula inherited `RHC=0.5 x HC`.

The production strategy is therefore:

1. Use Docling for layout, reading order, tables, provenance, and base chunks.
2. Treat `chunk.meta.headings` as advisory metadata.
3. Track chapter and annex transitions with explicit patterns.
4. Detect CMVR rule definitions, sub-rules, AIS clauses, and proviso markers in
   chunk text.
5. Split merged rule or clause boundaries before assigning context.
6. Prefix the LLM and embedding input with the repaired hierarchy while storing
   the raw Docling chunk text in MongoDB.

This produced the intended contextual paths in the validation runs, including:

- `CHAPTER V: ... > Rule 91`
- `CHAPTER V: ... > Rule 92 > Sub-rule (1)`
- `ANNEX G > Clause G-3.1.2`
- `ANNEX G > Clause G-3.1.3`
- `ANNEX G > Clause G-3.2`

## 3. Entity ID taxonomy

Entity IDs use uppercase ASCII tokens separated by one underscore. Publication
years, revisions, parts, dates, limits, and units remain properties instead of
creating separate aliases.

| Source forms | Canonical `entity_id` | Label |
| --- | --- | --- |
| Central Motor Vehicles Rules, CMVR 1989 | `CMVR_1989` | `Regulation` |
| Rule 115B, Rule 115 B, Rule 115-B, R. 115(B) | `RULE_115_B` | `Regulation` |
| AIS-024, AIS 024, AIS:024, AIS-024 (Rev.1) | `AIS_024` | `Standard` |
| IS:14557, IS 14557, IS:14557-1999 | `IS_14557` | `Standard` |
| M3, class M3, category M3 | `CATEGORY_M3` | `VehicleClass` |
| cooling system | `COOLING_SYSTEM` | `Component` |

Sub-rule and clause paths are stored in chunk/entity properties. They are not
appended to a base rule entity unless a later graph design explicitly models
sub-rules as independent regulation nodes.

## 4. Pipeline operation

Python 3.11 or newer is required because the script uses `StrEnum`. The verified
environment uses Python 3.12 and the versions in `requirements.txt`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Inspect repaired chunks without an LLM or MongoDB:

```bash
.venv/bin/python ingest_regulatory_graph.py cmvr-1989.pdf \
  --pages 95-98 --inspect-only --no-ocr
```

For page-scoped runs, the script parses page 1 through the requested end to
recover earlier chapter/annex transitions, but emits and processes only chunks
whose provenance overlaps the requested range. This avoids silently losing the
Chapter V context when a range begins at Rule 115.

Run extraction with Voyage AI `voyage-4-large` embeddings and MongoDB:

```bash
.venv/bin/python ingest_regulatory_graph.py cmvr-1989.pdf \
  --document-name "Central Motor Vehicles Rules, 1989"
```

Set `VOYAGE_API_KEY` before ingestion. The embedding adapter follows the
[MongoDB Voyage AI Python reference](https://www.mongodb.com/docs/voyageai/models/text-embeddings/?client=python):

- Model: `voyage-4-large`
- Input type: `document`, optimized for retrieval
- Output: float vectors with 1,024 dimensions by default
- Context: 32,000 tokens per chunk
- Request limit: 1,000 inputs and 120,000 aggregate tokens
- Truncation: disabled so regulatory text is never silently discarded

The SDK tokenizer enforces the individual and aggregate token limits before an
API request and splits batches without changing chunk order. The dimension can
be set to 256, 512, 1,024, or 2,048 with `--embedding-dimension`.

Existing MiniLM vectors are 384-dimensional and cannot be mixed with the new
1,024-dimensional vectors. Rebuild the `chunks` collection and its Atlas Vector
Search index, or ingest into a new database, before switching an existing
deployment. The writer rejects mixed dimensions.

For scanned documents, add `--ocr`. For a standalone MongoDB server, leave
transactions disabled. Atlas or replica-set deployments can use `--transactions`
to make each chunk/node/edge write atomic.

The script validates secrets only when the corresponding runtime stage starts.
This keeps imports, tests, and `--inspect-only` usable without API credentials.
It also creates unique node/edge indexes, retries transient LLM failures, checks
embedding dimensions, and records every relationship's source chunk ObjectId.
Voyage SDK retries use the configured `--api-attempts` count and
`--voyage-timeout` value.

OpenAI Structured Outputs does not accept arbitrary-key JSON objects. The LLM
schema therefore represents entity and relationship properties as closed arrays
of `{key, value}` objects. The ingestion boundary deterministically converts
those arrays back into the required MongoDB `properties` dictionaries. A schema
regression test verifies that every generated object has
`additionalProperties: false` and requires exactly its declared fields.

MongoDB currently marks its Voyage AI Embedding and Reranking API as a Preview
feature. Review that status before using it for a production deployment.

## 5. Validation performed

- Real Docling conversions succeeded for all representative CMVR and AIS ranges
  listed above.
- Real context assertions passed for Chapter V, Rule 115 tables and provisos,
  merged Rule 91/92 content, AIS merged clauses, and the Annex I to Rule 116/117
  transition.
- Forty-one automated tests pass. They cover strict Pydantic validation, entity
  aliases, amendment exclusion, context repair, page filtering, indexes, node
  and edge upserts, ObjectId lineage accumulation using `mongomock`, and the
  Voyage model parameters, token-aware batching, overlength rejection, Gradio
  argument mapping, secret redaction, callback execution, app construction,
  field-specific rule embeddings, reciprocal-rank fusion, and reranking.
- Python compilation, dependency consistency, CLI construction, and VS Code
  diagnostics were checked.

A live minimal `gpt-4.1-mini` request confirmed that OpenAI accepts the generated
`ChunkExtractionResult` schema and returns parsed entities and relationships.
The exact Voyage SDK call is validated with a mocked client; MongoDB write
semantics are exercised through the PyMongo-compatible in-memory integration
test.

## 6. Gradio UI

Launch the local interface with:

```bash
.venv/bin/python gradio_app.py --inbrowser
```

The server binds to `127.0.0.1:7860` by default and never creates a public share
link. `GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT` can override those values.
The interface contains exactly two operational tabs:

1. **Create Embeddings** generates independent 1,024-dimensional
  vectors for `canonicalTitle` and `ruleText`. A shared dropdown offers
  `voyage-4-large` (highest quality), `voyage-4` (balanced), and
  `voyage-4-lite` (lowest latency/cost). Users can select either field or both,
  skip current vectors or overwrite them, and provision the required Atlas
  indexes.
2. **Hybrid Search + Rerank** retrieves candidates from the rule-text-vector
  index and boosted Atlas lexical index. The vector branch
  always uses the complete `ruleText`; title-vector retrieval is not part of
  search ranking. One slider controls the balance from lexical-only (`0`) to
  ruleText-vector-only (`100`), with weighted reciprocal rank fusion between
  those endpoints. Voyage `rerank-2.5` produces the final order. Visible results
  include rule status but hide internal fusion scores and source ranks.

MongoDB URIs and Voyage keys are password-masked; blank fields fall back to
`.env`, and callback errors are redacted before display.

All dropdown models belong to the Voyage 4 series and produce mutually
compatible 1,024-dimensional vectors. The service applies each model's documented
aggregate request limit: 120K tokens for `voyage-4-large`, 320K for `voyage-4`,
and 1M for `voyage-4-lite`. Choosing a different model in Tab 1 refreshes stored
vectors and embedding metadata; choosing it in Tab 2 uses that model for the
query vector against the compatible index.

Embedding and search jobs share a single-concurrency queue so database writes,
Atlas retrieval, and model calls do not overlap.

The live `cmvr_rules` deployment has 177 title vectors and 177 rule-text vectors.
The Atlas indexes are:

- `cmvr_title_vector` on `canonicalTitleEmbedding`
- `cmvr_rule_text_vector` on `ruleTextEmbedding`
- `cmvr_rules_lexical` on `canonicalTitle` and `ruleText`

A live hybrid validation query for diesel smoke-emission testing returned Rule
115 first and Rule 116 second after `rerank-2.5`.

## 7. CMVR Rule and AIS Notebook

`extract_cmvr_rules_ais.ipynb` reuses the shared Docling converter and repaired
CMVR context tracker to build one canonical document per rule. It retains rule
continuations, tables, footnotes, and annexes until the next genuine legal rule
header, then extracts every source-cited AIS code and a source-grounded
description with OpenAI Structured Outputs. Each canonical document also stores
the complete normalized `ruleText`; validation requires it to exactly match the
grouped `RuleCandidate.source_text` before MongoDB writes are allowed.

The notebook defaults to `RUN_LLM_EXTRACTION=False` and
`WRITE_TO_MONGODB=False`. Review the 177 detected rule candidates, set a small
`MAX_RULES` for an initial paid run, and enable extraction. Results are appended
to `cmvr_rule_ais_checkpoint.jsonl`, so interrupted runs resume without repeating
completed calls. Enable MongoDB writes only after the validation tables are
clean; upserts use the canonical string `_id` and preserve source pages and
timestamps. Existing checkpoint records without `ruleText` are backfilled from
the parsed PDF and rewritten atomically, so AIS descriptions are preserved and
no additional paid extraction is required for the migration.

Rule 115 was validated live against the supplied CMVR PDF. It maps to Chapter V
and cites AIS-025, AIS-026, AIS-027, AIS-054, and AIS-055 in this source. AIS-052
is not cited under Rule 115 in this PDF, so the notebook never inserts it merely
because it appeared in an example schema. The MongoDB read-back check confirmed
that Rule 115 stores 72,921 normalized rule-text characters.