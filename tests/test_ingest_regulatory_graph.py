from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mongomock
from mongomock.collection import BulkOperationBuilder
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from ingest_regulatory_graph import (
    ChunkExtractionResult,
    DocumentKind,
    EntityLabel,
    EntityModel,
    MongoGraphWriter,
    PropertyModel,
    RegulatoryContextTracker,
    RegulatoryChunk,
    RelationType,
    RelationshipModel,
    VoyageAIEmbedder,
    _clean_properties,
    chunk_overlaps_page_range,
    normalize_entity_id,
    normalize_extraction,
    parse_page_range,
    resolve_document_kind,
    split_regulatory_boundaries,
)


class EntityIdTaxonomyTests(unittest.TestCase):
    def test_rule_variants_share_one_id(self) -> None:
        variants = ("Rule 115B", "R. 115(B)", "Rule 115 B", "RULE_115_B")
        self.assertEqual(
            {
                normalize_entity_id(value, EntityLabel.REGULATION)
                for value in variants
            },
            {"RULE_115_B"},
        )

    def test_standard_variants_share_base_ids(self) -> None:
        self.assertEqual(
            {
                normalize_entity_id(value, EntityLabel.STANDARD)
                for value in ("AIS-024", "AIS: 024", "A IS 024 (Rev.1)")
            },
            {"AIS_024"},
        )
        self.assertEqual(
            normalize_entity_id("IS:14557-1999", EntityLabel.STANDARD),
            "IS_14557",
        )

    def test_vehicle_class_and_component_ids(self) -> None:
        self.assertEqual(
            normalize_entity_id("vehicle class M3", EntityLabel.VEHICLE_CLASS),
            "CATEGORY_M3",
        )
        self.assertEqual(
            normalize_entity_id("Cooling system", EntityLabel.COMPONENT),
            "COOLING_SYSTEM",
        )

    def test_amendment_citation_is_not_normalized_as_a_rule(self) -> None:
        self.assertEqual(
            normalize_entity_id("G.S.R. 590(E)", EntityLabel.REGULATION),
            "G_S_R_590_E",
        )

    def test_aliases_are_resolved_across_relationships(self) -> None:
        result = ChunkExtractionResult(
            entities=[
                EntityModel(
                    entity_id="RULE_115B",
                    name="Rule 115-B",
                    label=EntityLabel.REGULATION,
                    properties=[
                        PropertyModel(key="source_form", value="Rule 115B")
                    ],
                ),
                EntityModel(
                    entity_id="RULE_115_B",
                    name="Rule 115-B",
                    label=EntityLabel.REGULATION,
                    properties=[PropertyModel(key="topic", value="emissions")],
                ),
                EntityModel(
                    entity_id="CATEGORY_M3",
                    name="Category M3",
                    label=EntityLabel.VEHICLE_CLASS,
                ),
            ],
            relationships=[
                RelationshipModel(
                    source_entity_id="RULE_115B",
                    target_entity_id="CATEGORY_M3",
                    relation_type=RelationType.APPLIES_TO,
                )
            ],
        )

        normalized = normalize_extraction(result)

        self.assertEqual(
            {entity.entity_id for entity in normalized.entities},
            {"RULE_115_B", "CATEGORY_M3"},
        )
        self.assertEqual(
            normalized.relationships[0].source_entity_id,
            "RULE_115_B",
        )


class ExtractionSchemaTests(unittest.TestCase):
    def test_schema_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            EntityModel.model_validate(
                {
                    "entity_id": "AIS_024",
                    "name": "AIS-024",
                    "label": EntityLabel.STANDARD,
                    "properties": {},
                    "confidence": 0.9,
                }
            )

    def test_schema_rejects_missing_relationship_endpoint(self) -> None:
        with self.assertRaises(ValidationError):
            ChunkExtractionResult(
                entities=[
                    EntityModel(
                        entity_id="RULE_115",
                        name="Rule 115",
                        label=EntityLabel.REGULATION,
                    )
                ],
                relationships=[
                    RelationshipModel(
                        source_entity_id="RULE_115",
                        target_entity_id="AIS_024",
                        relation_type=RelationType.TESTED_BY,
                    )
                ],
            )

    def test_strict_schema_parses_json_enum_values(self) -> None:
        result = ChunkExtractionResult.model_validate_json(
            '{"entities":[{"entity_id":"AIS_024","name":"AIS-024",'
            '"label":"Standard","properties":[]}],"relationships":[]}'
        )
        self.assertEqual(result.entities[0].label, EntityLabel.STANDARD)

    def test_openai_schema_uses_only_closed_required_objects(self) -> None:
        schema = to_strict_json_schema(ChunkExtractionResult)

        def assert_closed(node: object, path: tuple[object, ...] = ()) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object":
                    properties = set(node.get("properties", {}))
                    required = set(node.get("required", []))
                    self.assertIs(
                        node.get("additionalProperties"),
                        False,
                        path,
                    )
                    self.assertEqual(required, properties, path)
                for key, value in node.items():
                    assert_closed(value, (*path, key))
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    assert_closed(value, (*path, index))

        assert_closed(schema)


class RegulatoryContextTests(unittest.TestCase):
    def test_cmvr_chapter_rule_subrule_and_proviso(self) -> None:
        tracker = RegulatoryContextTracker(DocumentKind.CMVR)
        tracker.observe(
            "",
            ["CHAPTER V CONSTRUCTION, EQUIPMENT AND MAINTENANCE"],
        )
        rule = tracker.observe(
            "115. Emission of smoke. -(2) Every motor vehicle shall comply.",
            ["Emission of smoke"],
        )
        proviso = tracker.observe(
            "Provided further that category M3 may use the alternate test:",
            ["RHC=0.5 x HC"],
        )

        self.assertEqual(rule.rule, "Rule 115 > Sub-rule (2)")
        self.assertEqual(
            proviso.rule,
            "Rule 115 > Sub-rule (2) > Further proviso",
        )
        self.assertTrue((rule.chapter or "").startswith("CHAPTER V:"))

    def test_ais_number_is_a_clause_even_with_stale_heading(self) -> None:
        tracker = RegulatoryContextTracker(DocumentKind.STANDARD)
        tracker.observe("(See 7.1)", ["ANNEX G"])
        context = tracker.observe(
            "- G.3.1.3 CIL values for colourless devices shall comply.",
            ["CIL values for red devices"],
        )

        self.assertEqual(context.chapter, "ANNEX G")
        self.assertEqual(context.rule, "Clause G-3.1.3")

    def test_embedded_rule_and_clause_boundaries_are_split(self) -> None:
        cmvr_text = (
            '- (f) "subsidiary risk" means the subsidiary risk.\n'
            "92. General. -(1) Every motor vehicle shall comply."
        )
        ais_text = (
            "- G-3.1.2. Amber devices shall comply.\n"
            "- G.3.1.3 Colourless devices shall comply.\n"
            "- G-3.2. Class IVA devices shall comply."
        )

        self.assertEqual(
            len(
                split_regulatory_boundaries(
                    cmvr_text,
                    DocumentKind.CMVR,
                    inside_annex=False,
                )
            ),
            2,
        )
        self.assertEqual(
            len(
                split_regulatory_boundaries(
                    ais_text,
                    DocumentKind.STANDARD,
                    inside_annex=True,
                )
            ),
            3,
        )

    def test_embedded_and_amended_rule_headers_are_split(self) -> None:
        text = (
            "1. Short title and commencement.-(1) These rules apply. "
            "2. Definitions.-In these rules, context applies.\n"
            "9. Birth certificate,\n"
            "19 [5. Medical certificate.19 [(1) Every application applies."
        )

        pieces = split_regulatory_boundaries(
            text,
            DocumentKind.CMVR,
            inside_annex=False,
        )

        self.assertEqual(len(pieces), 3)
        self.assertTrue(pieces[1].startswith("2. Definitions"))
        self.assertTrue(pieces[2].startswith("19 [5. Medical certificate"))

    def test_next_cmvr_rule_exits_annex_context(self) -> None:
        tracker = RegulatoryContextTracker(DocumentKind.CMVR)
        tracker.observe("115. Emission standards.-(1) Limits apply.", [])
        tracker.observe("", ["ANNEX I"])

        context = tracker.observe(
            "116. Test for smoke emission level.-(1) The test shall apply.",
            [],
        )

        self.assertIsNone(tracker.annex)
        self.assertEqual(context.chapter, None)
        self.assertEqual(context.rule, "Rule 116 > Sub-rule (1)")

    def test_numbered_list_cannot_reset_cmvr_rule_context(self) -> None:
        tracker = RegulatoryContextTracker(DocumentKind.CMVR)
        tracker.observe("31. Syllabus for driving instruction.-(1) Topics follow.", [])

        context = tracker.observe("1. Driving regulations", [])
        forward_list = tracker.observe("38. Substituted by G.S.R. 400(E).", [])
        tracker.observe("", ["CHAPTER III REGISTRATION OF MOTOR VEHICLES"])
        next_rule = tracker.observe("32. Fees.-(1) The fee shall apply.", [])

        self.assertEqual(context.rule, "Rule 31 > Sub-rule (1)")
        self.assertEqual(forward_list.rule, "Rule 31 > Sub-rule (1)")
        self.assertEqual(next_rule.rule, "Rule 32 > Sub-rule (1)")

    def test_hyphenated_annexure_heading_is_recognized(self) -> None:
        tracker = RegulatoryContextTracker(DocumentKind.CMVR)
        context = tracker.observe("", ["ANNEXURE-VIII"])

        self.assertEqual(context.chapter, "ANNEX VIII")

    def test_unusable_property_key_is_dropped(self) -> None:
        self.assertEqual(
            _clean_properties({"condition": "wet", "$$$": "ignored"}),
            {"condition": "wet"},
        )


class ConfigurationTests(unittest.TestCase):
    def test_document_kind_is_inferred_from_known_names(self) -> None:
        self.assertEqual(
            resolve_document_kind(
                DocumentKind.AUTO,
                Path("AIS-057.pdf"),
                "AIS-057",
            ),
            DocumentKind.STANDARD,
        )
        self.assertEqual(
            resolve_document_kind(
                DocumentKind.AUTO,
                Path("cmvr-1989.pdf"),
                "Central Motor Vehicle Rules",
            ),
            DocumentKind.CMVR,
        )

    def test_page_range_parser(self) -> None:
        self.assertEqual(parse_page_range("95-98"), (95, 98))
        self.assertEqual(parse_page_range("20"), (20, 20))

    def test_chunk_page_filter_uses_provenance(self) -> None:
        chunk = RegulatoryChunk(
            source_index=1,
            segment_index=0,
            text="Rule text",
            chapter="CHAPTER V",
            rule="Rule 115",
            hierarchy=("CHAPTER V", "Rule 115"),
            docling_headings=(),
            page_numbers=(94, 95),
            content_type="text",
        )
        self.assertTrue(chunk_overlaps_page_range(chunk, (95, 98)))
        self.assertFalse(chunk_overlaps_page_range(chunk, (96, 98)))


class VoyageAIEmbedderTests(unittest.TestCase):
    def test_uses_voyage_4_large_document_embeddings(self) -> None:
        client = MagicMock()
        client.tokenize.return_value = [[1, 2], [3, 4]]
        client.embed.return_value = SimpleNamespace(
            embeddings=[[0.1] * 1_024, [0.2] * 1_024],
            total_tokens=4,
        )

        with patch(
            "ingest_regulatory_graph.voyageai.Client",
            return_value=client,
        ) as constructor:
            embedder = VoyageAIEmbedder(
                api_key="test-key",
                model="voyage-4-large",
                timeout=30,
                attempts=5,
                batch_size=16,
                output_dimension=1_024,
            )
            vectors = embedder.embed_documents(["Rule 115", "AIS-057"])

        constructor.assert_called_once_with(
            api_key="test-key",
            timeout=30,
            max_retries=4,
        )
        client.embed.assert_called_once_with(
            texts=["Rule 115", "AIS-057"],
            model="voyage-4-large",
            input_type="document",
            truncation=False,
            output_dtype="float",
            output_dimension=1_024,
        )
        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(len(vector) == 1_024 for vector in vectors))

    def test_batches_by_voyage_aggregate_token_limit(self) -> None:
        client = MagicMock()
        client.tokenize.return_value = [[1, 2, 3], [4, 5, 6], [7]]
        client.embed.side_effect = [
            SimpleNamespace(embeddings=[[0.1]], total_tokens=3),
            SimpleNamespace(embeddings=[[0.2], [0.3]], total_tokens=4),
        ]

        with patch(
            "ingest_regulatory_graph.voyageai.Client",
            return_value=client,
        ):
            embedder = VoyageAIEmbedder(
                api_key="test-key",
                model="voyage-4-large",
                timeout=30,
                attempts=1,
                batch_size=10,
                output_dimension=1_024,
            )
            embedder.MAX_TOKENS_PER_REQUEST = 5
            vectors = embedder.embed_documents(["one", "two", "three"])

        self.assertEqual(
            [call.kwargs["texts"] for call in client.embed.call_args_list],
            [["one"], ["two", "three"]],
        )
        self.assertEqual(vectors, [[0.1], [0.2], [0.3]])

    def test_rejects_input_over_model_context(self) -> None:
        client = MagicMock()
        client.tokenize.return_value = [[1, 2, 3]]

        with patch(
            "ingest_regulatory_graph.voyageai.Client",
            return_value=client,
        ):
            embedder = VoyageAIEmbedder(
                api_key="test-key",
                model="voyage-4-large",
                timeout=30,
                attempts=1,
                batch_size=10,
                output_dimension=1_024,
            )
            embedder.MAX_TOKENS_PER_INPUT = 2
            with self.assertRaisesRegex(ValueError, "exceeding"):
                embedder.embed_documents(["overlong text"])

        client.embed.assert_not_called()


class MongoGraphWriterTests(unittest.TestCase):
    def test_upserts_nodes_edge_and_accumulates_chunk_lineage(self) -> None:
        extraction = ChunkExtractionResult(
            entities=[
                EntityModel(
                    entity_id="RULE_115",
                    name="Rule 115",
                    label=EntityLabel.REGULATION,
                ),
                EntityModel(
                    entity_id="CATEGORY_M3",
                    name="Category M3",
                    label=EntityLabel.VEHICLE_CLASS,
                ),
            ],
            relationships=[
                RelationshipModel(
                    source_entity_id="RULE_115",
                    target_entity_id="CATEGORY_M3",
                    relation_type=RelationType.APPLIES_TO,
                    properties=[
                        PropertyModel(key="condition", value="diesel vehicles")
                    ],
                )
            ],
        )
        chunk = RegulatoryChunk(
            source_index=1,
            segment_index=0,
            text="Rule 115 applies to category M3.",
            chapter="CHAPTER V",
            rule="Rule 115",
            hierarchy=("CHAPTER V", "Rule 115"),
            docling_headings=(),
            page_numbers=(97,),
            content_type="text",
        )

        original_add_update = BulkOperationBuilder.add_update

        def compatible_add_update(
            builder: BulkOperationBuilder,
            *args: object,
            sort: object = None,
            **kwargs: object,
        ) -> object:
            del sort
            return original_add_update(builder, *args, **kwargs)

        with (
            patch("ingest_regulatory_graph.MongoClient", mongomock.MongoClient),
            patch.object(
                BulkOperationBuilder,
                "add_update",
                compatible_add_update,
            ),
        ):
            writer = MongoGraphWriter(
                uri="mongodb://localhost",
                database_name="test_graph",
                timeout_ms=1_000,
                use_transactions=False,
                create_indexes=True,
            )
            first_id = writer.ingest_chunk(
                chunk=chunk,
                document_name="CMVR 1989",
                extraction=extraction,
                embedding=[0.1, 0.2],
            )
            second_id = writer.ingest_chunk(
                chunk=chunk,
                document_name="CMVR 1989",
                extraction=extraction,
                embedding=[0.3, 0.4],
            )

            edge = writer.relationships.find_one({})
            self.assertIsNotNone(edge)
            self.assertEqual(writer.chunks.count_documents({}), 2)
            self.assertEqual(writer.entities.count_documents({}), 2)
            self.assertEqual(writer.relationships.count_documents({}), 1)
            self.assertEqual(
                set(edge["source_chunk_ids"]),
                {first_id, second_id},
            )
            self.assertEqual(
                edge["properties"]["condition"],
                "diesel vehicles",
            )
            writer.close()


if __name__ == "__main__":
    unittest.main()