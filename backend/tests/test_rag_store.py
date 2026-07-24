import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    __import__("dotenv")
except ImportError:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: False
    sys.modules["dotenv"] = dotenv_stub

try:
    __import__("sentence_transformers")
except ImportError:
    sentence_transformers_stub = types.ModuleType("sentence_transformers")
    sentence_transformers_stub.SentenceTransformer = object
    sys.modules["sentence_transformers"] = sentence_transformers_stub

from app import llm, rag
from app.document_scope import resolve_document_scope
from app.intent import (
    filter_sources_for_intent,
    has_explicit_no_solution_constraint,
    semantic_intent_decision,
)
from app.llm import (
    clean_response,
    has_malformed_markdown_table,
    likely_contains_solution,
    looks_like_prompt_echo,
    rag_prompt,
    response_output_contract,
    response_output_issues,
    response_retry_prompt,
    select_best_writing_output,
    writing_output_contract,
    writing_output_issues,
    writing_retry_prompt,
)
from app.schemas import ChatMessage
from app.structured_store import StructuredDocumentStore
from app.table_operations import (
    comparison_row,
    comparison_row_facts,
    table_cell_value,
    table_change_calculations,
    table_summary_facts,
)


class FakeVectorStore(rag.LocalVectorStore):
    fail_embedding = False

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.fail_embedding:
            raise RuntimeError("embedding failed")
        return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)


class LocalVectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        data_dir = Path(self.temp_dir.name)
        self.path_patch = patch.multiple(
            rag,
            DATA_DIR=data_dir,
            DOCS_PATH=data_dir / "documents.json",
            INDEX_PATH=data_dir / "embeddings.npy",
        )
        self.path_patch.start()

    def tearDown(self) -> None:
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_upsert_replaces_source_without_desynchronizing_index(self) -> None:
        store = FakeVectorStore()
        store.upsert([self._chunk("a-1", "a.pdf", "first")], "a.pdf")
        store.upsert([self._chunk("a-2", "a.pdf", "replacement")], "a.pdf")

        self.assertEqual(store.stats()["chunks"], 1)
        self.assertEqual(store._docs[0]["chunk_id"], "a-2")
        self.assertEqual(store._embeddings.shape, (1, 2))

    def test_same_filename_documents_are_isolated_by_document_id(self) -> None:
        store = FakeVectorStore()
        first = self._chunk("a-1", "reading.pdf", "first")
        first["document_id"] = "doc-a"
        second = self._chunk("b-1", "reading.pdf", "second")
        second["document_id"] = "doc-b"

        store.upsert([first], "reading.pdf")
        store.upsert([second], "reading.pdf")

        self.assertEqual(store.stats()["documents"], 2)
        self.assertEqual({doc["document_id"] for doc in store._docs}, {"doc-a", "doc-b"})
        self.assertEqual(len(store.document_catalog()), 2)

    def test_reupload_replaces_only_matching_document_id(self) -> None:
        store = FakeVectorStore()
        first = self._chunk("a-1", "reading.pdf", "first")
        first["document_id"] = "doc-a"
        other = self._chunk("b-1", "reading.pdf", "other")
        other["document_id"] = "doc-b"
        replacement = self._chunk("a-2", "reading.pdf", "replacement")
        replacement["document_id"] = "doc-a"

        store.upsert([first], "reading.pdf")
        store.upsert([other], "reading.pdf")
        store.upsert([replacement], "reading.pdf")

        self.assertEqual({doc["chunk_id"] for doc in store._docs}, {"a-2", "b-1"})

    def test_explicit_empty_document_scope_returns_no_results(self) -> None:
        store = FakeVectorStore()
        chunk = self._chunk("a-1", "reading.pdf", "first")
        chunk["document_id"] = "doc-a"
        store.upsert([chunk], "reading.pdf")
        store.fail_embedding = True

        self.assertEqual(store.search("first", document_ids=[]), [])
        self.assertEqual(store.hybrid_search("first", document_ids=[]), [])
        self.assertEqual(store.structured_lookup("Questions 1-4", "show_questions", document_ids=[]), [])
        self.assertEqual(store.document_catalog(document_ids=[]), [])

    def test_overview_keeps_same_passage_number_from_different_documents(self) -> None:
        docs = []
        for document_id in ("doc-a", "doc-b"):
            docs.extend(
                [
                    {
                        "chunk_id": f"{document_id}-outline",
                        "document_id": document_id,
                        "source_file": f"{document_id}.pdf",
                        "chunk_index": len(docs),
                        "metadata": {"unit_type": "document_outline"},
                    },
                    {
                        "chunk_id": f"{document_id}-passage-1",
                        "document_id": document_id,
                        "source_file": f"{document_id}.pdf",
                        "chunk_index": len(docs) + 1,
                        "metadata": {"unit_type": "passage", "passage_number": 1},
                    },
                ]
            )

        hits = StructuredDocumentStore(lambda: docs).overview(top_k=4)

        self.assertEqual(
            {hit["document_id"] for hit in hits if hit["metadata"]["unit_type"] == "passage"},
            {"doc-a", "doc-b"},
        )

    def test_structured_lookup_is_owned_by_structured_store(self) -> None:
        store = FakeVectorStore()

        self.assertIsInstance(store.structured_store, StructuredDocumentStore)

    def test_embedding_failure_preserves_previous_source(self) -> None:
        store = FakeVectorStore()
        store.upsert([self._chunk("a-1", "a.pdf", "first")], "a.pdf")
        original_docs = json.loads(rag.DOCS_PATH.read_text(encoding="utf-8"))
        store.fail_embedding = True

        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            store.upsert([self._chunk("a-2", "a.pdf", "replacement")], "a.pdf")

        self.assertEqual(store._docs, original_docs)
        self.assertEqual(json.loads(rag.DOCS_PATH.read_text(encoding="utf-8")), original_docs)

    def test_question_metadata_is_prioritized_over_dense_similarity(self) -> None:
        store = FakeVectorStore()
        question_chunk = self._chunk("questions-1-4", "reading.pdf", "Questions 1-4")
        question_chunk["metadata"] = {
            "unit_type": "question_group",
            "question_range": [1, 4],
        }
        unrelated = self._chunk("long", "reading.pdf", "unrelated " * 100)
        store.upsert([unrelated, question_chunk], "reading.pdf")

        probe = store.probe("Nội dung Questions 1-4", top_k=2)

        self.assertEqual(probe["results"][0]["chunk_id"], "questions-1-4")
        self.assertGreater(probe["top_question_score"], 0)
        self.assertTrue(probe["has_strong_hits"])

    def test_structured_question_lookup_uses_metadata_without_dense_search(self) -> None:
        store = FakeVectorStore()
        group = self._chunk("questions-1-4", "reading.pdf", "Questions 1-4")
        group["metadata"] = {"unit_type": "question_group", "question_range": [1, 4]}
        question = self._chunk("question-2", "reading.pdf", "2. Yeast is white-coloured.")
        question["metadata"] = {"unit_type": "question", "question_range": [2, 2], "parent_id": "questions-1-4"}
        unrelated = self._chunk("passage", "reading.pdf", "A long unrelated passage")
        unrelated["metadata"] = {"unit_type": "passage", "passage_number": 3}
        store.upsert([unrelated, question, group], "reading.pdf")

        hits = store.structured_lookup("Liệt kê Questions 1-4", "show_questions", top_k=3)

        self.assertEqual(hits[0]["chunk_id"], "questions-1-4")
        self.assertEqual(hits[0]["retrieval_method"], "structured_question")
        self.assertGreater(hits[0]["structured_score"], hits[1]["structured_score"])

    def test_structured_question_lookup_supports_vietnamese_from_to_range(self) -> None:
        store = FakeVectorStore()
        group = self._chunk("questions-1-4", "reading.pdf", "Questions 1-4")
        group["metadata"] = {"unit_type": "question_group", "question_range": [1, 4]}
        store.upsert([group], "reading.pdf")

        hits = store.structured_lookup("trả lời câu hỏi từ 1 đến 4 trong tài liệu", "solve_questions", top_k=1)

        self.assertEqual(hits[0]["chunk_id"], "questions-1-4")

    def test_structured_writing_table_lookup_prefers_writing_table(self) -> None:
        store = FakeVectorStore()
        reading_table = self._chunk("questions-5-10", "reading.pdf", "Questions 5-10 Complete the table")
        reading_table["metadata"] = {"unit_type": "table", "question_type": "table_completion", "question_range": [5, 10]}
        writing_table = self._chunk("writing-table", "writing.png", "Country B Smartphone Ownership 2024 94")
        writing_table["metadata"] = {
            "unit_type": "writing_table",
            "document_type": "ielts_writing_task_1",
            "table": {
                "columns": ["Country", "Internet Access 2019", "Internet Access 2024", "Smartphone Ownership 2019", "Smartphone Ownership 2024"],
                "rows": [["A", 78, 96, 82, 99], ["B", 61, 89, 67, 94]],
            },
        }
        store.upsert([reading_table], "reading.pdf")
        store.upsert([writing_table], "writing.png")

        hits = store.structured_lookup(
            "Smartphone Ownership của nước B năm 2024 là bao nhiêu?",
            "table_cell",
            top_k=2,
        )

        self.assertEqual(hits[0]["chunk_id"], "writing-table")
        self.assertEqual(hits[0]["retrieval_method"], "structured_table_cell")

    def test_vietnamese_overview_query_uses_outline_and_passage_context(self) -> None:
        store = FakeVectorStore()
        outline = self._chunk("outline", "reading.pdf", "IELTS Reading document outline")
        outline["metadata"] = {"unit_type": "document_outline"}
        passage_one = self._chunk("passage-1", "reading.pdf", "Passage 1: Make That Wine")
        passage_one["metadata"] = {"unit_type": "passage", "passage_number": 1}
        passage_two = self._chunk("passage-2", "reading.pdf", "Passage 2: That Vision Thing")
        passage_two["metadata"] = {"unit_type": "passage", "passage_number": 2}
        store.upsert([outline, passage_one, passage_two], "reading.pdf")

        probe = store.probe("Nội dung của tài liệu trên là gì", top_k=3)

        self.assertTrue(probe["is_overview"])
        self.assertEqual([item["chunk_id"] for item in probe["results"]], ["outline", "passage-1", "passage-2"])
        self.assertTrue(all(item["probe_overview_score"] == 1.0 for item in probe["results"]))

    def test_semantic_intent_keeps_router_decision_except_for_explicit_constraints(self) -> None:
        solve = semantic_intent_decision(
            "Trả lời Questions 1-4 và giải thích.",
            "solve_questions",
            0.94,
            "The user asks for answers.",
        )
        no_solve = semantic_intent_decision(
            "Hiển thị Questions 1-4, không giải bài.",
            "solve_questions",
            0.70,
            "The request concerns questions.",
        )

        self.assertEqual(solve.intent, "solve_questions")
        self.assertTrue(solve.allow_solution)
        self.assertEqual(no_solve.intent, "show_questions")
        self.assertFalse(no_solve.allow_solution)

    def test_keyword_only_probe_does_not_load_embedding_model(self) -> None:
        store = FakeVectorStore()
        chunk = self._chunk("a-1", "reading.pdf", "That Vision Thing")
        chunk["document_id"] = "doc-a"
        store.upsert([chunk], "reading.pdf")
        store.fail_embedding = True

        probe = store.probe("That Vision Thing", include_dense=False)

        self.assertTrue(probe["has_hits"])
        self.assertEqual(probe["results"][0]["probe_dense_score"], 0.0)

    def test_structured_table_operations_return_cell_calculation_and_comparison(self) -> None:
        table = {
            "columns": [
                "Country",
                "Internet Access 2019 (%)",
                "Internet Access 2024 (%)",
                "Smartphone Ownership 2019 (%)",
                "Smartphone Ownership 2024 (%)",
            ],
            "rows": [
                ["A", 78, 96, 82, 99],
                ["B", 61, 89, 67, 94],
                ["C", 42, 75, 48, 83],
            ],
        }

        cell = table_cell_value("Smartphone Ownership của Country B năm 2024 là bao nhiêu?", table)
        calculation = table_change_calculations(
            "Quốc gia nào tăng Internet Access nhiều nhất từ 2019 đến 2024? Trình bày phép tính.",
            table,
        )
        row = comparison_row(
            "So sánh Internet Access và Smartphone Ownership của Country A trong cả hai năm.",
            table,
        )

        self.assertEqual(cell[1], 94)
        self.assertEqual(
            [item["change"] for item in calculation["calculations"]],
            [18.0, 28.0, 33.0],
        )
        self.assertEqual(calculation["winner"]["label"], "C")
        self.assertEqual(row, ["A", 78, 96, 82, 99])

        facts = table_summary_facts(table)
        self.assertIn("Largest increase: C (+33)", facts[0])
        self.assertIn("A 78 -> 96 (+18)", facts[0])
        self.assertIn("Ranking in 2024: A 96 > B 89 > C 75", facts[1])
        self.assertIn("Highest final value in Internet Access 2024 (%): A (96)", facts[1])
        self.assertIn("Largest increase: C (+35)", facts[2])
        self.assertIn("Ranking in 2024: A 99 > B 94 > C 83", facts[3])
        self.assertIn("Highest final value in Smartphone Ownership 2024 (%): A (99)", facts[3])

    def test_document_scope_resolves_filename_and_reports_ambiguity(self) -> None:
        catalog = [
            {
                "source_file": "IZONE _ IELTS READING TEST 2.pdf",
                "document_ids": ["doc-2"],
                "mime_types": ["application/pdf"],
            },
            {
                "source_file": "IZONE _ IELTS READING TEST 4.pdf",
                "document_ids": ["doc-4"],
                "mime_types": ["application/pdf"],
            },
        ]

        resolved = resolve_document_scope("Liệt kê Questions 1-4 trong Reading Test 2", catalog)
        ambiguous = resolve_document_scope("Liệt kê Questions 1-4", catalog)

        self.assertEqual(resolved.resolved_document_ids, ["doc-2"])
        self.assertFalse(resolved.ambiguous)
        self.assertFalse(ambiguous.ambiguous)
        self.assertEqual(ambiguous.method, "unresolved")

        self.assertEqual(ambiguous.resolved_document_ids, [])
        self.assertEqual(resolved.resolved_document_ids, ["doc-2"])

    def test_explicit_document_scope_only_limits_allowed_documents(self) -> None:
        catalog = [
            {
                "source_file": "reading.pdf",
                "document_ids": ["doc-reading"],
                "mime_types": ["application/pdf"],
            }
        ]

        scope = resolve_document_scope(
            "How did the fence affect kangaroos?",
            catalog,
            ["doc-reading"],
        )

        self.assertFalse(scope.document_grounded)
        self.assertEqual(scope.resolved_document_ids, ["doc-reading"])

        general_writing = resolve_document_scope(
            "Give me three IELTS Writing tips.",
            catalog,
            ["doc-reading"],
        )
        self.assertFalse(general_writing.document_grounded)

        explicit = resolve_document_scope(
            "How did the fence affect kangaroos?",
            catalog,
            ["doc-reading"],
            request_mode="explicit",
        )
        self.assertFalse(explicit.document_grounded)
        self.assertEqual(explicit.method, "explicit_single")

    def test_available_document_ids_do_not_force_general_chat_into_rag(self) -> None:
        catalog = [
            {
                "source_file": "reading.pdf",
                "document_ids": ["doc-reading"],
                "mime_types": ["application/pdf"],
                "section_titles": ["Snow-makers"],
            }
        ]

        greeting = resolve_document_scope(
            "Hello",
            catalog,
            ["doc-reading"],
            request_mode="available",
        )
        section_query = resolve_document_scope(
            "What is Snow-makers about?",
            catalog,
            ["doc-reading"],
            request_mode="available",
        )

        self.assertFalse(greeting.document_grounded)
        self.assertEqual(greeting.method, "requested_single")
        self.assertTrue(section_query.document_grounded)
        self.assertEqual(section_query.method, "catalog_reference")

    def test_generic_ielts_terms_do_not_select_uploaded_documents(self) -> None:
        catalog = [
            {
                "source_file": "IELTS READING TEST 2.pdf",
                "document_ids": ["doc-reading"],
                "mime_types": ["application/pdf"],
            },
            {
                "source_file": "IELTS Task 1 Essay.pdf",
                "document_ids": ["doc-writing"],
                "mime_types": ["application/pdf"],
            },
        ]

        for message in [
            "Give me three IELTS Speaking Part 2 tips.",
            "How can I improve my IELTS Reading speed?",
            "Explain TRUE/FALSE/NOT GIVEN in IELTS Reading.",
            "Give me an IELTS Writing Task 2 discussion essay structure.",
        ]:
            with self.subTest(message=message):
                scope = resolve_document_scope(message, catalog)
                self.assertFalse(scope.document_grounded)
                self.assertFalse(scope.ambiguous)
                self.assertEqual(scope.resolved_document_ids, [])

        overview = resolve_document_scope("Nội dung tài liệu là gì?", catalog)
        self.assertFalse(overview.document_grounded)
        self.assertFalse(overview.ambiguous)
        self.assertEqual(overview.method, "unresolved")

    def test_structured_section_title_selects_one_document(self) -> None:
        catalog = [
            {
                "source_file": "reading-a.pdf",
                "document_ids": ["doc-a"],
                "section_titles": ["Snow-makers"],
            },
            {
                "source_file": "reading-b.pdf",
                "document_ids": ["doc-b"],
                "section_titles": ["IELTS Essay Task 1: Age Groups and Cinema Attendance"],
            },
        ]

        scope = resolve_document_scope("Summarize Age Groups and Cinema Attendance.", catalog)

        self.assertTrue(scope.document_grounded)
        self.assertFalse(scope.ambiguous)
        self.assertEqual(scope.resolved_document_ids, ["doc-b"])

    def test_document_catalog_exposes_structured_section_titles(self) -> None:
        store = FakeVectorStore()
        passage = self._chunk("passage", "reading.pdf", "Passage body")
        passage["document_id"] = "doc-reading"
        passage["metadata"] = {
            "unit_type": "passage",
            "passage_title": "A General Passage Title",
        }
        store.upsert([passage], "reading.pdf")

        catalog = store.document_catalog()

        self.assertEqual(catalog[0]["section_titles"], ["A General Passage Title"])

    def test_structured_lookup_filters_duplicate_question_ranges_by_document(self) -> None:
        store = FakeVectorStore()
        first = self._chunk("doc-a-questions", "a.pdf", "Questions 1-4")
        first["document_id"] = "doc-a"
        first["metadata"] = {"unit_type": "question_group", "question_range": [1, 4]}
        second = self._chunk("doc-b-questions", "b.pdf", "Questions 1-4")
        second["document_id"] = "doc-b"
        second["metadata"] = {"unit_type": "question_group", "question_range": [1, 4]}
        store.upsert([first], "a.pdf")
        store.upsert([second], "b.pdf")

        hits = store.structured_lookup(
            "Liệt kê Questions 1-4",
            "show_questions",
            top_k=5,
            document_ids=["doc-b"],
        )

        self.assertEqual([item["chunk_id"] for item in hits], ["doc-b-questions"])

    def test_dense_and_probe_search_filter_before_ranking(self) -> None:
        store = FakeVectorStore()
        first = self._chunk("doc-a-long", "a.pdf", "unrelated " * 100)
        first["document_id"] = "doc-a"
        second = self._chunk("doc-b-short", "b.pdf", "target text")
        second["document_id"] = "doc-b"
        store.upsert([first], "a.pdf")
        store.upsert([second], "b.pdf")

        dense = store.search("target", top_k=5, document_ids=["doc-b"])
        probe = store.probe("target", top_k=5, document_ids=["doc-b"])

        self.assertEqual([item["document_id"] for item in dense], ["doc-b"])
        self.assertTrue(
            all(item["document_id"] == "doc-b" for item in probe["results"])
        )

    def test_rrf_rewards_candidates_supported_by_multiple_retrievers(self) -> None:
        store = FakeVectorStore()

        hits = store._fuse_ranked_results(
            dense_results=[
                {"chunk_id": "dense-only", "score": 0.99},
                {"chunk_id": "consensus", "score": 0.7},
            ],
            keyword_results=[
                {"chunk_id": "consensus", "keyword_score": 3.0},
            ],
            question_results=[],
            top_k=2,
        )

        self.assertEqual(hits[0]["chunk_id"], "consensus")
        self.assertEqual(hits[0]["retrieval_methods"], ["dense", "keyword"])

    def test_hybrid_search_filters_unit_and_passage_before_ranking(self) -> None:
        store = FakeVectorStore()
        wrong_passage = self._chunk("wrong-passage", "reading.pdf", "target " * 100)
        wrong_passage["document_id"] = "doc-reading"
        wrong_passage["metadata"] = {"unit_type": "passage", "passage_number": 1}
        question = self._chunk("question", "reading.pdf", "target question")
        question["document_id"] = "doc-reading"
        question["metadata"] = {"unit_type": "question", "passage_number": 2}
        evidence = self._chunk("evidence", "reading.pdf", "target evidence")
        evidence["document_id"] = "doc-reading"
        evidence["metadata"] = {"unit_type": "passage", "passage_number": 2}
        store.upsert([wrong_passage, question, evidence], "reading.pdf")

        hits = store.hybrid_search(
            "target",
            top_k=5,
            document_ids=["doc-reading"],
            unit_types=["passage"],
            passage_numbers=[2],
        )

        self.assertEqual([item["chunk_id"] for item in hits], ["evidence"])

    def test_parent_expansion_preserves_document_id(self) -> None:
        store = FakeVectorStore()
        chunks = []
        for document_id, source_file in [("doc-a", "a.pdf"), ("doc-b", "b.pdf")]:
            group = self._chunk(f"{document_id}-group", source_file, "Questions 1-4")
            group["document_id"] = document_id
            group["metadata"] = {
                "unit_type": "question_group",
                "question_range": [1, 4],
            }
            passage = self._chunk(f"{document_id}-passage", source_file, "Passage 1")
            passage["document_id"] = document_id
            passage["metadata"] = {"unit_type": "passage", "passage_number": 1}
            chunks.extend([group, passage])
        store.upsert(chunks[:2], "a.pdf")
        store.upsert(chunks[2:], "b.pdf")
        source = dict(chunks[2])
        source["metadata"] = {**source["metadata"], "passage_number": 1}

        expanded = store.passage_context_for_sources(
            [source],
            max_chunks_per_passage=3,
            document_ids=["doc-b"],
        )

        self.assertEqual([item["document_id"] for item in expanded], ["doc-b"])

    def test_explicit_passage_filter_does_not_fallback_to_wrong_passage(self) -> None:
        sources = [
            {"chunk_id": "p1", "metadata": {"unit_type": "passage", "passage_number": 1}},
            {"chunk_id": "p3", "metadata": {"unit_type": "passage", "passage_number": 3}},
        ]

        filtered = filter_sources_for_intent(sources, "Passage 2 nói gì?", "semantic_qa")

        self.assertEqual(filtered, [])

    def test_writing_image_terms_are_document_intent(self) -> None:
        store = FakeVectorStore()
        table = self._chunk("writing-table", "writing.png", "Smartphone Ownership 2024 Country B 94")
        table["metadata"] = {"unit_type": "writing_table", "document_type": "ielts_writing_task_1"}
        store.upsert([table], "writing.png")

        probe = store.probe("Tỷ lệ sở hữu smartphone của nước B năm 2024 là bao nhiêu?", top_k=1)

        self.assertTrue(probe["has_document_intent"])
        self.assertEqual(probe["results"][0]["chunk_id"], "writing-table")

    def test_prompt_echo_is_detected(self) -> None:
        prompt = "You are an IELTS preparation assistant.\n\nStudy material context:\nQuestions 1-4..."
        echoed = prompt + "\n\nQuestion: trả lời question 1 đến question 4"

        self.assertTrue(looks_like_prompt_echo(echoed, prompt))
        self.assertFalse(looks_like_prompt_echo("Câu 1 là TRUE vì đoạn văn nêu...", prompt))

    def test_response_cleanup_removes_decorative_icons(self) -> None:
        cleaned = clean_response(
            "✅ Kết quả: C tăng 33%. 👉 A → B vẫn giữ mũi tên. [Source 2: reading.pdf, pages 1, 2]"
        )

        self.assertEqual(
            cleaned,
            "Kết quả: C tăng 33%. A → B vẫn giữ mũi tên. reading.pdf, pages 1, 2",
        )

    def test_writing_output_contract_and_validation(self) -> None:
        contract = writing_output_contract(
            "Viết bài IELTS Writing Task 1 dài 170-190 từ dựa trên bảng."
        )
        vietnamese_contract = writing_output_contract(
            "Viết overview bằng tiếng Việt."
        )

        self.assertEqual(contract.language, "English")
        self.assertEqual((contract.min_words, contract.max_words), (170, 190))
        self.assertEqual(contract.target_words, (178, 184))
        self.assertEqual(vietnamese_contract.language, "Vietnamese")
        self.assertTrue(vietnamese_contract.single_paragraph)
        self.assertTrue(vietnamese_contract.overview_only)
        issues = writing_output_issues(
            "Bảng này cho thấy tỷ lệ tăng đáng kể ở cả ba quốc gia trong giai đoạn nghiên cứu.",
            contract,
        )

        self.assertTrue(any("not written in English" in issue for issue in issues))
        self.assertTrue(any("below 170" in issue for issue in issues))
        self.assertIn(
            "The response is not written in Vietnamese.",
            writing_output_issues("The chart shows a consistent increase.", vietnamese_contract),
        )
        self.assertIn(
            "The response contains meta commentary instead of starting with the Writing content.",
            writing_output_issues("Here is the revised essay: The chart increased.", contract),
        )

        retry_prompt = writing_retry_prompt("original grounded context", contract)
        self.assertIn("original grounded context", retry_prompt)
        self.assertNotIn("previous draft", retry_prompt.lower())
        self.assertNotIn("below 170", retry_prompt.lower())

        first = "Here is the revised essay: " + " ".join(["word"] * 175)
        second = " ".join(["word"] * 169)
        self.assertEqual(select_best_writing_output(first, second, contract), second)

    def test_translation_contract_requires_vietnamese_and_all_question_numbers(self) -> None:
        contract = response_output_contract(
            "Dịch Questions 25-27 sang tiếng Việt, chưa trả lời.",
            "translate_questions",
            allow_solution=False,
        )

        self.assertEqual(contract.language, "Vietnamese")
        self.assertEqual(contract.required_question_numbers, (25, 26, 27))
        issues = response_output_issues(
            "25. Which body provides global tourist numbers?",
            contract,
        )
        self.assertTrue(any("not written in Vietnamese" in issue for issue in issues))
        self.assertTrue(any("26, 27" in issue for issue in issues))
        retry_prompt = response_retry_prompt("original context", contract)
        self.assertIn("Output language: Vietnamese", retry_prompt)
        self.assertNotIn("Which body", retry_prompt)

    def test_comparison_facts_describe_changes_instead_of_only_reprinting_row(self) -> None:
        table = {
            "columns": [
                "Country",
                "Internet Access 2019 (%)",
                "Internet Access 2024 (%)",
                "Smartphone Ownership 2019 (%)",
                "Smartphone Ownership 2024 (%)",
            ],
            "rows": [["A", 78, 96, 82, 99]],
        }

        facts = comparison_row_facts(table, table["rows"][0])

        self.assertIn("Internet Access: 78 (2019) → 96 (2024), tăng 18.", facts)
        self.assertIn("Smartphone Ownership: 82 (2019) → 99 (2024), tăng 17.", facts)

    def test_solve_prompt_requires_explicit_evidence_relationship(self) -> None:
        prompt = rag_prompt(
            "Trả lời Questions 1-4.",
            "[Source: reading.pdf, page 1] passage evidence",
            query_intent="solve_questions",
            allow_solution=True,
        )

        self.assertIn("one short evidence quote and its relationship", prompt)
        self.assertIn("supports -> TRUE; contradicts -> FALSE; absent -> NOT GIVEN", prompt)

    def test_no_solution_constraint_requires_an_explicit_marker(self) -> None:
        self.assertTrue(has_explicit_no_solution_constraint("Giải thích Questions 1-4, không chọn đáp án."))
        self.assertFalse(has_explicit_no_solution_constraint("Giải thích và trả lời Questions 1-4."))
        self.assertFalse(has_explicit_no_solution_constraint("Trả lời Question 1 nhưng không giải thích."))
        self.assertTrue(likely_contains_solution("Tóm lại:\n24: shade-grown\n25: full-sun"))
        self.assertTrue(likely_contains_solution("Câu hỏi 24 → shade-grown"))
        self.assertTrue(likely_contains_solution("Câu 36 phù hợp với A Levitin."))
        self.assertTrue(
            likely_contains_solution("Câu hỏi 24: nội dung này không thể phân loại từ passage.")
        )
        self.assertTrue(likely_contains_solution("Có thể loại trừ phương án B, chỉ còn A."))
        self.assertFalse(likely_contains_solution("Đối chiếu từng phát biểu với thông tin trong passage."))

    def test_writing_parent_context_keeps_one_task(self) -> None:
        docs = []
        for task_number in (1, 2):
            for unit_type in ("writing_task", "sample_answer"):
                docs.append(
                    {
                        "chunk_id": f"task-{task_number}-{unit_type}",
                        "document_id": "doc-writing",
                        "source_file": "writing.pdf",
                        "chunk_index": len(docs),
                        "metadata": {
                            "unit_type": unit_type,
                            "parent_id": f"writing-task-{task_number}",
                        },
                    }
                )
        store = StructuredDocumentStore(lambda: docs)

        hits = store.writing_context_for_sources([docs[2]], document_ids=["doc-writing"])

        self.assertEqual(len(hits), 2)
        self.assertEqual(
            {hit["metadata"]["parent_id"] for hit in hits},
            {"writing-task-2"},
        )

    def _chunk(self, chunk_id: str, source_file: str, text: str) -> dict:
        return {
            "chunk_id": chunk_id,
            "source_file": source_file,
            "text": text,
            "chunk_index": 0,
            "pages": [1],
            "metadata": {},
        }


class OllamaClientTests(unittest.IsolatedAsyncioTestCase):
    def test_route_classifier_prompt_keeps_only_latest_exchange(self) -> None:
        history = [
            ChatMessage(role="user", content="Old document question"),
            ChatMessage(role="assistant", content="Old document answer"),
            ChatMessage(role="user", content="Latest user turn"),
            ChatMessage(role="assistant", content="Latest assistant turn"),
        ]

        prompt = llm.route_classifier_prompt("Why?", history)

        self.assertNotIn("Old document question", prompt)
        self.assertNotIn("Old document answer", prompt)
        self.assertIn("User: Latest user turn", prompt)
        self.assertIn("Assistant: Latest assistant turn", prompt)
        self.assertIn("=== CURRENT REQUEST TO CLASSIFY ===\nWhy?", prompt)
        self.assertTrue(prompt.rstrip().endswith("Earlier context is supporting context, not the request itself."))

    def test_route_classifier_prompt_bounds_long_recent_messages(self) -> None:
        history = [
            ChatMessage(role="user", content="A" * 4_000),
            ChatMessage(role="assistant", content="B" * 4_000),
        ]

        prompt = llm.route_classifier_prompt("Hello", history)
        history_text = prompt.split("Previous conversation:\n", 1)[1].split(
            "\n\n=== CURRENT REQUEST TO CLASSIFY ===", 1
        )[0]

        self.assertLessEqual(len(history_text), 1_230)
        self.assertIn("\n...\n", history_text)

    def test_route_classifier_prompt_includes_metadata_and_highlights_request(self) -> None:
        document_context = (
            "- file=reading.pdf; attached_this_turn=true; sections=Urban transport"
        )

        prompt = llm.route_classifier_prompt(
            "Translate Questions 1-4.",
            document_context=document_context,
        )

        self.assertIn("Uploaded material signatures (metadata only", prompt)
        self.assertIn("file=reading.pdf", prompt)
        self.assertIn("=== CURRENT REQUEST TO CLASSIFY ===\nTranslate Questions 1-4.", prompt)
        self.assertIn("translating uploaded content", prompt)
        self.assertIn("Do not choose DIRECT by guessing", prompt)
        self.assertIn("attached_this_turn=true", prompt)
        self.assertIn("not sufficient by itself to choose RAG", prompt)

    def test_route_classifier_uses_document_dependency_not_topic_domain(self) -> None:
        prompt = llm.route_classifier_prompt("Explain a common technology concept.")
        compact_prompt = llm.route_classifier_prompt(
            "Explain a common technology concept.",
            compact=True,
        )

        self.assertIn("DIRECT: the answer is independent of uploaded-file content", prompt)
        self.assertIn("RAG: the answer needs to know or verify any specific content", prompt)
        self.assertNotIn("if the uploaded files were unavailable", prompt)
        self.assertIn("Do not choose DIRECT by guessing", compact_prompt)
        self.assertIn("not an automatic RAG decision", compact_prompt)
        self.assertIn("Transforming a preceding direct answer remains DIRECT", compact_prompt)

    async def test_route_classifier_returns_direct_without_generating_answer(self) -> None:
        model = AsyncMock(return_value='{"route":"direct"}')
        with patch.object(llm, "query_ollama", model):
            decision = await llm.classify_chat_route("Give me one Speaking Part 2 tip.")

        self.assertEqual(decision.route, "direct")
        self.assertEqual(model.await_args.kwargs["temperature"], 0.0)
        self.assertFalse(model.await_args.kwargs["clean_output"])
        self.assertEqual(model.await_args.kwargs["max_attempts"], 1)
        self.assertEqual(model.await_args.kwargs["num_predict"], 32)
        self.assertEqual(model.await_args.kwargs["response_format"], llm.ROUTE_RESPONSE_SCHEMA)
        self.assertEqual(model.await_args.kwargs["seed"], llm.settings.ollama_classifier_seed)

    async def test_route_classifier_accepts_rag_json(self) -> None:
        with patch.object(llm, "query_ollama", AsyncMock(return_value='{"route":"rag"}')):
            decision = await llm.classify_chat_route("What is Question 4 in the file?")

        self.assertEqual(decision.route, "rag")

    async def test_route_classifier_accepts_a_role_prefix_around_valid_json(self) -> None:
        with patch.object(
            llm,
            "query_ollama",
            AsyncMock(return_value='assistant\n```json\n{"route":"direct"}\n```'),
        ):
            decision = await llm.classify_chat_route("Hello")

        self.assertEqual(decision.route, "direct")

    async def test_route_classifier_retries_invalid_json(self) -> None:
        model = AsyncMock(side_effect=["not-json", '{"route":"rag"}'])
        with patch.object(llm, "query_ollama", model):
            decision = await llm.classify_chat_route("What does the uploaded file say?")

        self.assertEqual(decision.route, "rag")
        self.assertEqual(model.await_count, 2)

    async def test_route_classifier_retries_one_empty_response(self) -> None:
        model = AsyncMock(
            side_effect=[
                llm.OllamaRequestError("empty_response", "empty"),
                '{"route":"direct"}',
            ]
        )
        with patch.object(llm, "query_ollama", model):
            decision = await llm.classify_chat_route("Hello")

        self.assertEqual(decision.route, "direct")
        self.assertEqual(model.await_count, 2)
        first_prompt = model.await_args_list[0].args[0]
        retry_prompt = model.await_args_list[1].args[0]
        self.assertIn("semantic direct-or-document classifier", first_prompt)
        self.assertNotEqual(retry_prompt, first_prompt)

    async def test_route_classifier_returns_safe_undetermined_after_two_failures(self) -> None:
        model = AsyncMock(return_value='{"route":"unknown"}')
        with patch.object(llm, "query_ollama", model):
            decision = await llm.classify_chat_route("Ambiguous request")

        self.assertEqual(decision.route, "undetermined")
        self.assertEqual(decision.attempts, 2)

    async def test_intent_classifier_accepts_only_enum(self) -> None:
        with patch.object(
            llm,
            "query_ollama",
            AsyncMock(return_value='{"intent":"show_questions"}'),
        ):
            decision = await llm.classify_rag_intent("Show Questions 1-4")
        self.assertEqual(decision.intent, "show_questions")

    async def test_intent_classifier_rejects_enum_outside_candidates(self) -> None:
        with patch.object(
            llm,
            "query_ollama",
            AsyncMock(return_value='{"intent":"solve_questions"}'),
        ):
            decision = await llm.classify_rag_intent(
                "Explain the passage argument.",
                allowed_intents=("document_overview", "semantic_qa"),
            )
        self.assertEqual(decision.intent, "undetermined")
        self.assertEqual(decision.fallback_reason, "invalid_intent_output")

    def test_intent_classifier_prompt_lists_only_candidates(self) -> None:
        prompt = llm.intent_classifier_prompt(
            "Explain the passage argument.",
            allowed_intents=("document_overview", "semantic_qa"),
        )
        allowed_line = next(line for line in prompt.splitlines() if line.startswith("Allowed enums:"))
        self.assertIn("document_overview", allowed_line)
        self.assertIn("semantic_qa", allowed_line)
        self.assertNotIn("solve_questions", allowed_line)
        self.assertNotIn("Use solve_questions", prompt)

    async def test_intent_classifier_fails_closed_after_invalid_output(self) -> None:
        with patch.object(
            llm,
            "query_ollama",
            AsyncMock(return_value='{"intent":"unsupported_intent"}'),
        ):
            decision = await llm.classify_rag_intent("Show Questions 1-4")
        self.assertEqual(decision.intent, "undetermined")
        self.assertEqual(decision.fallback_reason, "invalid_intent_output")

    def test_intent_classifier_prompt_requests_final_enum_without_intermediate_labels(self) -> None:
        prompt = llm.intent_classifier_prompt("Summarize the uploaded document")
        self.assertIn('"intent":"<allowed enum>"', prompt)
        self.assertIn("document_overview", prompt)
        self.assertIn("semantic_qa", prompt)
        self.assertIn("inventory of its passages, sections, tasks, sample answers, or question groups", prompt)
        self.assertIn("specific numbered Reading questions", prompt)
        self.assertIn("topic, requirements, instructions, or discussion directions", prompt)
        self.assertIn("Answer Question 11 and cite evidence", prompt)
        self.assertIn("write an overview without an introduction or body", prompt)
        self.assertIn("explicitly targets a table", prompt)
        self.assertIn("sample answer compare two regions", prompt)
        self.assertIn("Rank the countries by the figures described in this sample answer", prompt)
        self.assertNotIn('"action"', prompt)
        self.assertNotIn('"target"', prompt)

    async def test_target_resolver_accepts_catalog_refs(self) -> None:
        catalog = "- D1: first.pdf\n- D2: second.pdf"
        with patch.object(
            llm,
            "query_ollama",
            AsyncMock(return_value='{"action":"selected","document_refs":["D2"]}'),
        ):
            decision = await llm.resolve_rag_target("Use second.pdf", catalog)
        self.assertEqual(decision.action, "selected")
        self.assertEqual(decision.document_refs, ("D2",))

    def test_target_resolver_treats_affinity_as_weak_context(self) -> None:
        prompt = llm.target_resolver_prompt(
            "Passage 2 nói gì?",
            "- D1: test-2.pdf\n- D2: test-4.pdf",
            [ChatMessage(role="user", content="Tóm tắt Test 2")],
            ("D1",),
        )

        self.assertIn("weak context, not a required scope", prompt)
        self.assertIn("Previous successful RAG document candidates: D1", prompt)
        self.assertIn("Tóm tắt Test 2", prompt)
        self.assertIn("Current user message:\nPassage 2 nói gì?", prompt)

    def test_direct_answer_prompt_requires_depth_for_tips_and_plans(self) -> None:
        prompt = llm.direct_answer_prompt("Lên kế hoạch học IELTS trong ba tháng")
        self.assertIn("practical Markdown table", prompt)
        self.assertIn("why it helps", prompt)
        self.assertIn("progress checks", prompt)
        self.assertIn("full requested timeline without gaps", prompt)
        self.assertIn("every row on exactly one physical line", prompt)
        self.assertIn("do not duplicate or skip periods", prompt)

    def test_direct_answer_prompt_accepts_general_knowledge_requests(self) -> None:
        prompt = llm.direct_answer_prompt("Explain a common technology concept.")

        self.assertIn("Answer this request from general knowledge", prompt)
        self.assertNotIn("Answer this general IELTS request", prompt)

    def test_malformed_markdown_table_detects_multiline_cells(self) -> None:
        malformed = """| Period | Activities | Time |
| --- | --- | --- |
| Weeks 1-4 | - Read daily
- Listen daily | 60 minutes |"""
        valid = """| Period | Activities | Time |
| --- | --- | --- |
| Weeks 1-4 | Read daily; listen daily | 60 minutes |"""

        self.assertTrue(has_malformed_markdown_table(malformed))
        self.assertFalse(has_malformed_markdown_table(valid))
        contract = response_output_contract("Create a plan", "direct", allow_solution=False)
        self.assertTrue(
            any("malformed Markdown table" in issue for issue in response_output_issues(malformed, contract))
        )

    async def test_non_stream_request_retries_one_server_error(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(503, text="model busy", request=request)
            return httpx.Response(200, json={"response": "Ready"}, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(llm.httpx, "AsyncClient", return_value=client):
            answer = await llm.query_ollama("Say ready", temperature=0.0)

        self.assertEqual(answer, "Ready")
        self.assertEqual(attempts, 2)

    def test_ollama_payload_disables_thinking_at_top_level(self) -> None:
        payload = llm._ollama_payload("Say ready", False, 0.0, 32)

        self.assertIs(payload["think"], False)
        self.assertNotIn("think", payload["options"])

    async def test_non_stream_request_does_not_expose_thinking_as_answer(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "", "thinking": "private reasoning"}, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(llm.httpx, "AsyncClient", return_value=client):
            with self.assertRaises(llm.OllamaRequestError) as raised:
                await llm.query_ollama("Say ready", temperature=0.0)

        self.assertEqual(raised.exception.kind, "empty_response")
        self.assertEqual(raised.exception.metadata["thinking_length"], len("private reasoning"))
        self.assertEqual(raised.exception.metadata["response_length"], 0)

    async def test_non_stream_error_keeps_status_and_response_body(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="model unavailable", request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(llm.httpx, "AsyncClient", return_value=client):
            with self.assertRaises(llm.OllamaRequestError) as raised:
                await llm.query_ollama("Say ready", temperature=0.0)

        detail = raised.exception.debug_detail()
        self.assertEqual(detail["status_code"], 503)
        self.assertEqual(detail["response_body"], "model unavailable")
        self.assertEqual(detail["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
