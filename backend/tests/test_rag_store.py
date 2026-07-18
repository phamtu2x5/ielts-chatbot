import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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

from app import rag
from app.document_scope import resolve_document_scope
from app.intent import (
    detect_query_intent,
    filter_sources_for_intent,
    has_explicit_no_solution_constraint,
)
from app.llm import (
    clean_response,
    likely_contains_solution,
    looks_like_prompt_echo,
    rag_prompt,
    select_best_writing_output,
    writing_output_contract,
    writing_output_issues,
    writing_retry_prompt,
)
from app.structured_store import StructuredDocumentStore
from app.table_operations import (
    comparison_row,
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

    def test_document_intent_matrix(self) -> None:
        probe = {"is_overview": False, "has_document_intent": True}
        cases = [
            ("trả lời question 1 đến question 4", "solve_questions"),
            ("trả lời câu hỏi từ 1 đến 4 trong tài liệu trên", "solve_questions"),
            ("Hiển thị lại toàn bộ bảng của Questions 5-10, giữ đúng ô trống. Không giải bài.", "show_table"),
            ("Hiển thị cấu trúc flowchart của Questions 18-23, chưa điền đáp án.", "show_flowchart"),
            ("Tỷ lệ sở hữu smartphone của nước B năm 2024 là bao nhiêu?", "table_cell"),
            ("Trả lời Question 40 và giải thích vì sao đáp án phù hợp.", "solve_questions"),
            ("Hiển thị Questions 20-22, không chọn ba đáp án.", "show_questions"),
            ("Liệt kê Questions 28-29 cùng phần hướng dẫn và các ô trống, chưa điền đáp án.", "show_questions"),
            ("Giải thích Questions 27-32, không ghép đáp án A-H.", "explain_questions"),
            ("Từ bảng, quốc gia nào tăng nhiều nhất? Trình bày phép tính.", "table_calculation"),
            ("So sánh hai chỉ số của Country A từ bảng.", "table_comparison"),
            (
                "Đề Writing trong ảnh yêu cầu gì? Chỉ giải thích yêu cầu, chưa viết bài.",
                "show_writing_prompt",
            ),
            (
                "Viết riêng một đoạn overview cho bảng Writing, không viết introduction hoặc body.",
                "writing_generation",
            ),
            ("Viết bài IELTS Writing Task 1 dài 170-190 từ dựa trên ảnh.", "writing_generation"),
        ]
        for message, expected in cases:
            with self.subTest(message=message):
                self.assertEqual(detect_query_intent(message, probe), expected)

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
        self.assertTrue(ambiguous.ambiguous)

    def test_explicit_document_scope_is_always_grounded(self) -> None:
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

        self.assertTrue(scope.document_grounded)
        self.assertEqual(scope.resolved_document_ids, ["doc-reading"])

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


if __name__ == "__main__":
    unittest.main()
