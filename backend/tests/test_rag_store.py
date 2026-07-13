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
from app.intent import detect_query_intent, filter_sources_for_intent
from app.llm import looks_like_prompt_echo
from app.structured_store import StructuredDocumentStore


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

        hits = store.structured_lookup("Smartphone Ownership của nước B năm 2024 là bao nhiêu?", "show_table", top_k=2)

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

    def test_answer_question_range_is_solve_intent(self) -> None:
        intent = detect_query_intent(
            "trả lời question 1 đến question 4",
            {"is_overview": False, "has_document_intent": True},
        )

        self.assertEqual(intent, "solve_questions")

    def test_vietnamese_from_to_question_range_is_solve_intent(self) -> None:
        intent = detect_query_intent(
            "trả lời câu hỏi từ 1 đến 4 trong tài liệu trên",
            {"is_overview": False, "has_document_intent": True},
        )

        self.assertEqual(intent, "solve_questions")

    def test_no_solution_constraint_wins_over_question_markers(self) -> None:
        intent = detect_query_intent(
            "Hiển thị lại toàn bộ bảng của Questions 5-10, giữ đúng ô trống. Không giải bài.",
            {"is_overview": False, "has_document_intent": True},
        )

        self.assertEqual(intent, "show_table")

    def test_flowchart_no_fill_is_show_flowchart_intent(self) -> None:
        intent = detect_query_intent(
            "Hiển thị cấu trúc flowchart của Questions 18-23, chưa điền đáp án.",
            {"is_overview": False, "has_document_intent": True},
        )

        self.assertEqual(intent, "show_flowchart")

    def test_exact_table_cell_query_is_show_table_intent(self) -> None:
        intent = detect_query_intent(
            "Tỷ lệ sở hữu smartphone của nước B năm 2024 là bao nhiêu?",
            {"is_overview": False, "has_document_intent": True},
        )

        self.assertEqual(intent, "show_table")

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
