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
from app.intent import detect_query_intent
from app.llm import looks_like_prompt_echo


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
