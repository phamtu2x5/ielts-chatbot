import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from backend.tools.chat_evaluation import capture_case, select_cases, verify_corpus


MANIFEST_PATH = BACKEND_DIR / "evaluation" / "chat_corpus_v2.json"
CORPUS_DIR = REPO_DIR / "docs"
REQUIRED_CATEGORIES = {
    "document_overview",
    "show_questions",
    "translate_questions",
    "explain_questions",
    "solve_questions",
    "semantic_qa",
    "show_table",
    "show_flowchart",
    "negative_document_qa",
    "writing_generation",
}


class ChatEvaluationManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_covers_all_corpus_documents(self) -> None:
        documents = self.manifest["documents"]
        self.assertEqual(len(documents), 7)
        verified = verify_corpus(self.manifest, CORPUS_DIR)
        self.assertEqual(len(verified), len(documents))

    def test_manifest_replaces_legacy_suite_with_broad_coverage(self) -> None:
        cases = self.manifest["cases"]
        self.assertGreaterEqual(len(cases), 50)
        ids = [case["id"] for case in cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotEqual(len(cases), 19)
        counts = Counter(case["target_files"][0] for case in cases)
        self.assertTrue(all(count >= 6 for count in counts.values()))
        self.assertEqual(len(counts), 7)

    def test_manifest_references_declared_documents(self) -> None:
        filenames = {document["filename"] for document in self.manifest["documents"]}
        categories = {case["category"] for case in self.manifest["cases"]}
        self.assertTrue(REQUIRED_CATEGORIES.issubset(categories))
        for case in self.manifest["cases"]:
            self.assertTrue(case["query"].strip())
            self.assertTrue(set(case["target_files"]).issubset(filenames))
            self.assertNotIn("expected_intent", case)
            self.assertNotIn("answer_terms", case)

    def test_corpus_hash_mismatch_is_rejected_before_requests(self) -> None:
        manifest = {
            "documents": [
                {"filename": "sample.txt", "sha256": "0" * 64},
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "sample.txt").write_text("different", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_corpus(manifest, Path(temp_dir))

    def test_capture_preserves_answer_sources_and_debug_without_scoring(self) -> None:
        case = {
            "id": "sample",
            "category": "semantic_qa",
            "query": "Question",
            "target_files": ["sample.pdf"],
        }
        result = {
            "http_status": 200,
            "duration_seconds": 0.1,
            "response": {
                "response": "Nội dung nói về Mars.",
                "route_used": "rag",
                "sources": [{"source_file": "sample.pdf"}],
                "debug": {"query_intent": "semantic_qa"},
            },
        }
        capture = capture_case(case, result)
        self.assertEqual(capture["answer"], "Nội dung nói về Mars.")
        self.assertEqual(capture["sources"], [{"source_file": "sample.pdf"}])
        self.assertEqual(capture["debug"], {"query_intent": "semantic_qa"})
        self.assertNotIn("status", capture)
        self.assertNotIn("failures", capture)

    def test_case_selection_rejects_unknown_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown case IDs"):
            select_cases(self.manifest, ["not-a-real-case"])


if __name__ == "__main__":
    unittest.main()
