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

from backend.tools.chat_evaluation import (
    capture_case,
    compact_upload_result,
    select_cases,
    verify_corpus,
)


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
                "sources": [
                    {
                        "chunk_id": "sample-c1",
                        "source_file": "sample.pdf",
                        "pages": [1],
                        "display_text": "Mars is potentially habitable.",
                        "probe_dense_score": 0.75,
                        "metadata": {"unit_type": "passage"},
                    }
                ],
                "debug": {
                    "query_intent": "semantic_qa",
                    "catalog": [{"source_file": "sample.pdf"}],
                },
            },
        }
        source_index = {}
        capture = capture_case(case, result, source_index, ["doc-1"])
        self.assertEqual(capture["answer"], "Nội dung nói về Mars.")
        self.assertEqual(capture["request_document_ids"], ["doc-1"])
        self.assertEqual(
            capture["sources"],
            [
                {
                    "source_ref": "sample-c1",
                    "source_file": "sample.pdf",
                    "pages": [1],
                    "dense_score": 0.75,
                    "unit_type": "passage",
                }
            ],
        )
        self.assertEqual(capture["debug"], {"query_intent": "semantic_qa"})
        self.assertEqual(
            source_index["sample-c1"]["text"],
            "Mars is potentially habitable.",
        )
        self.assertNotIn("raw_response", capture)
        self.assertNotIn("status", capture)
        self.assertNotIn("failures", capture)

    def test_capture_preserves_http_error_detail(self) -> None:
        case = {
            "id": "error",
            "category": "writing_generation",
            "query": "Viết overview.",
            "target_files": ["writing.png"],
        }
        capture = capture_case(
            case,
            {
                "http_status": 502,
                "duration_seconds": 0.2,
                "response": {"detail": "Ollama unavailable"},
            },
            request_document_ids=["doc-writing"],
        )

        self.assertEqual(capture["error_detail"], {"detail": "Ollama unavailable"})
        self.assertEqual(capture["request_document_ids"], ["doc-writing"])

    def test_upload_capture_omits_large_extraction_debug(self) -> None:
        result = {
            "filename": "sample.pdf",
            "http_status": 200,
            "duration_seconds": 1.2,
            "response": {
                "message": "Processed 3 chunks",
                "file_name": "sample.pdf",
                "document_id": "doc-1",
                "document_type": "ielts_reading",
                "chunks_processed": 3,
                "collection_stats": {"documents": 1, "chunks": 3},
                "debug": {
                    "timing": {"upload": {"total_seconds": 1.2}},
                    "structure": {"passages_found": 1},
                    "extraction": {"pages": [{"large": "payload"}]},
                },
            },
        }
        compact = compact_upload_result(result)
        self.assertEqual(compact["response"]["debug"]["structure"]["passages_found"], 1)
        self.assertNotIn("extraction", compact["response"]["debug"])

    def test_case_selection_rejects_unknown_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown case IDs"):
            select_cases(self.manifest, ["not-a-real-case"])


if __name__ == "__main__":
    unittest.main()
