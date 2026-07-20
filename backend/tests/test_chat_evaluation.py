import json
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from backend.tools.chat_evaluation import (
    case_request_document_ids,
    capture_case,
    compact_upload_result,
    verify_corpus,
)


MANIFEST_PATH = BACKEND_DIR / "evaluation" / "chat_corpus_v2.json"
CORPUS_DIR = REPO_DIR / "docs"
REQUIRED_CATEGORIES = {
    "direct_router",
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

    def test_manifest_is_valid_for_current_corpus(self) -> None:
        documents = self.manifest["documents"]
        verified = verify_corpus(self.manifest, CORPUS_DIR)
        self.assertEqual(len(verified), len(documents))
        cases = self.manifest["cases"]
        ids = [case["id"] for case in cases]
        self.assertEqual(len(ids), len(set(ids)))
        filenames = {document["filename"] for document in documents}
        categories = {case["category"] for case in cases}
        self.assertTrue(REQUIRED_CATEGORIES.issubset(categories))
        for case in cases:
            self.assertTrue(case["query"].strip())
            self.assertTrue(set(case["target_files"]).issubset(filenames))
            self.assertIn(case.get("request_scope", "target_files"), {"target_files", "all_uploaded"})
            self.assertNotIn("expected_intent", case)
            self.assertNotIn("answer_terms", case)

    def test_direct_router_cases_can_run_with_all_uploaded_documents_active(self) -> None:
        uploaded = {"reading.pdf": "doc-reading", "writing.png": "doc-writing"}
        document_ids = case_request_document_ids(
            {
                "target_files": [],
                "request_scope": "all_uploaded",
            },
            uploaded,
        )
        self.assertEqual(document_ids, ["doc-reading", "doc-writing"])

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
        self.assertEqual(capture["resolved_document_ids"], [])
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

    def test_capture_preserves_backend_scope_and_conversation_state(self) -> None:
        case = {
            "id": "state",
            "category": "semantic_qa",
            "query": "Tại sao?",
            "target_files": ["sample.pdf"],
        }
        state = {
            "last_route": "rag",
            "last_intent": "semantic_qa",
            "rag_affinity": {
                "document_ids": ["doc-1"],
                "passage_numbers": [2],
                "question_ranges": [],
            },
        }
        capture = capture_case(
            case,
            {
                "http_status": 200,
                "duration_seconds": 0.1,
                "response": {
                    "response": "Vì ...",
                    "route_used": "vector_rag",
                    "sources": [],
                    "conversation_state": state,
                    "debug": {
                        "document_resolution": {
                            "resolved_document_ids": ["doc-1"],
                        }
                    },
                },
            },
            request_document_ids=["doc-1"],
        )
        self.assertEqual(capture["conversation_state"], state)
        self.assertEqual(capture["resolved_document_ids"], ["doc-1"])

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

if __name__ == "__main__":
    unittest.main()
