import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from tools.extraction_regression import (
    evaluate_document,
    load_manifest,
    status_from_checks,
    verify_fixtures,
)
from tools.granite_docling_benchmark import (
    build_ab_comparison,
    normalize_docling_markdown,
    select_granite_dtype,
)


class ExtractionRegressionTests(unittest.TestCase):
    def test_fixture_hash_is_checked_before_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory)
            fixture = corpus / "sample.txt"
            fixture.write_text("IELTS", encoding="utf-8")
            digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
            manifest = {
                "documents": [
                    {
                        "filename": fixture.name,
                        "sha256": digest,
                        "kind": "ielts_reading",
                    }
                ]
            }

            passed = verify_fixtures(manifest, corpus)
            fixture.write_text("changed", encoding="utf-8")
            failed = verify_fixtures(manifest, corpus)

        self.assertTrue(passed["ok"])
        self.assertFalse(failed["ok"])
        self.assertEqual(failed["fixtures"][0]["status"], "hash_mismatch")

    def test_manifest_requires_identity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"documents": [{"filename": "sample.pdf"}]}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "filename, sha256, and kind"):
                load_manifest(path)

    def test_reading_evaluation_uses_passage_assignment_and_question_coverage(self) -> None:
        canonical = {
            "metadata": {
                "page_count": 2,
                "ielts_structure": {
                    "passages": [
                        {
                            "title": "Passage title",
                            "question_groups": [
                                {
                                    "question_start": 1,
                                    "question_end": 2,
                                    "visual_element": None,
                                }
                            ],
                        }
                    ],
                    "diagnostics": {},
                },
            },
            "pages": [{"page_number": 1}, {"page_number": 2}],
        }
        fixture = {
            "kind": "ielts_reading",
            "expected": {
                "page_count": 2,
                "reading": {
                    "passages": [
                        {
                            "number": 1,
                            "title": "Passage title",
                            "question_groups": [[1, 2]],
                        }
                    ],
                    "covered_question_numbers": {"start": 1, "end": 2},
                    "forbidden_titles": ["Instructions"],
                },
            },
        }

        checks = evaluate_document(canonical, [{"chunk_id": "c1"}], fixture)

        self.assertEqual(status_from_checks(checks), "passed")

    def test_writing_collection_is_reported_as_unsupported_until_sections_exist(self) -> None:
        canonical = {
            "metadata": {"page_count": 1},
            "pages": [{"page_number": 1}],
        }
        fixture = {
            "kind": "ielts_writing_collection",
            "expected": {
                "page_count": 1,
                "writing_collection": {"task_count": 1, "sample_answer_count": 1},
            },
        }

        checks = evaluate_document(canonical, [{"chunk_id": "c1"}], fixture)

        self.assertEqual(status_from_checks(checks), "unsupported")

    def test_granite_markdown_normalization_preserves_tables_and_removes_heading_markup(self) -> None:
        markdown = "# Passage title\n\n<!-- image -->\n\n| A | B |\n|---|---|\n| 1 | 2 |"

        normalized = normalize_docling_markdown(markdown)

        self.assertTrue(normalized.startswith("Passage title"))
        self.assertNotIn("<!-- image -->", normalized)
        self.assertIn("| A | B |", normalized)

    def test_granite_dtype_honors_explicit_value_without_loading_model(self) -> None:
        self.assertEqual(select_granite_dtype("float32"), "float32")
        self.assertEqual(select_granite_dtype("bfloat16"), "bfloat16")

    def test_ab_comparison_keeps_each_engine_result_separate(self) -> None:
        baseline = {
            "status": "degraded",
            "documents": [
                {
                    "filename": "sample.pdf",
                    "status": "degraded",
                    "duration_seconds": 2.0,
                    "checks": [{"name": "passage_count", "status": "passed"}],
                }
            ],
        }
        granite = {
            "status": "passed",
            "documents": [
                {
                    "filename": "sample.pdf",
                    "status": "passed",
                    "duration_seconds": 4.0,
                    "checks": [{"name": "passage_count", "status": "passed"}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_dir = root / "current"
            granite_dir = root / "granite"
            baseline_dir.mkdir()
            granite_dir.mkdir()
            (baseline_dir / "regression_summary.json").write_text(json.dumps(baseline), encoding="utf-8")
            (granite_dir / "regression_summary.json").write_text(json.dumps(granite), encoding="utf-8")

            comparison = build_ab_comparison(baseline_dir, granite_dir)

        result = comparison["documents"][0]
        self.assertEqual(result["current_pipeline"]["duration_seconds"], 2.0)
        self.assertEqual(result["granite_docling"]["duration_seconds"], 4.0)
        self.assertFalse(comparison["production_integrated"])


if __name__ == "__main__":
    unittest.main()
