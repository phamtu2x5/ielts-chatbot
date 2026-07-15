import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion_debug import IngestionDebugStore


class IngestionDebugStoreTests(unittest.TestCase):
    def test_persists_completed_trace_as_atomic_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ingestion_debug.json"
            store = IngestionDebugStore(path)
            store.start("request-1", "sample.pdf", {"ocr": {"model_size": "medium"}})
            store.update(
                "request-1",
                status="completed",
                stage="completed",
                timing={
                    "upload": {"total_seconds": 12.5},
                    "extraction": {"pages": [{"page": 1, "ocr_seconds": 8.0}]},
                    "embedding": {"embedding_seconds": 1.2},
                },
            )

            payload = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["pipeline"]["ocr"]["model_size"], "medium")
            self.assertEqual(payload["timing"]["extraction"]["pages"][0]["ocr_seconds"], 8.0)
            self.assertIn("finished_at", payload)
            self.assertGreaterEqual(payload["elapsed_seconds"], 0.0)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_persists_processing_state_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = IngestionDebugStore(Path(temp_dir) / "ingestion_debug.json")
            store.start("request-1", "sample.pdf", {})
            time.sleep(0.01)
            store.update(
                "request-1",
                status="processing",
                stage="process_file",
                timing={"upload": {"save_file_seconds": 0.1}},
            )

            payload = store.read()

            self.assertEqual(payload["status"], "processing")
            self.assertEqual(payload["stage"], "process_file")
            self.assertNotIn("finished_at", payload)

    def test_ignores_updates_from_an_older_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = IngestionDebugStore(Path(temp_dir) / "ingestion_debug.json")
            store.start("request-new", "new.pdf", {})

            result = store.update("request-old", status="failed")

            self.assertIsNone(result)
            self.assertEqual(store.read()["request_id"], "request-new")


if __name__ == "__main__":
    unittest.main()
