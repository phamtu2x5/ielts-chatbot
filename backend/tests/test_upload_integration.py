import sys
import tempfile
import types
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from starlette.datastructures import Headers, UploadFile


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    import aiofiles  # noqa: F401
except ImportError:
    class _AsyncFile:
        def __init__(self, path: Path, mode: str) -> None:
            self._path = path
            self._mode = mode
            self._handle = None

        async def __aenter__(self):
            self._handle = self._path.open(self._mode)
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            self._handle.close()

        async def write(self, data: bytes) -> int:
            return self._handle.write(data)

    sys.modules["aiofiles"] = types.SimpleNamespace(
        open=lambda path, mode: _AsyncFile(Path(path), mode)
    )

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

from app import main
from app.document_pipeline.models import DocumentChunk, ProcessedDocument, ProcessedPage


class _FakeProcessor:
    class Config:
        max_upload_mb = 1

    config = Config()

    def process_file(
        self,
        file_path: Path,
        filename: str,
        content_type: str | None,
    ) -> tuple[ProcessedDocument, list[DocumentChunk]]:
        if file_path.read_bytes() != b"sample content":
            raise AssertionError("Upload content was not saved before extraction.")
        document = ProcessedDocument(
            document_id="doc-1",
            filename=filename,
            mime_type=content_type or "text/plain",
            parser_version="1.10.0",
            metadata={
                "document_type": "ielts_reading",
                "timing": {
                    "process_file": {"total_seconds": 0.01},
                    "chunking": {"chunks": 1},
                },
                "extraction_report": {"pages": []},
                "ielts_structure": {"diagnostics": {}, "outline": {}},
            },
            pages=[ProcessedPage(page_number=1, processing_route="text", quality_score=1.0)],
        )
        chunk = DocumentChunk(
            chunk_id="doc-1-c1",
            document_id="doc-1",
            source_file=filename,
            pages=[1],
            element_ids=[],
            heading_path=[],
            text="sample content",
            token_count=2,
            min_confidence=1.0,
            chunk_index=0,
            metadata={"parser_version": "1.10.0"},
        )
        return document, [chunk]


class _FakeStore:
    def __init__(self) -> None:
        self.last_upsert_timing = {"embedding_seconds": 0.01, "chunks": 1}
        self.received_chunks: list[dict] = []

    def upsert(self, chunks: list[dict], source_file: str) -> int:
        self.received_chunks = chunks
        if source_file != "sample.txt":
            raise AssertionError("Source filename was not passed to the store.")
        return len(chunks)

    def stats(self) -> dict:
        return {"documents": 1, "chunks": 1, "embedding_model": "test"}


class _FakeChatStore:
    def __init__(self, catalog: list[dict]) -> None:
        self.catalog = catalog

    def stats(self) -> dict:
        return {"documents": len(self.catalog), "chunks": len(self.catalog), "embedding_model": "test"}

    def document_catalog(self, document_ids=None) -> list[dict]:
        if not document_ids:
            return self.catalog
        allowed = set(document_ids)
        return [
            item
            for item in self.catalog
            if allowed.intersection(item.get("document_ids", []))
        ]

    def probe_with_catalog(self, query, top_k, document_ids=None):
        return (
            {
                "results": [],
                "has_hits": False,
                "has_strong_hits": False,
                "has_document_intent": True,
                "is_overview": False,
            },
            self.document_catalog(document_ids),
        )

    def structured_lookup(self, query, intent, top_k, document_ids=None):
        return []

    def search(self, query, top_k, document_ids=None):
        return []


class UploadIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_connects_processor_chunks_and_store(self) -> None:
        store = _FakeStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            upload = UploadFile(
                file=BytesIO(b"sample content"),
                filename="sample.txt",
                headers=Headers({"content-type": "text/plain"}),
            )
            with (
                patch.object(main, "UPLOAD_DIR", Path(temp_dir)),
                patch.object(main, "DOCUMENT_PROCESSOR", _FakeProcessor()),
                patch.object(main, "get_store", return_value=store),
            ):
                response = await main.upload_document(upload)

            self.assertEqual(response.document_id, "doc-1")
            self.assertEqual(response.document_type, "ielts_reading")
            self.assertEqual(response.chunks_processed, 1)
            self.assertEqual(store.received_chunks[0]["metadata"]["parser_version"], "1.10.0")
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    async def test_document_query_without_sources_does_not_fall_back_to_base_model(self) -> None:
        catalog = [
            {
                "source_file": "sample.pdf",
                "document_ids": ["doc-1"],
                "mime_types": ["application/pdf"],
            }
        ]
        with patch.object(main, "get_store", return_value=_FakeChatStore(catalog)):
            prepared = await main.prepare_chat(
                main.ChatRequest(
                    message="Nội dung Questions 1-4 trong sample.pdf là gì?",
                    document_ids=["doc-1"],
                )
            )

        self.assertEqual(prepared.route_used, "vector_rag_no_match")
        self.assertIsNone(prepared.prompt)
        self.assertEqual(prepared.static_response, main.NO_RAG_MATCH_RESPONSE)

    async def test_explicit_document_scope_never_routes_semantic_query_to_base_model(self) -> None:
        catalog = [
            {
                "source_file": "reading.pdf",
                "document_ids": ["doc-1"],
                "mime_types": ["application/pdf"],
            }
        ]
        with patch.object(main, "get_store", return_value=_FakeChatStore(catalog)):
            prepared = await main.prepare_chat(
                main.ChatRequest(
                    message="How did the fence affect kangaroos?",
                    document_ids=["doc-1"],
                )
            )

        self.assertEqual(prepared.route_used, "vector_rag_no_match")
        self.assertNotEqual(prepared.query_intent, "direct")
        self.assertTrue(prepared.debug["target_resolution"]["document_grounded"])

    async def test_ambiguous_question_range_requests_a_document_choice(self) -> None:
        catalog = [
            {"source_file": "reading-2.pdf", "document_ids": ["doc-2"], "mime_types": ["application/pdf"]},
            {"source_file": "reading-4.pdf", "document_ids": ["doc-4"], "mime_types": ["application/pdf"]},
        ]
        with patch.object(main, "get_store", return_value=_FakeChatStore(catalog)):
            prepared = await main.prepare_chat(main.ChatRequest(message="Liệt kê Questions 1-4"))

        self.assertEqual(prepared.route_used, "vector_rag_ambiguous_document")
        self.assertIn("Vui lòng nêu tên file", prepared.static_response)

    async def test_static_table_operations_do_not_collapse_to_first_cell(self) -> None:
        source = {
            "source_file": "writing.png",
            "pages": [1],
            "metadata": {
                "unit_type": "writing_table",
                "table": {
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
                },
            },
        }

        cell = main.static_response_for_sources(
            "Smartphone Ownership của Country B năm 2024 là bao nhiêu?",
            "table_cell",
            [source],
        )
        calculation = main.static_response_for_sources(
            "Từ bảng, quốc gia nào tăng Internet Access nhiều nhất từ 2019 đến 2024? Trình bày phép tính.",
            "table_calculation",
            [source],
        )
        comparison = main.static_response_for_sources(
            "So sánh Internet Access và Smartphone Ownership của Country A trong cả hai năm.",
            "table_comparison",
            [source],
        )

        self.assertTrue(cell.startswith("94"))
        self.assertIn("A: 96 - 78 = 18", calculation)
        self.assertIn("B: 89 - 61 = 28", calculation)
        self.assertIn("C: 75 - 42 = 33", calculation)
        self.assertIn("| A | 78 | 96 | 82 | 99 |", comparison)

    def test_solve_context_rejects_missing_multiple_choice_options(self) -> None:
        incomplete = [
            {
                "text": "From the list below choose the most suitable title.",
                "metadata": {"unit_type": "question"},
            }
        ]
        complete = [
            {
                "text": "Choose the correct letter. A first title B second title C third title D fourth title",
                "metadata": {"unit_type": "question"},
            }
        ]

        self.assertEqual(main.solve_context_issue(incomplete), "missing_answer_options")
        self.assertIsNone(main.solve_context_issue(complete))


if __name__ == "__main__":
    unittest.main()
