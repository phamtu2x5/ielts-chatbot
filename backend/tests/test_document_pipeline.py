import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.document_pipeline.config import DocumentPipelineConfig
from app.document_pipeline.ielts import IELTSStructureParser, StructuredChunker
from app.document_pipeline.extractors.text import TextExtractor
from app.document_pipeline.models import DocumentElement, ProcessedDocument, ProcessedPage
from app.document_pipeline.ocr import OCRProcessor, OCRResult
from app.document_pipeline.reconciliation import NativeOCRReconciler
from app.document_pipeline.routing import FileRouter


def make_element(element_id: str, page: int, text: str, source: str = "native_pdf") -> DocumentElement:
    return DocumentElement(
        element_id=element_id,
        page=page,
        type="paragraph",
        raw_text=text,
        normalized_text=text,
        source=source,
        confidence=0.95,
    )


def make_document(pages: list[ProcessedPage]) -> ProcessedDocument:
    return ProcessedDocument(
        document_id="doc-1",
        filename="reading.pdf",
        mime_type="application/pdf",
        parser_version="test",
        metadata={},
        pages=pages,
    )


class NativeOCRReconcilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reconciler = NativeOCRReconciler(DocumentPipelineConfig())

    def test_duplicate_page_ocr_is_kept_only_as_page_provenance(self) -> None:
        text = "Questions 1-4 Wine is popular in Australia because it is healthy."
        native = make_element("p1-e1", 1, text)
        overlay = make_element("p1-e2", 1, text, source="pdf_page_ocr")
        overlay.type = "ocr_overlay"
        page = ProcessedPage(1, "native_pdf_plus_ocr", 0.9, [native, overlay])

        document = self.reconciler.reconcile(make_document([page]))

        self.assertEqual(len(document.pages[0].elements), 1)
        self.assertEqual(document.pages[0].metadata["duplicates_removed"], 1)
        self.assertEqual(document.pages[0].metadata["alternative_sources"][0]["source"], "pdf_page_ocr")

    def test_full_page_ocr_is_not_mislabeled_as_supplement(self) -> None:
        ocr = make_element("p1-e1", 1, "Scanned IELTS page", source="image_ocr")
        page = ProcessedPage(1, "image_ocr", 0.8, [ocr])

        document = self.reconciler.reconcile(make_document([page]))

        self.assertEqual(document.pages[0].elements[0].type, "paragraph")
        self.assertEqual(document.pages[0].processing_route, "image_ocr")


class FileRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DocumentPipelineConfig()
        self.router = FileRouter(self.config)

    def test_supported_extension_wins_over_incorrect_mime_type(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            route = self.router.route(Path(handle.name), "reading.pdf", "text/plain")

        self.assertEqual(route, "pdf")

    def test_mime_fallback_is_only_used_without_extension(self) -> None:
        with tempfile.NamedTemporaryFile() as handle:
            route = self.router.route(Path(handle.name), "upload", "application/pdf")

        self.assertEqual(route, "pdf")

    def test_unsupported_extension_is_rejected_even_with_image_mime(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".bin") as handle:
            with self.assertRaises(ValueError):
                self.router.route(Path(handle.name), "upload.bin", "image/png")

    def test_invalid_utf8_text_is_rejected(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt") as handle:
            Path(handle.name).write_bytes(b"\xff\xfe\x00")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                TextExtractor(self.config).extract(
                    Path(handle.name),
                    "invalid.txt",
                    "text/plain",
                    "doc",
                )


class OCRProcessorTests(unittest.TestCase):
    def test_tesseract_fallback_keeps_paddle_failure_diagnostics(self) -> None:
        processor = OCRProcessor(DocumentPipelineConfig())
        small = OCRResult("", 0.0, "paddleocr_error", {"error": "small failed"})
        medium = OCRResult("", 0.0, "paddleocr_error", {"error": "medium failed"})
        tesseract = OCRResult("fallback text", 0.7, "tesseract", {})

        with patch.object(processor, "_image_to_text_with_paddle", side_effect=[small, medium]):
            with patch.object(processor, "_image_to_text_with_tesseract", return_value=tesseract):
                result = processor.image_to_text(Image.new("RGB", (20, 20), "white"))

        self.assertEqual(result.engine, "tesseract")
        self.assertEqual(
            [attempt["error"] for attempt in result.metadata["cascade_attempts"]],
            ["small failed", "medium failed"],
        )


class IELTSStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DocumentPipelineConfig(
            chunk_target_tokens=40,
            chunk_max_tokens=50,
            chunk_overlap_tokens=10,
        )

    def test_group_ranges_drive_diagnostics_for_visual_questions(self) -> None:
        passage_one = "First Passage. " + "This passage discusses wine production and history. " * 30
        questions_one = (
            "Questions 1-4 Do the following statements agree with the passage? "
            "Write TRUE, FALSE or NOT GIVEN. "
            "1. Wine is popular. 2. Yeast is white. 3. Wine came from the Near East. 4. Blends are cheaper."
        )
        passage_two = "Second Passage. " + "This passage discusses management and shared vision. " * 30
        questions_two = "Questions 5-8 Complete the table below using NO MORE THAN TWO WORDS."
        document = make_document(
            [
                ProcessedPage(
                    1,
                    "native_pdf",
                    0.95,
                    [
                        make_element("p1-e1", 1, passage_one),
                        make_element("p1-e2", 1, questions_one),
                    ],
                ),
                ProcessedPage(
                    2,
                    "native_pdf",
                    0.95,
                    [
                        make_element("p2-e1", 2, passage_two),
                        make_element("p2-e2", 2, questions_two),
                    ],
                ),
            ]
        )

        structured = IELTSStructureParser(self.config).parse(document)

        self.assertEqual(len(structured.passages), 2)
        self.assertEqual(structured.diagnostics["questions_found"], 8)
        self.assertEqual(structured.diagnostics["individual_questions_found"], 4)
        self.assertEqual(structured.diagnostics["missing_questions"], [])

    def test_long_unpunctuated_passage_is_split_below_limit(self) -> None:
        passage = "Long Passage " + "word " * 700
        questions = "Questions 1-1 Choose the correct answer. 1. What is the topic?"
        document = make_document(
            [
                ProcessedPage(
                    1,
                    "native_pdf",
                    0.95,
                    [make_element("p1-e1", 1, passage), make_element("p1-e2", 1, questions)],
                )
            ]
        )
        structured = IELTSStructureParser(self.config).parse(document)

        chunks = StructuredChunker(self.config).chunk(document, structured)
        passage_chunks = [chunk for chunk in chunks if chunk.metadata["unit_type"] == "passage"]

        self.assertGreater(len(passage_chunks), 1)
        self.assertTrue(all(chunk.token_count <= self.config.chunk_max_tokens + 10 for chunk in passage_chunks))

    def test_title_after_last_question_is_assigned_to_next_passage(self) -> None:
        first = "First Passage. " + "Background text. " * 100
        first_questions = "Questions 1-1 Choose the answer. 1. First question?\nDestination Mars"
        second = "Mars is the closest potentially habitable planet. " * 100
        second_questions = "Questions 2-2 Choose the answer. 2. Second question?"
        document = make_document(
            [
                ProcessedPage(
                    1,
                    "native_pdf",
                    0.95,
                    [make_element("p1-e1", 1, first), make_element("p1-e2", 1, first_questions)],
                ),
                ProcessedPage(
                    2,
                    "native_pdf",
                    0.95,
                    [make_element("p2-e1", 2, second), make_element("p2-e2", 2, second_questions)],
                ),
            ]
        )

        structured = IELTSStructureParser(self.config).parse(document)

        self.assertEqual(len(structured.passages), 2)
        self.assertEqual(structured.passages[1].title, "Destination Mars")


if __name__ == "__main__":
    unittest.main()
