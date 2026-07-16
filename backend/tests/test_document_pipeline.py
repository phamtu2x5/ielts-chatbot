import os
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
from app.document_pipeline.chunking import SemanticChunker
from app.document_pipeline.ielts import IELTSStructureParser, StructuredChunker
from app.document_pipeline.extractors.text import TextExtractor
from app.document_pipeline.layout import DocLayoutDetector
from app.document_pipeline.models import DocumentElement, ProcessedDocument, ProcessedPage
from app.document_pipeline.ocr import OCRProcessor, OCRResult
from app.document_pipeline.reconciliation import NativeOCRReconciler
from app.document_pipeline.routing import FileRouter
from app.document_pipeline.visual import WritingTaskTableParser


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
    def test_ocr_engine_must_be_rapidocr(self) -> None:
        with patch.dict(os.environ, {"OCR_ENGINE": "easyocr"}):
            with self.assertRaisesRegex(ValueError, "OCR_ENGINE must be rapidocr"):
                DocumentPipelineConfig()

    def test_ocr_runtime_must_be_torch(self) -> None:
        with patch.dict(os.environ, {"OCR_RUNTIME": "onnxruntime"}):
            with self.assertRaisesRegex(ValueError, "OCR_RUNTIME must be torch"):
                DocumentPipelineConfig()

    def test_ocr_device_must_be_cuda(self) -> None:
        with patch.dict(os.environ, {"OCR_DEVICE": "cpu"}):
            with self.assertRaisesRegex(ValueError, "OCR_DEVICE must be cuda"):
                DocumentPipelineConfig()

    def test_cuda_device_id_is_parsed_from_config(self) -> None:
        processor = OCRProcessor(DocumentPipelineConfig(ocr_device="cuda:2"))

        self.assertEqual(processor._cuda_device_id(), 2)

    def test_rapidocr_failure_returns_diagnostics_without_external_fallback(self) -> None:
        processor = OCRProcessor(DocumentPipelineConfig())
        failure = OCRResult("", 0.0, "rapidocr_error", {"error": "ocr failed"})

        with patch.object(processor, "_image_to_text_with_rapidocr", return_value=failure):
            result = processor.image_to_text(Image.new("RGB", (20, 20), "white"))

        self.assertEqual(result.engine, "rapidocr_failed")
        self.assertFalse(result.text)
        self.assertEqual(result.metadata["attempt"]["error"], "ocr failed")

    def test_rapidocr_output_schema_is_normalized(self) -> None:
        class RapidOCRLikeResult:
            txts = ("Hello", "IELTS")
            scores = (0.9, 0.8)
            boxes = [[[0, 0], [10, 0], [10, 10], [0, 10]]]

        processor = OCRProcessor(DocumentPipelineConfig())

        texts, scores, boxes = processor._extract_rapidocr_result(RapidOCRLikeResult())

        self.assertEqual(texts, ["Hello", "IELTS"])
        self.assertEqual(scores, [0.9, 0.8])
        self.assertEqual(boxes, [[[0, 0], [10, 0], [10, 10], [0, 10]]])


class DocLayoutDetectorTests(unittest.TestCase):
    def test_doclayout_yolo_output_schema_is_normalized(self) -> None:
        class Boxes:
            xyxy = [[0, 0, 100, 50]]
            conf = [0.91]
            cls = [2]

        class Result:
            names = {2: "table"}
            boxes = Boxes()

        detector = DocLayoutDetector(DocumentPipelineConfig())

        regions = detector._extract_regions([Result()])

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].type, "table")
        self.assertEqual(regions[0].confidence, 0.91)
        self.assertEqual(regions[0].bbox, [0.0, 0.0, 100.0, 50.0])


class WritingVisualParserTests(unittest.TestCase):
    def test_spatial_table_parser_uses_layout_region_and_ocr_boxes(self) -> None:
        def line(text: str, x: int, y: int, width: int = 120) -> dict:
            return {
                "text": text,
                "confidence": 0.98,
                "bbox": [[x, y], [x + width, y], [x + width, y + 24], [x, y + 24]],
            }

        text = (
            "You should spend about 20 minutes on this task.\n"
            "The table below shows the percentage of households with internet access and smartphone ownership.\n"
            "Summarise the information by selecting and reporting the main features.\n"
            "Write at least 150 words."
        )
        ocr_lines = [
            line("Country", 20, 320),
            line("Internet Access", 190, 320, 150),
            line("Internet Access", 390, 320, 150),
            line("Smartphone Ownership", 570, 320, 180),
            line("Smartphone Ownership", 770, 320, 180),
            line("2019 (%)", 210, 360),
            line("2024 (%)", 410, 360),
            line("2019 (%)", 610, 360),
            line("2024 (%)", 810, 360),
            line("A", 30, 430, 30),
            line("78", 220, 430, 40),
            line("96", 420, 430, 40),
            line("82", 620, 430, 40),
            line("99", 820, 430, 40),
            line("B", 30, 480, 30),
            line("61", 220, 480, 40),
            line("89", 420, 480, 40),
            line("67", 620, 480, 40),
            line("94", 820, 480, 40),
            line("C", 30, 530, 30),
            line("42", 220, 530, 40),
            line("75", 420, 530, 40),
            line("48", 620, 530, 40),
            line("83", 820, 530, 40),
        ]

        parsed = WritingTaskTableParser().parse(
            text,
            ocr_lines=ocr_lines,
            layout_regions=[{"type": "table", "confidence": 0.93, "bbox": [0, 300, 1000, 600]}],
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(
            parsed.table["columns"],
            [
                "Country",
                "Internet Access 2019 (%)",
                "Internet Access 2024 (%)",
                "Smartphone Ownership 2019 (%)",
                "Smartphone Ownership 2024 (%)",
            ],
        )
        self.assertEqual(parsed.table["rows"], [["A", 78, 96, 82, 99], ["B", 61, 89, 67, 94], ["C", 42, 75, 48, 83]])
        self.assertEqual(parsed.table["source"], "doclayout_yolo+rapidocr_boxes")
        self.assertIn("percentage of households", parsed.prompt["description"])
        self.assertIn("main features", parsed.prompt["instruction"])

    def test_writing_task_table_parser_extracts_rows_and_columns(self) -> None:
        text = (
            "WRITING TASK 1 You should spend about 20 minutes on this task. "
            "The table below shows the percentages of people in three countries who had Internet Access "
            "and Smartphone Ownership in 2019 and 2024. Summarise the information by selecting and reporting "
            "the main features, and make comparisons where relevant. Write at least 150 words. "
            "Country Internet Access 2019 Internet Access 2024 Smartphone Ownership 2019 Smartphone Ownership 2024 "
            "A 78% 96% 82% 99% B 61% 89% 67% 94% Cc 42% 75% 48% 83%"
        )

        parsed = WritingTaskTableParser().parse(text)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.document_type, "ielts_writing_task_1")
        self.assertEqual(parsed.table["columns"][0], "Country")
        self.assertEqual(parsed.table["rows"], [["A", 78, 96, 82, 99], ["B", 61, 89, 67, 94], ["C", 42, 75, 48, 83]])

    def test_spatial_table_supports_numeric_row_labels(self) -> None:
        def line(text: str, x: int, y: int) -> dict:
            return {
                "text": text,
                "confidence": 0.97,
                "bbox": [[x, y], [x + 60, y], [x + 60, y + 20], [x, y + 20]],
            }

        parsed = WritingTaskTableParser().parse(
            "The table below shows annual values. Summarise the information.",
            ocr_lines=[
                line("Year", 20, 100),
                line("A", 220, 100),
                line("B", 420, 100),
                line("2019", 20, 160),
                line("45", 220, 160),
                line("60", 420, 160),
                line("2024", 20, 210),
                line("55", 220, 210),
                line("70", 420, 210),
            ],
            layout_regions=[{"type": "table", "confidence": 0.9, "bbox": [0, 80, 500, 260]}],
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.table["rows"], [["2019", 45, 60], ["2024", 55, 70]])

    def test_writing_task_chunker_emits_prompt_table_and_rows(self) -> None:
        parser = WritingTaskTableParser()
        parsed = parser.parse(
            "The table below shows Internet Access and Smartphone Ownership in 2019 and 2024. "
            "Summarise the information. Write at least 150 words. "
            "Country Internet Access 2019 Internet Access 2024 Smartphone Ownership 2019 Smartphone Ownership 2024 "
            "A 78 96 82 99 B 61 89 67 94 C 42 75 48 83"
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        document = ProcessedDocument(
            document_id="writing-doc",
            filename="writing.png",
            mime_type="image/png",
            parser_version="test",
            metadata={
                "document_type": parsed.document_type,
                "task_type": parsed.task_type,
            },
            pages=[
                ProcessedPage(
                    1,
                    "image_ocr",
                    0.9,
                    [
                        DocumentElement(
                            "p1-e1",
                            1,
                            "writing_prompt",
                            parsed.prompt_text(),
                            parsed.prompt_text(),
                            "image_ocr_structured",
                            0.9,
                        ),
                        DocumentElement(
                            "p1-e2",
                            1,
                            "table",
                            parsed.table_markdown(),
                            parsed.table_markdown(),
                            "image_ocr_structured",
                            0.9,
                            metadata={"table": parsed.table},
                        ),
                    ],
                )
            ],
        )

        chunks = SemanticChunker(DocumentPipelineConfig()).chunk(document)

        self.assertEqual([chunk.metadata["unit_type"] for chunk in chunks[:2]], ["writing_prompt", "writing_table"])
        self.assertIn("table_row", [chunk.metadata["unit_type"] for chunk in chunks])
        self.assertEqual(chunks[1].metadata["table"]["rows"][1], ["B", 61, 89, 67, 94])

    def test_writing_document_has_visual_structure_diagnostics(self) -> None:
        table = {
            "type": "table",
            "columns": ["Country", "Value"],
            "rows": [["A", 10]],
            "confidence": 0.9,
        }
        document = ProcessedDocument(
            document_id="writing-doc",
            filename="writing.png",
            mime_type="image/png",
            parser_version="test",
            metadata={
                "document_type": "ielts_writing_task_1",
                "task_type": "academic_task_1_table",
                "visual_structure": {"prompt": {}, "visual_elements": [table]},
            },
            pages=[ProcessedPage(1, "image_ocr", 0.9, [])],
        )

        structured = IELTSStructureParser(DocumentPipelineConfig()).parse(document)

        self.assertEqual(structured.outline["document_type"], "ielts_writing_task_1")
        self.assertEqual(structured.diagnostics["visual_elements_found"], 1)
        self.assertEqual(structured.diagnostics["tables_found"], 1)


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

    def test_duplicate_question_group_does_not_leak_into_next_passage(self) -> None:
        passage_one = "Make That Wine!\n" + "Wine production background. " * 30
        questions = (
            "Questions 11-13 Choose the correct letter, A, B, C, or D. "
            "11. First item A one. B two. C three. D four. "
            "12. Second item A one. B two. C three. D four. "
            "13. Wine A popular. B discussed. C classified. D rationed."
        )
        passage_two = "That Vision Thing\n" + "Management and motivation are discussed here. " * 30
        document = make_document(
            [
                ProcessedPage(
                    1,
                    "native_pdf_plus_ocr_reconciled",
                    0.95,
                    [
                        make_element("p1-e1", 1, passage_one),
                        make_element("p1-e2", 1, questions),
                        make_element("p1-e3", 1, questions, source="pdf_page_ocr"),
                        make_element("p1-e4", 1, passage_two),
                        make_element("p1-e5", 1, "Questions 14-14 Answer the question. 14. What motivates staff?"),
                    ],
                )
            ]
        )

        structured = IELTSStructureParser(self.config).parse(document)

        self.assertEqual(len(structured.passages), 2)
        self.assertEqual(structured.passages[1].title, "That Vision Thing")
        self.assertNotIn("Wine A popular", structured.passages[1].text)

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

    def test_passage_chunks_do_not_keep_previous_question_sections(self) -> None:
        text = (
            "Make That Wine! Australia is a nation of wine drinkers. "
            "Questions 1-4 Do the following statements agree with the information given in Reading Passage One? "
            "1. Wine is popular in Australia because it is healthy. "
            "2. Yeast is white-coloured. "
            "3. Wine is popular in the Near East. "
            "4. Blended wines are usually cheaper. "
            "Questions 5-5 Complete the table. "
            "That Vision Thing In the past, management took a minor role in influencing motivation. "
            "People in organisations were considered personnel. "
            "Questions 6-6 Choose the correct letter, A, B, C, or D. "
            "6. With regard to envisioning, the author feels A critical. B contemptuous. C impartial. D suspicious. "
            "Destination Mars Mars is the closest potentially habitable planet. It has solid ground and frozen water. "
            "Questions 7-7 Do the following statements agree with the information? "
            "7. Mars has basic minerals."
        )
        document = make_document([ProcessedPage(1, "native_pdf", 0.95, [make_element("p1-e1", 1, text)])])

        structured = IELTSStructureParser(self.config).parse(document)
        chunks = StructuredChunker(self.config).chunk(document, structured)
        passage_chunks = [chunk for chunk in chunks if chunk.metadata["unit_type"] == "passage"]
        question_six = next(chunk for chunk in chunks if chunk.chunk_id.endswith("-question-6"))

        self.assertEqual([passage.title for passage in structured.passages], ["Make That Wine!", "That Vision Thing", "Destination Mars"])
        self.assertTrue(all("Questions 1-4" not in chunk.text for chunk in passage_chunks))
        self.assertTrue(all("Questions 6-6" not in chunk.text for chunk in passage_chunks))
        self.assertNotIn("Destination Mars", question_six.text)
        self.assertNotIn("It has solid ground", question_six.text)

    def test_instruction_only_gaps_do_not_create_passages(self) -> None:
        text = (
            "Make That Wine! Australia is a nation of wine drinkers. "
            "Wine is made by fermentation and can be classified in several ways. "
            "Questions 1-4 Do the following statements agree with the information given in Reading Passage One? "
            "1. Wine is popular in Australia because it is healthy. "
            "2. Yeast is white-coloured. "
            "3. Wine is popular in the Near East. "
            "4. Blended wines are usually cheaper. "
            "Questions 5-10 Complete the table. "
            "Choose NO MORE THAN TWO WORDS from the passage for each answer. "
            "Classification based on colour grape species location method. "
            "Questions 11-13 Choose the correct letter, A, B, C, or D. "
            "11. First multiple choice? A one. B two. C three. D four. "
            "12. Second multiple choice? A one. B two. C three. D four. "
            "13. Third multiple choice? A one. B two. C three. D four. "
            "That Vision Thing In the past, management took a minor role in influencing motivation. "
            "People in organisations were considered personnel. "
            "Questions 14-17 Answer the questions. Choose NO MORE THAN TWO WORDS from the passage for each answer. "
            "14. Broadly, what do staff need? "
            "15. Which people advise envisioning? "
            "16. What can a lack of vision cause? "
            "17. Which aspects are never shared? "
        )
        document = make_document([ProcessedPage(1, "native_pdf", 0.95, [make_element("p1-e1", 1, text)])])

        structured = IELTSStructureParser(self.config).parse(document)

        self.assertEqual([passage.title for passage in structured.passages], ["Make That Wine!", "That Vision Thing"])
        self.assertEqual(
            [
                (group.question_start, group.question_end)
                for group in structured.passages[0].question_groups
            ],
            [(1, 4), (5, 10), (11, 13)],
        )
        self.assertEqual(structured.diagnostics["instruction_as_title"], [])

    def test_visual_question_groups_emit_table_and_flowchart_chunks(self) -> None:
        text = (
            "A Practice Passage\n"
            "This passage explains a process with several categories and stages.\n"
            "Questions 1-2 Complete the table below. Choose NO MORE THAN TWO WORDS from the passage.\n"
            "| Category | Answer |\n"
            "| --- | --- |\n"
            "| Colour | 1 |\n"
            "| Location | 2 |\n"
            "Questions 3-4 Complete the flow chart. Choose NO MORE THAN TWO WORDS from the passage.\n"
            "Start -> 3 -> Result -> 4\n"
        )
        document = make_document([ProcessedPage(1, "native_pdf", 0.95, [make_element("p1-e1", 1, text)])])

        structured = IELTSStructureParser(self.config).parse(document)
        chunks = StructuredChunker(self.config).chunk(document, structured)
        table_chunk = next(chunk for chunk in chunks if chunk.metadata["unit_type"] == "table")
        flowchart_chunk = next(chunk for chunk in chunks if chunk.metadata["unit_type"] == "flowchart")

        self.assertEqual(structured.diagnostics["tables_found"], 1)
        self.assertEqual(structured.diagnostics["flowcharts_found"], 1)
        self.assertEqual(table_chunk.metadata["table"]["blank_question_numbers"], [1, 2])
        self.assertEqual(table_chunk.metadata["table"]["columns"], ["Category", "Answer"])
        self.assertEqual(flowchart_chunk.metadata["flowchart"]["blank_question_numbers"], [3, 4])
        self.assertGreater(len(flowchart_chunk.metadata["flowchart"]["edges"]), 0)

    def test_visual_question_group_keeps_low_confidence_fallback_when_layout_is_flat(self) -> None:
        text = (
            "A Practice Passage This passage explains a process. "
            "Questions 5-7 Complete the table below. Choose NO MORE THAN TWO WORDS from the passage. "
            "Category Answer Colour 5 Location 6 Method 7 "
        )
        document = make_document([ProcessedPage(1, "native_pdf", 0.95, [make_element("p1-e1", 1, text)])])

        structured = IELTSStructureParser(self.config).parse(document)
        chunks = StructuredChunker(self.config).chunk(document, structured)
        table_chunk = next(chunk for chunk in chunks if chunk.metadata["unit_type"] == "table")

        self.assertEqual(table_chunk.metadata["table"]["blank_question_numbers"], [5, 6, 7])
        self.assertEqual(table_chunk.metadata["table"]["rows"], [])
        self.assertLess(table_chunk.metadata["table"]["confidence"], 0.6)
        self.assertEqual(
            structured.diagnostics["low_confidence_visual_elements"][0]["question_range"],
            [5, 7],
        )

    def test_task_noise_is_removed_from_question_group(self) -> None:
        text = (
            "Destination Mars Mars is the closest potentially habitable planet. "
            "Questions 36-40 Give TWO examples of the following categories. "
            "Choose NO MORE THAN TWO WORDS from the passage for each example. "
            "Task 2 - Some people think children should receive full-time education."
        )
        document = make_document([ProcessedPage(1, "native_pdf", 0.95, [make_element("p1-e1", 1, text)])])

        structured = IELTSStructureParser(self.config).parse(document)
        group = structured.passages[0].question_groups[0]

        self.assertNotIn("Task 2", group.text)

    def test_tail_exclamation_is_not_used_as_next_passage_title(self) -> None:
        first = "First Topic " + "The passage introduces a broad subject. " * 30
        questions = "Questions 1-1 Choose the correct letter. 1. The topic is A old. B simple. C narrow. D broad. Closing Remark! "
        second = "Second Main Topic In recent years, researchers have changed their explanation of the issue. " * 20
        second_questions = "Questions 2-2 Choose the correct letter. 2. The writer feels A critical. B neutral. C hopeful. D unclear."
        document = make_document(
            [
                ProcessedPage(
                    1,
                    "native_pdf",
                    0.95,
                    [
                        make_element("p1-e1", 1, first),
                        make_element("p1-e2", 1, questions + second),
                        make_element("p1-e3", 1, second_questions),
                    ],
                )
            ]
        )

        structured = IELTSStructureParser(self.config).parse(document)

        self.assertEqual(structured.passages[1].title, "Second Main Topic")
        self.assertNotEqual(structured.passages[1].title, "Closing Remark!")


if __name__ == "__main__":
    unittest.main()
