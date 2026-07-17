import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from PIL import Image, ImageDraw


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.document_pipeline.config import DocumentPipelineConfig
from app.document_pipeline.connectors import RasterConnectorDetector
from app.document_pipeline.chunking import SemanticChunker
from app.document_pipeline.extractors.pdf import PDFExtractor
from app.document_pipeline.extractors.text import TextExtractor
from app.document_pipeline.ielts import IELTSStructureParser, StructuredChunker
from app.document_pipeline.layout import DocLayoutDetector
from app.document_pipeline.models import DocumentElement, ProcessedDocument, ProcessedPage
from app.document_pipeline.ocr import OCRProcessor, OCRResult
from app.document_pipeline.reconciliation import NativeOCRReconciler
from app.document_pipeline.routing import FileRouter
from app.document_pipeline.visual import IELTSQuestionVisualParser, WritingTaskTableParser


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

    def test_all_rapidocr_components_use_torch(self) -> None:
        typings = ModuleType("rapidocr.utils.typings")

        class FakeEnum:
            TORCH = "torch"

            def __new__(cls, value):
                return value

        typings.EngineType = FakeEnum
        typings.LangDet = FakeEnum
        typings.LangRec = FakeEnum
        typings.ModelType = FakeEnum
        typings.OCRVersion = FakeEnum

        processor = OCRProcessor(DocumentPipelineConfig())
        with patch.dict(sys.modules, {"rapidocr.utils.typings": typings}):
            params = processor._rapidocr_params()

        self.assertEqual(params["Det.engine_type"], "torch")
        self.assertEqual(params["Cls.engine_type"], "torch")
        self.assertEqual(params["Rec.engine_type"], "torch")
        self.assertFalse(params["Global.use_cls"])

    def test_rapidocr_call_disables_line_classification(self) -> None:
        class RapidOCRLikeResult:
            txts = ("IELTS",)
            scores = (0.9,)
            boxes = ()

        calls = []

        def fake_ocr(path: str, **kwargs):
            calls.append((path, kwargs))
            return RapidOCRLikeResult()

        processor = OCRProcessor(DocumentPipelineConfig())
        with (
            patch.object(processor, "_rapidocr_is_available", return_value=True),
            patch.object(processor, "_get_rapidocr", return_value=fake_ocr),
        ):
            result = processor._image_to_text_with_rapidocr(Image.new("RGB", (20, 20), "white"))

        self.assertEqual(result.text, "IELTS")
        self.assertEqual(calls[0][1], {"use_cls": False})

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


class PDFVisualOCRRetryTests(unittest.TestCase):
    def test_visual_region_retry_skips_complete_question_range(self) -> None:
        class UnexpectedOCR:
            def image_to_text(self, image: Image.Image) -> OCRResult:
                raise AssertionError("Complete visual question ranges must not trigger another OCR pass.")

        extractor = PDFExtractor(DocumentPipelineConfig(), UnexpectedOCR(), None)
        original = OCRResult(
            "Questions 5-7\nComplete the table.\n(5) first\n(6) second\n(7) third",
            0.97,
            "rapidocr_test",
            {"lines": []},
        )

        result = extractor._retry_visual_ocr(
            Image.new("RGB", (120, 80), "white"),
            [{"type": "table", "bbox": [0, 0, 120, 80]}],
            original,
        )

        self.assertIs(result, original)

    def test_visual_region_retry_replaces_overlapping_misread_question_number(self) -> None:
        retry_result = OCRResult(
            "can be (7)",
            0.98,
            "rapidocr_test",
            {
                "lines": [
                    {
                        "text": "can be (7)",
                        "confidence": 0.98,
                        "bbox": [[40, 40], [120, 40], [120, 60], [40, 60]],
                    }
                ]
            },
        )

        class RetryOCR:
            def image_to_text(self, image: Image.Image) -> OCRResult:
                return retry_result

        extractor = PDFExtractor(DocumentPipelineConfig(), RetryOCR(), None)
        original = OCRResult(
            "Questions 5-10\nComplete the table.\ncan be (Z)\n(5) (6) (8) (9) (10)",
            0.96,
            "rapidocr_test",
            {
                "lines": [
                    {"text": "Questions 5-10", "confidence": 0.99, "bbox": [0, 0, 80, 10]},
                    {"text": "Complete the table.", "confidence": 0.99, "bbox": [0, 10, 100, 20]},
                    {"text": "can be (Z)", "confidence": 0.96, "bbox": [20, 20, 60, 30]},
                    {"text": "(5) (6) (8) (9) (10)", "confidence": 0.99, "bbox": [0, 35, 100, 45]},
                ]
            },
        )

        result = extractor._retry_visual_ocr(
            Image.new("RGB", (120, 80), "white"),
            [{"type": "table", "bbox": [0, 0, 120, 80]}],
            original,
        )

        self.assertIn("can be (7)", result.text)
        self.assertNotIn("can be (Z)", result.text)
        self.assertEqual(result.metadata["visual_ocr_retry_recovered"], [7])
        accepted = result.metadata["visual_ocr_retries"][0]["accepted_lines"][0]
        self.assertEqual(
            accepted["bbox"],
            [[20.0, 20.0], [60.0, 20.0], [60.0, 30.0], [20.0, 30.0]],
        )


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


class PDFSpatialVisualParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = IELTSQuestionVisualParser()

    def _line(self, text: str, x: int, y: int, width: int = 150) -> dict:
        return {
            "text": text,
            "confidence": 0.96,
            "bbox": [[x, y], [x + width, y], [x + width, y + 20], [x, y + 20]],
        }

    def test_pdf_table_uses_layout_region_and_ocr_line_boxes(self) -> None:
        lines = [
            self._line("Classification based on", 20, 20),
            self._line("Associated fact", 220, 20),
            self._line("Related example", 420, 20),
            self._line("Colour", 20, 80),
            self._line("Red wines use (5) in fermentation", 220, 80),
            self._line("(6)", 420, 80),
            self._line("grape species", 20, 140),
            self._line("can be (7) or blended", 220, 140),
            self._line("Cote Rotie wines", 420, 140),
        ]
        visual = self.parser.parse(
            "Questions 5-10 Complete the table.",
            5,
            10,
            "table_completion",
            [2],
            ["p2-e4"],
            spatial_pages=[
                {
                    "page": 2,
                    "ocr_lines": lines,
                    "layout_regions": [
                        {"type": "table", "confidence": 0.93, "bbox": [0, 0, 600, 220]},
                    ],
                }
            ],
        )

        self.assertIsNotNone(visual)
        self.assertEqual(visual.visual_type, "table")
        self.assertEqual(
            visual.payload["columns"],
            ["Classification based on", "Associated fact", "Related example"],
        )
        self.assertEqual(len(visual.payload["rows"]), 2)
        self.assertEqual(visual.payload["source"], "doclayout_yolo+rapidocr_boxes")
        self.assertEqual(visual.payload["bbox"], [0.0, 0.0, 600.0, 220.0])

    def test_pdf_table_uses_header_columns_when_a_cell_is_split_into_ocr_fragments(self) -> None:
        lines = [
            self._line("Category", 20, 20),
            self._line("Fact", 220, 20),
            self._line("Example", 480, 20),
            self._line("Colour", 20, 80),
            self._line("uses (5)", 190, 80),
            self._line("in fermentation", 330, 80),
            self._line("(6)", 480, 80),
            self._line("Species", 20, 140),
            self._line("can be (7)", 190, 140),
            self._line("or blended", 330, 140),
            self._line("Example wine", 480, 140),
        ]

        visual = self.parser.parse(
            "Questions 5-7 Complete the table.",
            5,
            7,
            "table_completion",
            [2],
            ["p2-e4"],
            spatial_pages=[
                {
                    "page": 2,
                    "ocr_lines": lines,
                    "layout_regions": [{"type": "table", "confidence": 0.93, "bbox": [0, 0, 650, 220]}],
                }
            ],
        )

        self.assertIsNotNone(visual)
        self.assertEqual(visual.payload["columns"], ["Category", "Fact", "Example"])
        self.assertEqual(
            visual.payload["rows"],
            [
                ["Colour", "uses (5) in fermentation", "(6)"],
                ["Species", "can be (7) or blended", "Example wine"],
            ],
        )
        self.assertEqual(visual.payload["quality_status"], "passed")

    def test_pdf_table_reports_missing_question_numbers_without_rewriting_ocr(self) -> None:
        lines = [
            self._line("Category", 20, 20),
            self._line("Answer", 300, 20),
            self._line("Colour", 20, 80),
            self._line("(5)", 300, 80),
            self._line("Species", 20, 140),
            self._line("(Z)", 300, 140),
        ]

        visual = self.parser.parse(
            "Questions 5-6 Complete the table.",
            5,
            6,
            "table_completion",
            [2],
            ["p2-e4"],
            spatial_pages=[
                {
                    "page": 2,
                    "ocr_lines": lines,
                    "layout_regions": [{"type": "table", "confidence": 0.93, "bbox": [0, 0, 600, 220]}],
                }
            ],
        )

        self.assertIsNotNone(visual)
        self.assertEqual(visual.payload["missing_question_numbers"], [6])
        self.assertEqual(visual.payload["quality_status"], "degraded")
        self.assertIn("(Z)", visual.payload["rows"][1])

    def test_spatial_table_can_override_non_table_instruction_type(self) -> None:
        lines = [
            self._line("Category", 20, 20, 250),
            self._line("Example", 350, 20),
            self._line("human attributes", 20, 80, 250),
            self._line("Physical strength and (36)", 350, 80),
            self._line("medical conditions", 20, 140, 250),
            self._line("(37) and fungal attack", 350, 140),
        ]
        visual = self.parser.parse(
            "Questions 36-40 Give TWO examples.",
            36,
            40,
            "short_answer_examples",
            [6],
            ["p6-e6"],
            spatial_pages=[
                {
                    "page": 6,
                    "ocr_lines": lines,
                    "layout_regions": [
                        {"type": "table", "confidence": 0.91, "bbox": [0, 0, 600, 220]},
                    ],
                }
            ],
        )

        self.assertIsNotNone(visual)
        self.assertEqual(visual.visual_type, "table")
        self.assertEqual(visual.payload["question_range"], [36, 40])

    def test_pdf_flowchart_keeps_spatial_nodes_without_inventing_edges(self) -> None:
        lines = [
            self._line("manager having", 20, 20),
            self._line("obtained (18)", 20, 50),
            self._line("people greatly varied (19)", 20, 140),
            self._line("playing with language", 250, 50),
            self._line("(20) set of words", 470, 50),
            self._line("obeyed only at (21)", 700, 80),
            self._line("can cause lack of (22)", 700, 110),
            self._line("want (23) payback", 300, 170),
        ]
        visual = self.parser.parse(
            "Questions 18-23 Complete the flow chart.",
            18,
            23,
            "flowchart_completion",
            [4],
            ["p4-e5"],
            spatial_pages=[
                {
                    "page": 4,
                    "ocr_lines": lines,
                    "layout_regions": [
                        {"type": "figure", "confidence": 0.92, "bbox": [0, 0, 900, 240]},
                    ],
                }
            ],
        )

        self.assertIsNotNone(visual)
        self.assertEqual(visual.visual_type, "flowchart")
        self.assertTrue(any(node["question_numbers"] for node in visual.payload["nodes"]))
        self.assertTrue(all(node["bbox"] for node in visual.payload["nodes"]))
        self.assertEqual(visual.payload["edges"], [])
        self.assertEqual(visual.payload["edge_detection"], "not_available")

    def test_pdf_flowchart_maps_raster_arrowheads_to_nodes(self) -> None:
        lines = [
            self._line("step (18)", 20, 50),
            self._line("step (19)", 320, 50),
            self._line("step (20)", 620, 50),
        ]
        connectors = [
            {
                "id": "connector-1",
                "bbox": [170, 45, 320, 75],
                "endpoints": [[170, 60], [320, 60]],
                "arrowhead_point": [310, 60],
                "direction_confidence": 0.9,
            },
            {
                "id": "connector-2",
                "bbox": [470, 45, 620, 75],
                "endpoints": [[470, 60], [620, 60]],
                "arrowhead_point": [610, 60],
                "direction_confidence": 0.88,
            },
        ]
        visual = self.parser.parse(
            "Questions 18-20 Complete the flow chart.",
            18,
            20,
            "flowchart_completion",
            [4],
            ["p4-e5"],
            spatial_pages=[
                {
                    "page": 4,
                    "ocr_lines": lines,
                    "layout_regions": [{"type": "figure", "confidence": 0.92, "bbox": [0, 0, 800, 180]}],
                    "connector_regions": [{"bbox": [0, 0, 800, 180], "connectors": connectors}],
                }
            ],
        )

        self.assertIsNotNone(visual)
        self.assertEqual(
            [(edge["from"], edge["to"]) for edge in visual.payload["edges"]],
            [("node-1", "node-2"), ("node-2", "node-3")],
        )
        self.assertEqual(visual.payload["edge_detection"], "raster_arrowheads")

    def test_duplicate_nearest_pair_uses_connector_axis_for_branch_target(self) -> None:
        nodes = [
            {"id": "node-1", "bbox": [120, 20, 220, 100]},
            {"id": "node-2", "bbox": [20, 140, 100, 200]},
            {"id": "node-3", "bbox": [320, 130, 400, 190]},
        ]
        connectors = [
            {
                "id": "connector-1",
                "bbox": [90, 90, 150, 170],
                "endpoints": [[100, 160], [145, 100]],
                "arrowhead_point": [145, 105],
                "direction_confidence": 0.9,
            },
            {
                "id": "connector-2",
                "bbox": [90, 150, 230, 185],
                "endpoints": [[100, 170], [230, 170]],
                "arrowhead_point": [220, 170],
                "direction_confidence": 0.8,
            },
        ]

        edges, unresolved = self.parser._connector_graph(nodes, connectors, [0, 0, 800, 240])

        self.assertEqual(
            [(edge["from"], edge["to"]) for edge in edges],
            [("node-2", "node-1"), ("node-2", "node-3")],
        )
        self.assertEqual(edges[1]["evidence"], "raster_arrowhead_directional_remap")
        self.assertEqual(unresolved, [])

    def test_textual_flowchart_preserves_order_without_inventing_edges(self) -> None:
        visual = self.parser.parse(
            """Questions 34-37
Complete the flow chart below.
• For (34) early communities produced paintings.
• Early period: groups used (35) for paintings.
• Mid period: paintings appeared in (36).
• Later period: patterns were painted on (37).""",
            34,
            37,
            "flowchart_completion",
            [6],
            ["p6-e3"],
        )

        self.assertIsNotNone(visual)
        self.assertEqual(len(visual.payload["nodes"]), 4)
        self.assertEqual(len(visual.payload["ordered_items"]), 4)
        self.assertEqual(visual.payload["edges"], [])
        self.assertEqual(visual.payload["edge_detection"], "not_present_in_source")
        self.assertEqual(visual.payload["quality_status"], "passed")

    def test_diagram_labels_keep_bbox_without_inventing_edges(self) -> None:
        lines = [
            self._line("water is atomized into", 250, 30, 220),
            self._line("7.", 250, 60, 40),
            self._line("8.", 20, 150, 40),
            self._line("are formed", 20, 180, 120),
            self._line("6.", 650, 150, 40),
            self._line("air", 720, 150, 60),
        ]
        visual = self.parser.parse(
            "Questions 6-8 Label the diagram below.",
            6,
            8,
            "diagram_labeling",
            [2],
            ["p2-e3"],
            spatial_pages=[
                {
                    "page": 2,
                    "ocr_lines": lines,
                    "layout_regions": [{"type": "figure", "confidence": 0.94, "bbox": [0, 0, 850, 260]}],
                    "connector_regions": [
                        {
                            "bbox": [0, 0, 850, 260],
                            "connectors": [
                                {
                                    "id": "connector-1",
                                    "bbox": [290, 80, 400, 140],
                                    "endpoints": [[290, 80], [400, 140]],
                                    "arrowhead_point": [400, 140],
                                    "direction_confidence": 0.4,
                                }
                            ],
                        }
                    ],
                }
            ],
        )

        self.assertIsNotNone(visual)
        self.assertEqual(visual.visual_type, "diagram")
        self.assertEqual({label["question_number"] for label in visual.payload["labels"]}, {6, 7, 8})
        self.assertEqual(visual.payload["edges"], [])
        self.assertEqual(visual.payload["quality_status"], "passed")
        self.assertEqual(visual.payload["connector_status"], "partial")
        self.assertEqual(
            visual.payload["connector_issues"],
            ["connector_coverage_low", "low_confidence_connectors"],
        )


class RasterConnectorDetectorTests(unittest.TestCase):
    def test_detector_emits_json_safe_geometry_for_diagonal_arrow(self) -> None:
        image = Image.new("RGB", (400, 200), "white")
        draw = ImageDraw.Draw(image)
        draw.line((90, 150, 300, 50), fill="black", width=8)
        draw.polygon([(300, 50), (276, 53), (288, 73)], fill="black")
        detector = RasterConnectorDetector(DocumentPipelineConfig())

        result = detector.detect(
            image,
            [{"type": "figure", "confidence": 0.9, "bbox": [0, 0, 400, 200]}],
            [],
        )

        self.assertGreaterEqual(result.metadata["connectors_found"], 1)
        connector = result.regions[0]["connectors"][0]
        self.assertEqual(len(connector["endpoints"]), 2)
        self.assertGreaterEqual(connector["direction_confidence"], 0.55)
        self.assertIn("head_strength", connector["direction_evidence"])
        self.assertIn("endpoint_separation", connector["direction_evidence"])
        json.dumps(result.regions)


class WritingVisualParserAdditionalTests(unittest.TestCase):

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

    def test_label_diagram_group_emits_structured_diagram_chunk(self) -> None:
        text = (
            "Snow production equipment\n"
            "This passage explains a mechanical process in detail.\n"
            "Questions 6-8 Label the diagram below using NO MORE THAN TWO WORDS.\n"
            "6. air 7. droplets 8. crystals"
        )
        ocr_lines = [
            {"text": "6. air", "confidence": 0.96, "bbox": [[20, 80], [120, 80], [120, 100], [20, 100]]},
            {"text": "7. droplets", "confidence": 0.96, "bbox": [[300, 40], [430, 40], [430, 60], [300, 60]]},
            {"text": "8. crystals", "confidence": 0.96, "bbox": [[20, 180], [150, 180], [150, 200], [20, 200]]},
        ]
        document = make_document(
            [
                ProcessedPage(
                    1,
                    "native_pdf_plus_ocr",
                    0.9,
                    [make_element("p1-e1", 1, text)],
                    metadata={
                        "ocr_metadata": {"lines": ocr_lines},
                        "layout_regions": [{"type": "figure", "confidence": 0.93, "bbox": [0, 0, 600, 240]}],
                    },
                )
            ]
        )

        structured = IELTSStructureParser(self.config).parse(document)
        groups = [group for passage in structured.passages for group in passage.question_groups]
        group = next(group for group in groups if group.question_start == 6)
        chunks = StructuredChunker(self.config).chunk(document, structured)

        self.assertEqual(group.question_type, "diagram_labeling")
        self.assertEqual(group.visual_element["type"], "diagram")
        self.assertEqual(len(group.visual_element["labels"]), 3)
        self.assertIn("diagram", [chunk.metadata.get("unit_type") for chunk in chunks])

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

    def test_passage_titles_use_block_boundaries_instead_of_proper_nouns(self) -> None:
        first = (
            "A Broad Environmental Topic\n"
            "The opening section discusses field research in several regions.\n"
            "Researchers from North Valley University worked near Costa Rica.\n"
            "They described the results and their wider significance."
        )
        first_questions = (
            "Questions 1-2\n"
            "Answer the questions.\n"
            "1. What was studied?\n"
            "2. Where was the work conducted?\n\n"
            "A Different Social Topic\n"
            "The next passage considers how organisations make decisions.\n"
            "Professor Jane Example is quoted in the discussion."
        )
        second_questions = (
            "Questions 3-4\n"
            "Answer the questions.\n"
            "3. What do organisations consider?\n"
            "4. Who is quoted?"
        )
        document = make_document(
            [
                ProcessedPage(
                    1,
                    "native_pdf",
                    0.95,
                    [
                        make_element("p1-e1", 1, first),
                        make_element("p1-e2", 1, first_questions),
                        make_element("p1-e3", 1, second_questions),
                    ],
                )
            ]
        )

        structured = IELTSStructureParser(self.config).parse(document)

        self.assertEqual(
            [passage.title for passage in structured.passages],
            ["A Broad Environmental Topic", "A Different Social Topic"],
        )
        self.assertNotIn("Costa Rica", [passage.title for passage in structured.passages])
        self.assertNotIn("Professor Jane Example", [passage.title for passage in structured.passages])

    def test_single_question_header_is_kept_as_its_own_group(self) -> None:
        text = (
            "A Passage Without a Fixed Question Count\n"
            "This passage contains enough background for several tasks. " * 20
            + "\nQuestions 35-39\n"
            "Do the following statements agree with the writer?\n"
            "35. First statement. 36. Second statement. 37. Third statement. "
            "38. Fourth statement. 39. Fifth statement.\n"
            "Question 40\n"
            "40. Choose the most suitable title. A First B Second C Third D Fourth"
        )
        document = make_document(
            [ProcessedPage(1, "native_pdf", 0.95, [make_element("p1-e1", 1, text)])]
        )

        structured = IELTSStructureParser(self.config).parse(document)

        self.assertEqual(
            [
                (group.question_start, group.question_end)
                for group in structured.passages[0].question_groups
            ],
            [(35, 39), (40, 40)],
        )
        self.assertEqual(structured.diagnostics["questions_found"], 6)

    def test_passage_marker_without_explicit_title_does_not_invent_one(self) -> None:
        first = "First Passage\n" + "This is the first passage body. " * 30
        transition = (
            "Questions 1-1\n1. First question?\n\n"
            "Reading Passage 2\n"
            "What do we mean by exceptional ability? "
            + "The discussion compares several definitions of ability. " * 20
        )
        second_questions = "Questions 2-2\n2. Second question?"
        document = make_document(
            [
                ProcessedPage(
                    1,
                    "native_pdf",
                    0.95,
                    [
                        make_element("p1-e1", 1, first),
                        make_element("p1-e2", 1, transition),
                        make_element("p1-e3", 1, second_questions),
                    ],
                )
            ]
        )

        structured = IELTSStructureParser(self.config).parse(document)

        self.assertEqual(len(structured.passages), 2)
        self.assertIsNone(structured.passages[1].title)
        self.assertIn("exceptional ability", structured.passages[1].text)
        self.assertEqual(structured.diagnostics["suspicious_boundaries"], [])

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

    def test_structure_parser_passes_pdf_spatial_context_to_visual_parser(self) -> None:
        def line(text: str, x: int, y: int) -> dict:
            return {
                "text": text,
                "confidence": 0.96,
                "bbox": [[x, y], [x + 140, y], [x + 140, y + 20], [x, y + 20]],
            }

        text = (
            "A Practice Passage\n"
            "This passage explains several classifications and examples.\n"
            "Questions 5-10 Complete the table. Choose NO MORE THAN TWO WORDS from the passage.\n"
        )
        page = ProcessedPage(
            1,
            "native_pdf_plus_ocr_reconciled",
            0.95,
            [make_element("p1-e1", 1, text)],
            metadata={
                "layout_regions": [{"type": "table", "confidence": 0.93, "bbox": [0, 0, 600, 220]}],
                "ocr_metadata": {
                    "lines": [
                        line("Category", 20, 20),
                        line("Fact", 220, 20),
                        line("Example", 420, 20),
                        line("Colour", 20, 80),
                        line("uses (5)", 220, 80),
                        line("(6)", 420, 80),
                        line("Species", 20, 140),
                        line("can be (7)", 220, 140),
                        line("Example wine", 420, 140),
                    ]
                },
            },
        )
        document = make_document([page])

        structured = IELTSStructureParser(self.config).parse(document)
        visual = structured.passages[0].question_groups[0].visual_element

        self.assertIsNotNone(visual)
        self.assertEqual(visual["source"], "doclayout_yolo+rapidocr_boxes")
        self.assertEqual(visual["columns"], ["Category", "Fact", "Example"])
        self.assertEqual(len(visual["rows"]), 2)

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
