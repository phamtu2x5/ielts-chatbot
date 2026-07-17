import unittest

from app.document_pipeline.config import DocumentPipelineConfig
from app.document_pipeline.ielts import IELTSStructureParser, StructuredChunker
from app.document_pipeline.models import DocumentElement, ProcessedDocument, ProcessedPage


def element(
    element_id: str,
    page: int,
    text: str,
    bbox: list[float],
    source: str = "native_pdf",
    element_type: str = "paragraph",
) -> DocumentElement:
    return DocumentElement(
        element_id=element_id,
        page=page,
        type=element_type,
        raw_text=text,
        normalized_text=text,
        source=source,
        confidence=0.95,
        bbox=bbox,
    )


class WritingCollectionParserTests(unittest.TestCase):
    def test_sections_preserve_task_boundaries_and_ignore_ocr_heading_duplicates(self) -> None:
        pages = [
            ProcessedPage(
                page_number=1,
                processing_route="native_pdf",
                quality_score=1.0,
                elements=[
                    element("p1-e1", 1, "IELTS Task 1 Essay: Forest Data", [10, 10, 300, 30]),
                    element(
                        "p1-e2",
                        1,
                        "The pie charts show changes over time. Summarise the main features.",
                        [10, 40, 500, 65],
                    ),
                    element("p1-e3", 1, "The pie charts compare two datasets. Overall, both changed.", [10, 300, 500, 340]),
                ],
            ),
            ProcessedPage(
                page_number=2,
                processing_route="native_pdf_plus_ocr_reconciled",
                quality_score=1.0,
                elements=[
                    element("p2-e1", 2, "The first sample answer continues on this page.", [10, 10, 500, 30]),
                    element("p2-e2", 2, "IELTS Task 1 Essay: Sales", [10, 100, 300, 120]),
                    element("p2-e3", 2, "The bar chart compares sales. Overall, one series led.", [10, 400, 500, 450]),
                    element(
                        "p2-e4",
                        2,
                        "IELTS Task 1 Essay: Sales The bar chart compares sales.",
                        [10, 100, 500, 450],
                        source="pdf_page_ocr",
                        element_type="ocr_supplement",
                    ),
                ],
            ),
            ProcessedPage(
                page_number=3,
                processing_route="native_pdf",
                quality_score=1.0,
                elements=[
                    element("p3-e1", 3, "The second sample answer continues.", [10, 10, 500, 30]),
                    element("p3-e2", 3, "The line chart depicts crime rates over time.", [10, 400, 500, 430]),
                    element("p3-e3", 3, "Overall, the three series followed different trends.", [10, 450, 500, 480]),
                ],
                metadata={"requires_layout": True},
            ),
        ]
        document = ProcessedDocument(
            document_id="doc",
            filename="writing.pdf",
            mime_type="application/pdf",
            parser_version="test",
            metadata={},
            pages=pages,
        )
        config = DocumentPipelineConfig()

        structured = IELTSStructureParser(config).parse(document)
        chunks = StructuredChunker(config).chunk(document, structured)
        tasks = [section for section in structured.sections if section.type == "writing_task_1"]
        answers = [section for section in structured.sections if section.type == "sample_answer"]

        self.assertEqual(len(tasks), 3)
        self.assertEqual(len(answers), 3)
        self.assertEqual([task.visual_type for task in tasks], ["pie_chart", "bar_chart", "line_chart"])
        self.assertEqual([task.title for task in tasks], ["IELTS Task 1 Essay: Forest Data", "IELTS Task 1 Essay: Sales", None])
        self.assertIn("continues on this page", answers[0].text)
        self.assertIn("second sample answer continues", answers[1].text)
        self.assertNotIn("p2-e4", {element_id for section in structured.sections for element_id in section.element_ids})
        self.assertEqual({chunk.metadata["task_index"] for chunk in chunks}, {1, 2, 3})
        self.assertTrue(all(chunk.metadata["unit_type"] in {"writing_task", "sample_answer"} for chunk in chunks))

    def test_reading_like_prompts_do_not_create_writing_collection_without_task_anchor(self) -> None:
        document = ProcessedDocument(
            document_id="doc",
            filename="reading.pdf",
            mime_type="application/pdf",
            parser_version="test",
            metadata={},
            pages=[
                ProcessedPage(
                    page_number=1,
                    processing_route="native_pdf",
                    quality_score=1.0,
                    elements=[
                        element("p1-e1", 1, "The table shows survey results.", [10, 10, 400, 30]),
                        element("p1-e2", 1, "The map describes an urban area.", [10, 100, 400, 120]),
                    ],
                )
            ],
        )

        structured = IELTSStructureParser(DocumentPipelineConfig()).parse(document)

        self.assertEqual(structured.sections, [])
        self.assertNotEqual(document.metadata.get("document_type"), "ielts_writing_collection")


if __name__ == "__main__":
    unittest.main()
