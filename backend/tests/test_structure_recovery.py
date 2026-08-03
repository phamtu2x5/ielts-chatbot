import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.document_pipeline.config import DocumentPipelineConfig
from app.document_pipeline.ielts import IELTSStructureParser, StructuredChunker
from app.document_pipeline.models import DocumentElement, ProcessedDocument, ProcessedPage


class StructureRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DocumentPipelineConfig()
        self.parser = IELTSStructureParser(self.config)
        self.chunker = StructuredChunker(self.config)

    def _document(self, *page_texts: str) -> ProcessedDocument:
        pages = []
        for page_number, text in enumerate(page_texts, 1):
            element = DocumentElement(
                element_id=f"p{page_number}-e1",
                page=page_number,
                type="paragraph",
                raw_text=text,
                normalized_text=text,
                source="native_pdf",
                confidence=1.0,
            )
            pages.append(
                ProcessedPage(
                    page_number=page_number,
                    processing_route="native_pdf",
                    quality_score=1.0,
                    elements=[element],
                )
            )
        return ProcessedDocument(
            document_id="doc-1",
            filename="reading.pdf",
            mime_type="application/pdf",
            parser_version=self.config.parser_version,
            metadata={},
            pages=pages,
        )

    def test_preserves_title_options_and_embedded_writing_task(self) -> None:
        document = self._document(
            """Reading Passage 1
Creative families develop talent over time.
Question 40
40. From the list below choose the most suitable title for the whole of Reading Passage 1.
A Geniuses in their time
B Education for the gifted
C Revising the definition of intelligence
D Nurturing talent within the family""",
            """Task 1
The chart shows changes in household appliance ownership.
Summarise the information by selecting and reporting the main features.""",
        )

        structured = self.parser.parse(document)
        chunks = self.chunker.chunk(document, structured)
        question = next(
            chunk
            for chunk in chunks
            if chunk.metadata.get("unit_type") == "question"
            and chunk.metadata.get("question_range") == [40, 40]
        )
        writing = next(
            chunk for chunk in chunks if chunk.metadata.get("unit_type") == "writing_task"
        )

        self.assertEqual(question.metadata["question_type"], "multiple_choice")
        self.assertIn("A Geniuses in their time", question.text)
        self.assertIn("D Nurturing talent within the family", question.text)
        self.assertEqual(writing.metadata["writing_task_number"], 1)
        self.assertIn("household appliance ownership", writing.text)

    def test_skips_empty_task_one_and_keeps_following_task_two(self) -> None:
        document = self._document(
            """Reading Passage 1
The passage discusses a social issue.
Question 1
1. What is the main issue?""",
            "Task 1",
            """TASK 2
In many countries, crime rates amongst younger people have been rising.
Discuss the causes and solutions for this problem.""",
        )

        structured = self.parser.parse(document)
        chunks = self.chunker.chunk(document, structured)
        writing = [
            chunk for chunk in chunks if chunk.metadata.get("unit_type") == "writing_task"
        ]

        self.assertEqual(len(writing), 1)
        self.assertEqual(writing[0].metadata["writing_task_number"], 2)
        self.assertIn("causes and solutions", writing[0].text)


if __name__ == "__main__":
    unittest.main()
