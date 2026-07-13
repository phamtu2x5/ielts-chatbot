from pathlib import Path

from PIL import Image

from ..config import DocumentPipelineConfig
from ..models import DocumentElement, ProcessedDocument, ProcessedPage
from ..normalization import normalize_text
from ..layout import DocLayoutDetector
from ..ocr import OCRProcessor
from ..visual import WritingTaskTableParser


class ImageExtractor:
    def __init__(self, config: DocumentPipelineConfig, ocr: OCRProcessor, layout: DocLayoutDetector) -> None:
        self.config = config
        self.ocr = ocr
        self.layout = layout
        self.visual_parser = WritingTaskTableParser()

    def extract(self, file_path: Path, filename: str, mime_type: str, document_id: str) -> ProcessedDocument:
        with Image.open(file_path) as image:
            image = image.convert("RGB")
            layout_result = self.layout.detect(image)
            ocr_result = self.ocr.image_to_text(image)

        text = normalize_text(ocr_result.text)
        elements = []
        if text:
            elements.append(
                DocumentElement(
                    element_id="p1-e1",
                    page=1,
                    type="paragraph",
                    raw_text=ocr_result.text,
                    normalized_text=text,
                    source="image_ocr",
                    confidence=ocr_result.confidence,
                    metadata=ocr_result.metadata,
                )
            )
        parsed_visual = self.visual_parser.parse(text)
        metadata = {
            "page_count": 1,
            "languages": [],
            "ocr_engine": ocr_result.engine,
            "ocr_quality": ocr_result.confidence,
            "ocr_metadata": ocr_result.metadata,
            "layout_engine": layout_result.engine,
            "layout_regions": layout_result.region_dicts(),
            "layout_metadata": layout_result.metadata,
        }
        if parsed_visual:
            metadata.update(
                {
                    "document_type": parsed_visual.document_type,
                    "task_type": parsed_visual.task_type,
                    "visual_structure": {
                        "prompt": parsed_visual.prompt,
                        "visual_elements": [parsed_visual.table],
                    },
                }
            )
            prompt_text = parsed_visual.prompt_text()
            table_text = parsed_visual.table_markdown()
            if prompt_text:
                elements.append(
                    DocumentElement(
                        element_id="p1-e2",
                        page=1,
                        type="writing_prompt",
                        raw_text=prompt_text,
                        normalized_text=prompt_text,
                        source="image_ocr_structured",
                        confidence=ocr_result.confidence,
                        metadata={
                            "document_type": parsed_visual.document_type,
                            "task_type": parsed_visual.task_type,
                            "prompt": parsed_visual.prompt,
                        },
                    )
                )
            if table_text:
                elements.append(
                    DocumentElement(
                        element_id="p1-e3",
                        page=1,
                        type="table",
                        raw_text=table_text,
                        normalized_text=table_text,
                        source="image_ocr_structured",
                        confidence=ocr_result.confidence,
                        metadata={
                            "document_type": parsed_visual.document_type,
                            "task_type": parsed_visual.task_type,
                            "table": parsed_visual.table,
                        },
                    )
                )

        return ProcessedDocument(
            document_id=document_id,
            filename=filename,
            mime_type=mime_type,
            parser_version=self.config.parser_version,
            metadata=metadata,
            pages=[
                ProcessedPage(
                    page_number=1,
                    processing_route="image_ocr",
                    quality_score=ocr_result.confidence,
                    elements=elements,
                )
            ],
        )
