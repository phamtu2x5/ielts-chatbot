from pathlib import Path

from PIL import Image

from ..config import DocumentPipelineConfig
from ..models import DocumentElement, ProcessedDocument, ProcessedPage
from ..normalization import normalize_text
from ..ocr import OCRProcessor


class ImageExtractor:
    def __init__(self, config: DocumentPipelineConfig, ocr: OCRProcessor) -> None:
        self.config = config
        self.ocr = ocr

    def extract(self, file_path: Path, filename: str, mime_type: str, document_id: str) -> ProcessedDocument:
        with Image.open(file_path) as image:
            image = image.convert("RGB")
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

        return ProcessedDocument(
            document_id=document_id,
            filename=filename,
            mime_type=mime_type,
            parser_version=self.config.parser_version,
            metadata={"page_count": 1, "languages": [], "ocr_engine": ocr_result.engine},
            pages=[
                ProcessedPage(
                    page_number=1,
                    processing_route="image_ocr",
                    quality_score=ocr_result.confidence,
                    elements=elements,
                )
            ],
        )
