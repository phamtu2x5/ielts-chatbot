from pathlib import Path

from ..config import DocumentPipelineConfig
from ..models import DocumentElement, ProcessedDocument, ProcessedPage
from ..normalization import normalize_text


class TextExtractor:
    def __init__(self, config: DocumentPipelineConfig) -> None:
        self.config = config

    def extract(self, file_path: Path, filename: str, mime_type: str, document_id: str) -> ProcessedDocument:
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        normalized = normalize_text(raw)
        elements = []
        for index, paragraph in enumerate([part.strip() for part in normalized.split("\n\n") if part.strip()], 1):
            elements.append(
                DocumentElement(
                    element_id=f"p1-e{index}",
                    page=1,
                    type="paragraph",
                    raw_text=paragraph,
                    normalized_text=paragraph,
                    source="native_text",
                    confidence=1.0,
                )
            )

        return ProcessedDocument(
            document_id=document_id,
            filename=filename,
            mime_type=mime_type,
            parser_version=self.config.parser_version,
            metadata={"page_count": 1, "languages": []},
            pages=[ProcessedPage(page_number=1, processing_route="native_text", quality_score=1.0, elements=elements)],
        )
