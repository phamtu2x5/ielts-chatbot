from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PageQuality:
    native_text_is_usable: bool
    score: float
    reasons: List[str]
    requires_ocr: bool
    requires_layout: bool
    requires_table_analysis: bool
    recommended_dpi: int


@dataclass
class DocumentElement:
    element_id: str
    page: int
    type: str
    raw_text: str
    normalized_text: str
    source: str
    confidence: float
    bbox: Optional[List[float]] = None
    parent_heading: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_id": self.element_id,
            "page": self.page,
            "type": self.type,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "bbox": self.bbox,
            "source": self.source,
            "confidence": self.confidence,
            "parent_heading": self.parent_heading,
            "metadata": self.metadata,
        }


@dataclass
class ProcessedPage:
    page_number: int
    processing_route: str
    quality_score: float
    elements: List[DocumentElement] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "processing_route": self.processing_route,
            "quality_score": self.quality_score,
            "elements": [element.to_dict() for element in self.elements],
        }


@dataclass
class ProcessedDocument:
    document_id: str
    filename: str
    mime_type: str
    parser_version: str
    metadata: Dict[str, Any]
    pages: List[ProcessedPage]

    @property
    def elements(self) -> List[DocumentElement]:
        return [element for page in self.pages for element in page.elements]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "parser_version": self.parser_version,
            "metadata": self.metadata,
            "pages": [page.to_dict() for page in self.pages],
        }


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    source_file: str
    pages: List[int]
    element_ids: List[str]
    heading_path: List[str]
    text: str
    token_count: int
    min_confidence: float
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "pages": self.pages,
            "element_ids": self.element_ids,
            "heading_path": self.heading_path,
            "text": self.text,
            "token_count": self.token_count,
            "min_confidence": self.min_confidence,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }
