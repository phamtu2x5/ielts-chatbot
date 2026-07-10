import hashlib
from pathlib import Path

from .chunking import SemanticChunker
from .config import DocumentPipelineConfig
from .extractors import DOCXExtractor, ImageExtractor, PDFExtractor, TextExtractor
from .models import DocumentChunk, ProcessedDocument
from .ocr import OCRProcessor
from .routing import FileRouter


class DocumentProcessor:
    def __init__(self, config: DocumentPipelineConfig | None = None) -> None:
        self.config = config or DocumentPipelineConfig()
        self.router = FileRouter(self.config)
        self.ocr = OCRProcessor(self.config)
        self.chunker = SemanticChunker(self.config)
        self.extractors = {
            "text": TextExtractor(self.config),
            "pdf": PDFExtractor(self.config, self.ocr),
            "docx": DOCXExtractor(self.config, self.ocr),
            "image": ImageExtractor(self.config, self.ocr),
        }

    def process_file(
        self,
        file_path: Path,
        filename: str,
        content_type: str | None = None,
    ) -> tuple[ProcessedDocument, list[DocumentChunk]]:
        document_id = self._sha256(file_path)
        route = self.router.route(file_path, filename, content_type)
        mime_type = content_type or self._mime_for_route(route)
        document = self.extractors[route].extract(file_path, filename, mime_type, document_id)
        chunks = self.chunker.chunk(document)
        return document, chunks

    def warmup_ocr(self) -> dict:
        if not self.config.warmup_ocr:
            return {"skipped": True}
        return self.ocr.warmup(include_medium=self.config.warmup_ocr_medium)

    def _sha256(self, file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _mime_for_route(self, route: str) -> str:
        return {
            "text": "text/plain",
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "image": "image/*",
        }[route]
