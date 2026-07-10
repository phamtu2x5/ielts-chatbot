import hashlib
import logging
import time
from pathlib import Path

from .chunking import SemanticChunker
from .config import DocumentPipelineConfig
from .extractors import DOCXExtractor, ImageExtractor, PDFExtractor, TextExtractor
from .ielts import IELTSStructureParser, StructuredChunker
from .models import DocumentChunk, ProcessedDocument
from .ocr import OCRProcessor
from .reconciliation import NativeOCRReconciler
from .routing import FileRouter


logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, config: DocumentPipelineConfig | None = None) -> None:
        self.config = config or DocumentPipelineConfig()
        self.router = FileRouter(self.config)
        self.ocr = OCRProcessor(self.config)
        self.reconciler = NativeOCRReconciler(self.config)
        self.structure_parser = IELTSStructureParser(self.config)
        self.structured_chunker = StructuredChunker(self.config)
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
        started = time.perf_counter()
        document_id = self._sha256(file_path)
        route = self.router.route(file_path, filename, content_type)
        mime_type = content_type or self._mime_for_route(route)
        document = self.extractors[route].extract(file_path, filename, mime_type, document_id)
        document = self.reconciler.reconcile(document)
        if self.config.enable_ielts_structure_parser:
            structured_document = self.structure_parser.parse(document)
            chunks = self.structured_chunker.chunk(document, structured_document)
            if not chunks:
                chunks = self.chunker.chunk(document)
        else:
            chunks = self.chunker.chunk(document)
        logger.info(
            "Document pipeline completed route=%s pages=%d chunks=%d duration_seconds=%.3f",
            route,
            len(document.pages),
            len(chunks),
            time.perf_counter() - started,
        )
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
