import hashlib
import logging
import time
from pathlib import Path

from .chunking import SemanticChunker
from .config import DocumentPipelineConfig
from .extractors import DOCXExtractor, ImageExtractor, PDFExtractor, TextExtractor
from .ielts import IELTSStructureParser, StructuredChunker
from .layout import DocLayoutDetector
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
        self.layout = DocLayoutDetector(self.config)
        self.reconciler = NativeOCRReconciler(self.config)
        self.structure_parser = IELTSStructureParser(self.config)
        self.structured_chunker = StructuredChunker(self.config)
        self.chunker = SemanticChunker(self.config)
        self.extractors = {
            "text": TextExtractor(self.config),
            "pdf": PDFExtractor(self.config, self.ocr, self.layout),
            "docx": DOCXExtractor(self.config, self.ocr),
            "image": ImageExtractor(self.config, self.ocr, self.layout),
        }

    def process_file(
        self,
        file_path: Path,
        filename: str,
        content_type: str | None = None,
    ) -> tuple[ProcessedDocument, list[DocumentChunk]]:
        started = time.perf_counter()
        document_id = self._sha256(file_path)
        timing = {"document_id_hash_seconds": self._elapsed(started)}
        route_started = time.perf_counter()
        route = self.router.route(file_path, filename, content_type)
        timing["route_seconds"] = self._elapsed(route_started)
        mime_type = content_type or self._mime_for_route(route)
        extract_started = time.perf_counter()
        document = self.extractors[route].extract(file_path, filename, mime_type, document_id)
        timing["extract_seconds"] = self._elapsed(extract_started)
        reconcile_started = time.perf_counter()
        document = self.reconciler.reconcile(document)
        timing["reconcile_seconds"] = self._elapsed(reconcile_started)
        if self.config.enable_ielts_structure_parser:
            structure_started = time.perf_counter()
            structured_document = self.structure_parser.parse(document)
            timing["structure_parse_seconds"] = self._elapsed(structure_started)
            chunk_started = time.perf_counter()
            chunks = self.structured_chunker.chunk(document, structured_document)
            timing["chunk_seconds"] = self._elapsed(chunk_started)
            if not chunks:
                fallback_chunk_started = time.perf_counter()
                chunks = self.chunker.chunk(document)
                timing["fallback_chunk_seconds"] = self._elapsed(fallback_chunk_started)
        else:
            chunk_started = time.perf_counter()
            chunks = self.chunker.chunk(document)
            timing["chunk_seconds"] = self._elapsed(chunk_started)
        timing["chunks"] = len(chunks)
        timing["total_seconds"] = self._elapsed(started)
        document.metadata.setdefault("timing", {})
        document.metadata["timing"].update(
            {
                "process_file": {
                    "route": route,
                    **timing,
                },
                "chunking": {
                    "structure_parse_seconds": timing.get("structure_parse_seconds", 0.0),
                    "chunk_seconds": timing.get("chunk_seconds", 0.0),
                    "fallback_chunk_seconds": timing.get("fallback_chunk_seconds", 0.0),
                    "chunks": len(chunks),
                },
            }
        )
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
        return self.ocr.warmup()

    def warmup_layout(self) -> dict:
        return self.layout.warmup()

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

    def _elapsed(self, started: float) -> float:
        return round(time.perf_counter() - started, 3)
