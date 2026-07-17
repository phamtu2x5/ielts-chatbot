import re
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from ..config import DocumentPipelineConfig
from ..connectors import ConnectorDetectionResult, RasterConnectorDetector
from ..layout import DocLayoutDetector, LayoutResult
from ..models import DocumentElement, ProcessedDocument, ProcessedPage
from ..normalization import normalize_inline_text, normalize_text
from ..ocr import OCRProcessor
from ..quality import evaluate_native_page_text


ProgressCallback = Callable[[str, dict[str, Any]], None]


class PDFExtractor:
    def __init__(self, config: DocumentPipelineConfig, ocr: OCRProcessor, layout: DocLayoutDetector) -> None:
        self.config = config
        self.ocr = ocr
        self.layout = layout
        self.connectors = RasterConnectorDetector(config)

    def extract(
        self,
        file_path: Path,
        filename: str,
        mime_type: str,
        document_id: str,
        progress: ProgressCallback | None = None,
    ) -> ProcessedDocument:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to process PDF files") from exc

        pages = []
        pdf_started = time.perf_counter()
        with fitz.open(file_path) as document:
            pdf_open_seconds = self._elapsed(pdf_started)
            page_count = len(document)
            self._emit(
                progress,
                "pdf_opened",
                pages=page_count,
                duration_seconds=pdf_open_seconds,
            )
            if page_count > self.config.max_pdf_pages:
                raise ValueError(f"PDF có {page_count} trang, vượt giới hạn {self.config.max_pdf_pages} trang.")

            for page_index, page in enumerate(document, 1):
                self._emit(progress, "page_started", page=page_index)
                page_timing = {"page": page_index}
                native_started = time.perf_counter()
                blocks = self._native_text_blocks(page)
                native_text = "\n".join(block["normalized_text"] for block in blocks)
                image_coverage = self._image_coverage(page)
                page_timing["native_extract_seconds"] = self._elapsed(native_started)
                self._emit(
                    progress,
                    "native_extract_finished",
                    page=page_index,
                    duration_seconds=page_timing["native_extract_seconds"],
                    text_blocks=len(blocks),
                    characters=len(native_text),
                    image_coverage=round(image_coverage, 4),
                )
                quality_started = time.perf_counter()
                quality = evaluate_native_page_text(native_text, len(blocks), image_coverage, self.config)
                page_timing["quality_eval_seconds"] = self._elapsed(quality_started)
                self._emit(
                    progress,
                    "quality_evaluation_finished",
                    page=page_index,
                    duration_seconds=page_timing["quality_eval_seconds"],
                    score=quality.score,
                    native_text_is_usable=quality.native_text_is_usable,
                    requires_layout=quality.requires_layout,
                    requires_table_analysis=quality.requires_table_analysis,
                )
                page_timing["render_seconds"] = 0.0
                page_timing["layout_seconds"] = 0.0
                page_timing["ocr_seconds"] = 0.0
                page_timing["connector_seconds"] = 0.0

                if quality.native_text_is_usable:
                    elements = [
                        DocumentElement(
                            element_id=f"p{page_index}-e{idx}",
                            page=page_index,
                            type="paragraph",
                            raw_text=block["raw_text"],
                            normalized_text=block["normalized_text"],
                            source="native_pdf",
                            confidence=quality.score,
                            bbox=block["bbox"],
                            metadata={"quality_reasons": quality.reasons},
                        )
                        for idx, block in enumerate(blocks, 1)
                        if block["normalized_text"]
                    ]
                    processing_route = "native_pdf"
                    page_quality_score = quality.score
                    ocr_result = None
                    merge_reasons = self._ocr_merge_reasons(native_text)
                    layout_result = self._empty_layout_result()
                    connector_result = self._empty_connector_result()
                    rendered_page = None
                    layout_needed = bool(merge_reasons)
                    if layout_needed:
                        render_started = time.perf_counter()
                        self._emit(progress, "page_render_started", page=page_index)
                        rendered_page = self._render_page(page)
                        page_timing["render_seconds"] = self._elapsed(render_started)
                        self._emit(
                            progress,
                            "page_render_finished",
                            page=page_index,
                            duration_seconds=page_timing["render_seconds"],
                            width=rendered_page.width,
                            height=rendered_page.height,
                        )
                        layout_started = time.perf_counter()
                        self._emit(progress, "layout_started", page=page_index)
                        layout_result = self.layout.detect(rendered_page)
                        page_timing["layout_seconds"] = self._elapsed(layout_started)
                        self._emit(
                            progress,
                            "layout_finished",
                            page=page_index,
                            duration_seconds=page_timing["layout_seconds"],
                            engine=layout_result.engine,
                            regions=len(layout_result.regions),
                        )
                    if merge_reasons and rendered_page is not None:
                        ocr_started = time.perf_counter()
                        self._emit(progress, "ocr_started", page=page_index)
                        ocr_result = self.ocr.image_to_text(rendered_page)
                        page_timing["ocr_seconds"] = self._elapsed(ocr_started)
                        self._emit(
                            progress,
                            "ocr_finished",
                            page=page_index,
                            duration_seconds=page_timing["ocr_seconds"],
                            engine=ocr_result.engine,
                            confidence=ocr_result.confidence,
                            characters=len(ocr_result.text),
                        )
                        ocr_text = normalize_text(ocr_result.text)
                        if ocr_text and self._adds_new_text(native_text, ocr_text):
                            elements.append(
                                DocumentElement(
                                    element_id=f"p{page_index}-e{len(elements) + 1}",
                                    page=page_index,
                                    type="ocr_overlay",
                                    raw_text=ocr_result.text,
                                    normalized_text=ocr_text,
                                    source="pdf_page_ocr",
                                    confidence=ocr_result.confidence,
                                    metadata={
                                        "merge_reasons": merge_reasons,
                                        "native_quality_score": quality.score,
                                        **ocr_result.metadata,
                                    },
                                )
                            )
                            processing_route = "native_pdf_plus_ocr"
                            page_quality_score = min(quality.score, ocr_result.confidence or quality.score)
                    if rendered_page is not None and ocr_result is not None:
                        connector_started = time.perf_counter()
                        connector_result = self.connectors.detect(
                            rendered_page,
                            layout_result.region_dicts(),
                            ocr_result.metadata.get("lines") or [],
                        )
                        page_timing["connector_seconds"] = self._elapsed(connector_started)
                    pages.append(
                        ProcessedPage(
                            page_number=page_index,
                            processing_route=processing_route,
                            quality_score=page_quality_score,
                            elements=elements,
                            metadata={
                                "native_quality": quality.score,
                                "native_quality_reasons": quality.reasons,
                                "native_text_blocks": len(blocks),
                                "image_coverage": image_coverage,
                                "requires_layout": quality.requires_layout,
                                "requires_table_analysis": quality.requires_table_analysis,
                                "ocr_attempted": bool(ocr_result),
                                "ocr_engine": ocr_result.engine if ocr_result else None,
                                "ocr_quality": ocr_result.confidence if ocr_result else None,
                                "ocr_metadata": ocr_result.metadata if ocr_result else None,
                                "layout_engine": layout_result.engine,
                                "layout_regions": layout_result.region_dicts(),
                                "layout_metadata": layout_result.metadata,
                                "connector_regions": connector_result.regions,
                                "connector_metadata": connector_result.metadata,
                                "timing": {**page_timing, "route": processing_route},
                            },
                        )
                    )
                    self._emit(
                        progress,
                        "page_finished",
                        page=page_index,
                        route=processing_route,
                        timing=page_timing,
                    )
                    continue

                render_started = time.perf_counter()
                self._emit(progress, "page_render_started", page=page_index)
                rendered_page = self._render_page(page)
                page_timing["render_seconds"] = self._elapsed(render_started)
                self._emit(
                    progress,
                    "page_render_finished",
                    page=page_index,
                    duration_seconds=page_timing["render_seconds"],
                    width=rendered_page.width,
                    height=rendered_page.height,
                )
                layout_started = time.perf_counter()
                self._emit(progress, "layout_started", page=page_index)
                layout_result = self.layout.detect(rendered_page)
                page_timing["layout_seconds"] = self._elapsed(layout_started)
                self._emit(
                    progress,
                    "layout_finished",
                    page=page_index,
                    duration_seconds=page_timing["layout_seconds"],
                    engine=layout_result.engine,
                    regions=len(layout_result.regions),
                )
                ocr_started = time.perf_counter()
                self._emit(progress, "ocr_started", page=page_index)
                ocr_result = self.ocr.image_to_text(rendered_page)
                page_timing["ocr_seconds"] = self._elapsed(ocr_started)
                self._emit(
                    progress,
                    "ocr_finished",
                    page=page_index,
                    duration_seconds=page_timing["ocr_seconds"],
                    engine=ocr_result.engine,
                    confidence=ocr_result.confidence,
                    characters=len(ocr_result.text),
                )
                text = normalize_text(ocr_result.text)
                connector_started = time.perf_counter()
                connector_result = self.connectors.detect(
                    rendered_page,
                    layout_result.region_dicts(),
                    ocr_result.metadata.get("lines") or [],
                )
                page_timing["connector_seconds"] = self._elapsed(connector_started)
                elements = []
                if text:
                    elements.append(
                        DocumentElement(
                            element_id=f"p{page_index}-e1",
                            page=page_index,
                            type="paragraph",
                            raw_text=ocr_result.text,
                            normalized_text=text,
                            source="pdf_page_ocr",
                            confidence=ocr_result.confidence,
                            metadata={"quality_reasons": quality.reasons, **ocr_result.metadata},
                        )
                    )
                pages.append(
                    ProcessedPage(
                        page_number=page_index,
                        processing_route="pdf_page_ocr",
                        quality_score=ocr_result.confidence,
                        elements=elements,
                        metadata={
                            "native_quality": quality.score,
                            "native_quality_reasons": quality.reasons,
                            "native_text_blocks": len(blocks),
                            "image_coverage": image_coverage,
                            "ocr_engine": ocr_result.engine,
                            "ocr_quality": ocr_result.confidence,
                            "ocr_metadata": ocr_result.metadata,
                            "layout_engine": layout_result.engine,
                            "layout_regions": layout_result.region_dicts(),
                            "layout_metadata": layout_result.metadata,
                            "connector_regions": connector_result.regions,
                            "connector_metadata": connector_result.metadata,
                            "timing": {**page_timing, "route": "pdf_page_ocr"},
                        },
                    )
                )
                self._emit(
                    progress,
                    "page_finished",
                    page=page_index,
                    route="pdf_page_ocr",
                    timing=page_timing,
                )

        return ProcessedDocument(
            document_id=document_id,
            filename=filename,
            mime_type=mime_type,
            parser_version=self.config.parser_version,
            metadata={
                "page_count": len(pages),
                "languages": [],
                "timing": {
                    "extraction": {
                        "pdf_open_seconds": pdf_open_seconds,
                        "pages": [page.metadata.get("timing", {}) for page in pages],
                    }
                },
            },
            pages=pages,
        )

    def _native_text_blocks(self, page) -> list[dict]:
        raw_blocks = page.get_text("blocks") or []
        blocks = []
        for block in raw_blocks:
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            raw_text = text or ""
            normalized = normalize_inline_text(raw_text)
            if not normalized:
                continue
            blocks.append(
                {
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "raw_text": raw_text,
                    "normalized_text": normalized,
                }
            )
        return blocks

    def _image_coverage(self, page) -> float:
        page_area = max(float(page.rect.width * page.rect.height), 1.0)
        try:
            infos = page.get_image_info()
        except Exception:
            infos = []

        image_area = 0.0
        for info in infos:
            bbox = info.get("bbox")
            if not bbox:
                continue
            x0, y0, x1, y1 = bbox
            image_area += max(0.0, float(x1 - x0)) * max(0.0, float(y1 - y0))
        return min(1.0, image_area / page_area)

    def _render_page(self, page) -> Image.Image:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to render PDF pages for OCR") from exc

        matrix = fitz.Matrix(self.config.ocr_dpi / 72, self.config.ocr_dpi / 72)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    def _elapsed(self, started: float) -> float:
        return round(time.perf_counter() - started, 3)

    def _emit(self, progress: ProgressCallback | None, event: str, **details: Any) -> None:
        if progress is not None:
            progress(event, details)

    def _empty_layout_result(self) -> LayoutResult:
        return LayoutResult("layout_not_attempted", [], {"skipped": True})

    def _empty_connector_result(self) -> ConnectorDetectionResult:
        return ConnectorDetectionResult([], {"skipped": True, "reason": "ocr_not_attempted"})

    def _ocr_merge_reasons(self, native_text: str) -> list[str]:
        reasons = []
        lowered = native_text.lower()
        layout_markers = [
            "complete the table",
            "complete the flow chart",
            "complete the flowchart",
            "complete the diagram",
            "flow chart",
            "flowchart",
            "table.",
        ]
        if any(marker in lowered for marker in layout_markers):
            reasons.append("ielts_layout_question")

        for start, end in self._question_ranges(lowered):
            expected = end - start + 1
            if expected <= 1:
                continue
            text_without_headers = re.sub(r"questions?\s+\d{1,2}\s*(?:-|–|to)\s*\d{1,2}", "", lowered)
            present = 0
            for number in range(start, end + 1):
                if re.search(rf"(?<!\d){number}\s*[\.)]", text_without_headers) or re.search(
                    rf"\({number}\)", text_without_headers
                ):
                    present += 1
            if present < max(1, expected // 2):
                reasons.append(f"missing_question_numbers_{start}_{end}")
        return reasons

    def _question_ranges(self, text: str) -> list[tuple[int, int]]:
        ranges = []
        for match in re.finditer(r"questions?\s+(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})", text):
            start = int(match.group(1))
            end = int(match.group(2))
            if start > end:
                start, end = end, start
            ranges.append((start, end))
        return ranges

    def _adds_new_text(self, native_text: str, ocr_text: str) -> bool:
        native_terms = set(re.findall(r"[\w]+", native_text.lower(), flags=re.UNICODE))
        ocr_terms = set(re.findall(r"[\w]+", ocr_text.lower(), flags=re.UNICODE))
        if not ocr_terms:
            return False
        new_terms = ocr_terms - native_terms
        return len(new_terms) >= 5 or len(ocr_text) > len(native_text) * 1.1
