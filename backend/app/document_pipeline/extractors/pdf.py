import re
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps

from ..config import DocumentPipelineConfig
from ..connectors import ConnectorDetectionResult, RasterConnectorDetector
from ..layout import DocLayoutDetector, LayoutResult
from ..models import DocumentElement, ProcessedDocument, ProcessedPage
from ..normalization import normalize_inline_text, normalize_text
from ..ocr import OCRProcessor, OCRResult
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
                page_timing["visual_ocr_retry_seconds"] = 0.0
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
                        retry_started = time.perf_counter()
                        ocr_result = self._retry_visual_ocr(
                            rendered_page,
                            layout_result.region_dicts(),
                            ocr_result,
                        )
                        page_timing["visual_ocr_retry_seconds"] = self._elapsed(retry_started)
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
                retry_started = time.perf_counter()
                ocr_result = self._retry_visual_ocr(
                    rendered_page,
                    layout_result.region_dicts(),
                    ocr_result,
                )
                page_timing["visual_ocr_retry_seconds"] = self._elapsed(retry_started)
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

    def _retry_visual_ocr(
        self,
        image: Image.Image,
        layout_regions: list[dict[str, Any]],
        result: OCRResult,
    ) -> OCRResult:
        if not self.config.visual_ocr_retry_enabled or not result.text:
            return result

        missing_numbers = self._missing_visual_question_numbers(result.text)
        if not missing_numbers:
            return result

        visual_regions = [
            region
            for region in layout_regions
            if str(region.get("type") or "").strip().lower().replace(" ", "_") in {"table", "figure"}
        ][: self.config.visual_ocr_retry_max_regions]
        if not visual_regions:
            return result

        metadata = dict(result.metadata)
        merged_lines = [dict(line) for line in metadata.get("lines") or []]
        retries = []
        recovered: set[int] = set()
        scale = self.config.visual_ocr_retry_scale
        for region in visual_regions:
            bbox = self._flat_bbox(region.get("bbox"))
            if bbox is None:
                continue
            x0, y0, x1, y1 = self._clip_image_bbox(bbox, image.width, image.height)
            crop = ImageOps.autocontrast(image.crop((x0, y0, x1, y1)).convert("L")).convert("RGB")
            if scale > 1:
                crop = crop.resize(
                    (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            retry = self.ocr.image_to_text(crop)
            retry_lines = []
            retry_recovered: set[int] = set()
            for line in retry.metadata.get("lines") or []:
                line_numbers = self._question_numbers_in_text(str(line.get("text") or ""))
                matched = line_numbers & (missing_numbers - recovered)
                translated_bbox = self._translate_bbox(line.get("bbox"), x0, y0, scale)
                if not matched or translated_bbox is None:
                    continue
                translated = {**line, "bbox": translated_bbox, "source": "visual_region_ocr_retry"}
                self._replace_overlapping_line(merged_lines, translated)
                retry_lines.append(translated)
                retry_recovered.update(matched)
            recovered.update(retry_recovered)
            retries.append(
                {
                    "region_type": region.get("type"),
                    "bbox": [float(value) for value in (x0, y0, x1, y1)],
                    "scale": scale,
                    "engine": retry.engine,
                    "confidence": round(retry.confidence, 4),
                    "recovered_question_numbers": sorted(retry_recovered),
                    "accepted_lines": retry_lines,
                }
            )
            if missing_numbers <= recovered:
                break

        metadata["visual_ocr_retries"] = retries
        metadata["visual_ocr_retry_missing"] = sorted(missing_numbers)
        metadata["visual_ocr_retry_recovered"] = sorted(recovered)
        if not recovered:
            return OCRResult(result.text, result.confidence, result.engine, metadata)

        metadata["lines"] = merged_lines
        metadata["boxes"] = [line.get("bbox") for line in merged_lines if line.get("bbox")]
        text = normalize_text("\n".join(str(line.get("text") or "") for line in merged_lines))
        return OCRResult(text, result.confidence, result.engine, metadata)

    def _missing_visual_question_numbers(self, text: str) -> set[int]:
        header_pattern = re.compile(r"questions?\s+(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})", re.IGNORECASE)
        headers = list(header_pattern.finditer(text))
        missing: set[int] = set()
        visual_markers = (
            "complete the table",
            "complete the flow chart",
            "complete the flowchart",
            "label the diagram",
            "label the figure",
        )
        for index, header in enumerate(headers):
            section_end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
            section = text[header.end() : section_end]
            if not any(marker in section.lower() for marker in visual_markers):
                continue
            start, end = int(header.group(1)), int(header.group(2))
            if start > end:
                start, end = end, start
            present = self._question_numbers_in_text(section)
            missing.update(set(range(start, end + 1)) - present)
        return missing

    def _question_numbers_in_text(self, text: str) -> set[int]:
        parenthesized = {int(number) for number in re.findall(r"\((\d{1,2})\)", text)}
        numbered = {
            int(number)
            for number in re.findall(r"(?:^|\s)(\d{1,2})\s*[\.)](?=\s|$)", text, flags=re.MULTILINE)
        }
        return parenthesized | numbered

    def _replace_overlapping_line(self, lines: list[dict[str, Any]], replacement: dict[str, Any]) -> None:
        replacement_bbox = self._flat_bbox(replacement.get("bbox"))
        if replacement_bbox is None:
            lines.append(replacement)
            return
        for index, line in enumerate(lines):
            bbox = self._flat_bbox(line.get("bbox"))
            if bbox is not None and self._bbox_overlap_ratio(bbox, replacement_bbox) >= 0.5:
                lines[index] = replacement
                return
        lines.append(replacement)

    def _translate_bbox(
        self,
        value: Any,
        offset_x: int,
        offset_y: int,
        scale: float,
    ) -> list[list[float]] | list[float] | None:
        if not isinstance(value, (list, tuple)):
            return None
        if len(value) >= 4 and all(isinstance(item, (int, float)) for item in value[:4]):
            x0, y0, x1, y1 = (float(item) for item in value[:4])
            return [
                offset_x + x0 / scale,
                offset_y + y0 / scale,
                offset_x + x1 / scale,
                offset_y + y1 / scale,
            ]
        points = [point for point in value if isinstance(point, (list, tuple)) and len(point) >= 2]
        if not points:
            return None
        return [
            [offset_x + float(point[0]) / scale, offset_y + float(point[1]) / scale]
            for point in points
        ]

    def _flat_bbox(self, value: Any) -> list[float] | None:
        if not isinstance(value, (list, tuple)):
            return None
        if len(value) >= 4 and all(isinstance(item, (int, float)) for item in value[:4]):
            x0, y0, x1, y1 = (float(item) for item in value[:4])
            return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        points = [point for point in value if isinstance(point, (list, tuple)) and len(point) >= 2]
        if not points:
            return None
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return [min(xs), min(ys), max(xs), max(ys)]

    def _clip_image_bbox(self, bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
        x0 = max(0, min(width - 1, int(bbox[0])))
        y0 = max(0, min(height - 1, int(bbox[1])))
        x1 = max(x0 + 1, min(width, int(bbox[2] + 0.999)))
        y1 = max(y0 + 1, min(height, int(bbox[3] + 0.999)))
        return x0, y0, x1, y1

    def _bbox_overlap_ratio(self, first: list[float], second: list[float]) -> float:
        width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
        height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
        intersection = width * height
        smaller = min(
            max(1.0, (first[2] - first[0]) * (first[3] - first[1])),
            max(1.0, (second[2] - second[0]) * (second[3] - second[1])),
        )
        return intersection / smaller

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
