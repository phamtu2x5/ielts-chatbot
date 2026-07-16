import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from ..config import DocumentPipelineConfig
from ..models import DocumentElement, ProcessedDocument, ProcessedPage
from ..normalization import normalize_text
from ..layout import DocLayoutDetector
from ..ocr import OCRProcessor
from ..visual import WritingTaskTableParser


ProgressCallback = Callable[[str, dict[str, Any]], None]


class ImageExtractor:
    def __init__(self, config: DocumentPipelineConfig, ocr: OCRProcessor, layout: DocLayoutDetector) -> None:
        self.config = config
        self.ocr = ocr
        self.layout = layout
        self.visual_parser = WritingTaskTableParser()

    def extract(
        self,
        file_path: Path,
        filename: str,
        mime_type: str,
        document_id: str,
        progress: ProgressCallback | None = None,
    ) -> ProcessedDocument:
        started = time.perf_counter()
        open_started = time.perf_counter()
        self._emit(progress, "image_open_started")
        with Image.open(file_path) as image:
            image = image.convert("RGB")
            open_seconds = self._elapsed(open_started)
            self._emit(
                progress,
                "image_open_finished",
                duration_seconds=open_seconds,
                width=image.width,
                height=image.height,
            )
            layout_started = time.perf_counter()
            self._emit(progress, "layout_started", page=1)
            layout_result = self.layout.detect(image)
            layout_seconds = self._elapsed(layout_started)
            self._emit(
                progress,
                "layout_finished",
                page=1,
                duration_seconds=layout_seconds,
                engine=layout_result.engine,
                regions=len(layout_result.regions),
                metadata=layout_result.metadata,
            )
            ocr_started = time.perf_counter()
            self._emit(progress, "ocr_started", page=1)
            ocr_result = self.ocr.image_to_text(image)
            ocr_seconds = self._elapsed(ocr_started)
            self._emit(
                progress,
                "ocr_finished",
                page=1,
                duration_seconds=ocr_seconds,
                engine=ocr_result.engine,
                confidence=ocr_result.confidence,
                characters=len(ocr_result.text),
                words=ocr_result.metadata.get("word_count"),
                boxes=len(ocr_result.metadata.get("boxes") or []),
                text_preview=ocr_result.text[:4000],
            )

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
        visual_parse_started = time.perf_counter()
        self._emit(progress, "visual_parse_started", page=1)
        parsed_visual = self.visual_parser.parse(
            text,
            ocr_lines=ocr_result.metadata.get("lines") or [],
            layout_regions=layout_result.region_dicts(),
        )
        visual_parse_seconds = self._elapsed(visual_parse_started)
        parsed_table = parsed_visual.table if parsed_visual else {}
        self._emit(
            progress,
            "visual_parse_finished",
            page=1,
            duration_seconds=visual_parse_seconds,
            parsed=bool(parsed_visual),
            document_type=parsed_visual.document_type if parsed_visual else None,
            columns=len(parsed_table.get("columns") or []),
            rows=len(parsed_table.get("rows") or []),
        )
        total_seconds = self._elapsed(started)
        extraction_timing = {
            "image_open_seconds": open_seconds,
            "layout_seconds": layout_seconds,
            "ocr_seconds": ocr_seconds,
            "visual_parse_seconds": visual_parse_seconds,
            "total_seconds": total_seconds,
        }
        metadata = {
            "page_count": 1,
            "languages": [],
            "ocr_engine": ocr_result.engine,
            "ocr_quality": ocr_result.confidence,
            "ocr_metadata": ocr_result.metadata,
            "layout_engine": layout_result.engine,
            "layout_regions": layout_result.region_dicts(),
            "layout_metadata": layout_result.metadata,
            "timing": {"extraction": extraction_timing},
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
                        bbox=parsed_visual.table.get("bbox") or None,
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
                    metadata={
                        "ocr_engine": ocr_result.engine,
                        "ocr_quality": ocr_result.confidence,
                        "ocr_metadata": ocr_result.metadata,
                        "layout_engine": layout_result.engine,
                        "layout_regions": layout_result.region_dicts(),
                        "layout_metadata": layout_result.metadata,
                        "timing": extraction_timing,
                    },
                )
            ],
        )

    def _elapsed(self, started: float) -> float:
        return round(time.perf_counter() - started, 3)

    def _emit(self, progress: ProgressCallback | None, event: str, **details: Any) -> None:
        if progress is not None:
            progress(event, details)
