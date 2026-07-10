from pathlib import Path

from PIL import Image

from ..config import DocumentPipelineConfig
from ..models import DocumentElement, ProcessedDocument, ProcessedPage
from ..normalization import normalize_inline_text, normalize_text
from ..ocr import OCRProcessor
from ..quality import evaluate_native_page_text


class PDFExtractor:
    def __init__(self, config: DocumentPipelineConfig, ocr: OCRProcessor) -> None:
        self.config = config
        self.ocr = ocr

    def extract(self, file_path: Path, filename: str, mime_type: str, document_id: str) -> ProcessedDocument:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to process PDF files") from exc

        pages = []
        with fitz.open(file_path) as document:
            page_count = len(document)
            if page_count > self.config.max_pdf_pages:
                raise ValueError(f"PDF có {page_count} trang, vượt giới hạn {self.config.max_pdf_pages} trang.")

            for page_index, page in enumerate(document, 1):
                blocks = self._native_text_blocks(page)
                native_text = "\n".join(block["text"] for block in blocks)
                image_coverage = self._image_coverage(page)
                quality = evaluate_native_page_text(native_text, len(blocks), image_coverage, self.config)

                if quality.native_text_is_usable:
                    elements = [
                        DocumentElement(
                            element_id=f"p{page_index}-e{idx}",
                            page=page_index,
                            type="paragraph",
                            raw_text=block["text"],
                            normalized_text=normalize_inline_text(block["text"]),
                            source="native_pdf",
                            confidence=quality.score,
                            bbox=block["bbox"],
                            metadata={"quality_reasons": quality.reasons},
                        )
                        for idx, block in enumerate(blocks, 1)
                        if normalize_inline_text(block["text"])
                    ]
                    pages.append(
                        ProcessedPage(
                            page_number=page_index,
                            processing_route="native_pdf",
                            quality_score=quality.score,
                            elements=elements,
                        )
                    )
                    continue

                ocr_result = self._ocr_page(page)
                text = normalize_text(ocr_result.text)
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
                    )
                )

        return ProcessedDocument(
            document_id=document_id,
            filename=filename,
            mime_type=mime_type,
            parser_version=self.config.parser_version,
            metadata={"page_count": len(pages), "languages": []},
            pages=pages,
        )

    def _native_text_blocks(self, page) -> list[dict]:
        raw_blocks = page.get_text("blocks") or []
        blocks = []
        for block in raw_blocks:
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            normalized = normalize_inline_text(text or "")
            if not normalized:
                continue
            blocks.append({"bbox": [float(x0), float(y0), float(x1), float(y1)], "text": normalized})
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

    def _ocr_page(self, page):
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to render PDF pages for OCR") from exc

        matrix = fitz.Matrix(self.config.ocr_dpi / 72, self.config.ocr_dpi / 72)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        return self.ocr.image_to_text(image)
