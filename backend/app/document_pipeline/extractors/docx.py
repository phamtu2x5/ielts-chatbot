from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from ..config import DocumentPipelineConfig
from ..models import DocumentElement, ProcessedDocument, ProcessedPage
from ..normalization import normalize_inline_text, normalize_text
from ..ocr import OCRProcessor


class DOCXExtractor:
    def __init__(self, config: DocumentPipelineConfig, ocr: OCRProcessor) -> None:
        self.config = config
        self.ocr = ocr

    def extract(self, file_path: Path, filename: str, mime_type: str, document_id: str) -> ProcessedDocument:
        try:
            from docx import Document
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise RuntimeError("python-docx is required to process DOCX files") from exc

        document = Document(str(file_path))
        elements: list[DocumentElement] = []
        current_heading = None
        element_index = 1

        for child in document.element.body.iterchildren():
            if isinstance(child, CT_P):
                paragraph = Paragraph(child, document)
                text = normalize_inline_text(paragraph.text)
                if not text:
                    continue
                style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
                element_type = "heading" if style_name.startswith("heading") else "paragraph"
                if element_type == "heading":
                    current_heading = text
                elements.append(
                    DocumentElement(
                        element_id=f"p1-e{element_index}",
                        page=1,
                        type=element_type,
                        raw_text=paragraph.text,
                        normalized_text=text,
                        source="native_docx",
                        confidence=1.0,
                        parent_heading=None if element_type == "heading" else current_heading,
                        metadata={"style": paragraph.style.name if paragraph.style else None},
                    )
                )
                element_index += 1
            elif isinstance(child, CT_Tbl):
                table = Table(child, document)
                markdown = self._table_to_markdown(table)
                if not markdown:
                    continue
                elements.append(
                    DocumentElement(
                        element_id=f"p1-e{element_index}",
                        page=1,
                        type="table",
                        raw_text=markdown,
                        normalized_text=markdown,
                        source="native_docx",
                        confidence=1.0,
                        parent_heading=current_heading,
                    )
                )
                element_index += 1

        for image_name, image in self._embedded_images(file_path):
            ocr_result = self.ocr.image_to_text(image)
            text = normalize_text(ocr_result.text)
            if not text:
                continue
            elements.append(
                DocumentElement(
                    element_id=f"p1-e{element_index}",
                    page=1,
                    type="figure_text",
                    raw_text=ocr_result.text,
                    normalized_text=text,
                    source="docx_image_ocr",
                    confidence=ocr_result.confidence,
                    parent_heading=current_heading,
                    metadata={"image_name": image_name, **ocr_result.metadata},
                )
            )
            element_index += 1

        return ProcessedDocument(
            document_id=document_id,
            filename=filename,
            mime_type=mime_type,
            parser_version=self.config.parser_version,
            metadata={"page_count": 1, "languages": []},
            pages=[ProcessedPage(page_number=1, processing_route="native_docx", quality_score=1.0, elements=elements)],
        )

    def _table_to_markdown(self, table) -> str:
        rows = []
        for row in table.rows:
            cells = [normalize_inline_text(cell.text) for cell in row.cells]
            if any(cells):
                rows.append(cells)
        if not rows:
            return ""

        max_cols = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]
        header = normalized_rows[0]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * max_cols) + " |",
        ]
        for row in normalized_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def _embedded_images(self, file_path: Path):
        with ZipFile(file_path) as archive:
            for name in archive.namelist():
                if not name.startswith("word/media/"):
                    continue
                try:
                    data = archive.read(name)
                    image = Image.open(BytesIO(data)).convert("RGB")
                    image.load()
                    yield name, image
                except (UnidentifiedImageError, OSError):
                    continue
