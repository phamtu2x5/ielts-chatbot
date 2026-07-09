import os
import shutil
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


def extract_pdf_text(path: Path) -> str:
    text = extract_with_pypdf(path)
    if text.strip():
        return text

    text = extract_with_pymupdf(path)
    if text.strip():
        return text

    return extract_with_ocr(path)


def extract_with_pypdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def extract_with_pymupdf(path: Path) -> str:
    try:
        import fitz
    except ImportError:
        return ""

    pages = []
    with fitz.open(path) as document:
        for page in document:
            text = page.get_text("text") or ""
            if text.strip():
                pages.append(text.strip())
    return "\n\n".join(pages)


def ocr_image(pytesseract, image) -> str:
    lang = os.getenv("PDF_OCR_LANG", "vie+eng")
    try:
        return pytesseract.image_to_string(image, lang=lang)
    except pytesseract.TesseractError:
        if lang == "eng":
            return ""
        try:
            return pytesseract.image_to_string(image, lang="eng")
        except pytesseract.TesseractError:
            return ""


def extract_with_ocr(path: Path) -> str:
    if not shutil.which("tesseract"):
        return ""

    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    max_pages = int(os.getenv("PDF_OCR_MAX_PAGES", "20"))
    dpi = int(os.getenv("PDF_OCR_DPI", "180"))
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pages = []

    with fitz.open(path) as document:
        for page_index, page in enumerate(document):
            if page_index >= max_pages:
                break
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            text = ocr_image(pytesseract, image)
            if text.strip():
                pages.append(text.strip())

    return "\n\n".join(pages)


def chunk_text(text: str, source_file: str, chunk_size: int, overlap: int) -> List[Dict]:
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks = []
    start = 0
    index = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(
                {
                    "text": chunk,
                    "source_file": source_file,
                    "chunk_index": index,
                    "metadata": {"start_char": start, "end_char": end},
                }
            )
            index += 1
        if end == len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks
