from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
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
