from .config import DocumentPipelineConfig
from .models import DocumentChunk, DocumentElement, ProcessedDocument


def estimate_tokens(text: str) -> int:
    # Good enough for routing chunk size without adding tokenizer weight.
    return max(1, int(len(text) / 4))


class SemanticChunker:
    def __init__(self, config: DocumentPipelineConfig) -> None:
        self.config = config

    def chunk(self, document: ProcessedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        current: list[DocumentElement] = []
        heading_path: list[str] = []

        for element in document.elements:
            if not element.normalized_text.strip():
                continue
            if element.type == "heading":
                if current:
                    chunks.append(self._make_chunk(document, current, heading_path, len(chunks)))
                    current = self._overlap_elements(current)
                heading_path = [element.normalized_text]
                current.append(element)
                continue

            current_text = self._elements_text(current, heading_path)
            projected_text = self._elements_text(current + [element], heading_path)
            current_is_ready = estimate_tokens(current_text) >= self.config.chunk_target_tokens
            projected_is_too_large = estimate_tokens(projected_text) > self.config.chunk_max_tokens
            if current and (current_is_ready or projected_is_too_large):
                chunks.append(self._make_chunk(document, current, heading_path, len(chunks)))
                current = self._overlap_elements(current)
            current.append(element)

        if current:
            chunks.append(self._make_chunk(document, current, heading_path, len(chunks)))

        return [chunk for chunk in chunks if chunk.text.strip()]

    def _overlap_elements(self, elements: list[DocumentElement]) -> list[DocumentElement]:
        overlap: list[DocumentElement] = []
        tokens = 0
        for element in reversed(elements):
            if element.type == "heading":
                continue
            element_tokens = estimate_tokens(element.normalized_text)
            if tokens + element_tokens > self.config.chunk_overlap_tokens:
                break
            overlap.insert(0, element)
            tokens += element_tokens
        return overlap

    def _make_chunk(
        self,
        document: ProcessedDocument,
        elements: list[DocumentElement],
        heading_path: list[str],
        chunk_index: int,
    ) -> DocumentChunk:
        text = self._elements_text(elements, heading_path)
        pages = sorted({element.page for element in elements})
        confidences = [element.confidence for element in elements]
        return DocumentChunk(
            chunk_id=f"{document.document_id}-c{chunk_index + 1}",
            document_id=document.document_id,
            source_file=document.filename,
            pages=pages,
            element_ids=[element.element_id for element in elements],
            heading_path=heading_path,
            text=text,
            token_count=estimate_tokens(text),
            min_confidence=min(confidences) if confidences else 0.0,
            chunk_index=chunk_index,
            metadata={
                "mime_type": document.mime_type,
                "parser_version": document.parser_version,
                "element_types": sorted({element.type for element in elements}),
                "sources": sorted({element.source for element in elements}),
            },
        )

    def _elements_text(self, elements: list[DocumentElement], heading_path: list[str]) -> str:
        parts = []
        if heading_path:
            parts.append("# " + " > ".join(heading_path))
        current_page = None
        for element in elements:
            if element.page != current_page:
                current_page = element.page
                parts.append(f"[Page {current_page}]")
            text = element.normalized_text.strip()
            if not text:
                continue
            if element.type == "heading":
                parts.append(f"## {text}")
            else:
                parts.append(text)
        return "\n\n".join(parts)
