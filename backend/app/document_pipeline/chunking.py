from .config import DocumentPipelineConfig
from .models import DocumentChunk, DocumentElement, ProcessedDocument


def estimate_tokens(text: str) -> int:
    # Good enough for routing chunk size without adding tokenizer weight.
    return max(1, int(len(text) / 4))


class SemanticChunker:
    def __init__(self, config: DocumentPipelineConfig) -> None:
        self.config = config

    def chunk(self, document: ProcessedDocument) -> list[DocumentChunk]:
        if document.metadata.get("document_type") == "ielts_writing_task_1":
            chunks = self._chunk_writing_task_1(document)
            if chunks:
                return chunks

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

    def _chunk_writing_task_1(self, document: ProcessedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        prompt_elements = [element for element in document.elements if element.type == "writing_prompt"]
        table_elements = [element for element in document.elements if element.type == "table"]
        raw_elements = [element for element in document.elements if element.type == "paragraph"]

        for element in prompt_elements:
            chunks.append(
                self._make_special_chunk(
                    document=document,
                    element=element,
                    chunk_index=len(chunks),
                    unit_type="writing_prompt",
                    chunk_reason="writing_prompt",
                    retrieval_prefix="IELTS Writing Task 1 prompt. Academic table task.",
                )
            )

        for element in table_elements:
            table = element.metadata.get("table") or {}
            chunks.append(
                self._make_special_chunk(
                    document=document,
                    element=element,
                    chunk_index=len(chunks),
                    unit_type="writing_table",
                    chunk_reason="writing_table",
                    retrieval_prefix="IELTS Writing Task 1 data table. Structured table values.",
                    extra_metadata={"table": table},
                )
            )
            columns = table.get("columns") or []
            for row_index, row in enumerate(table.get("rows") or [], 1):
                row_text = self._row_text(columns, row)
                row_element = DocumentElement(
                    element_id=f"{element.element_id}-r{row_index}",
                    page=element.page,
                    type="table_row",
                    raw_text=row_text,
                    normalized_text=row_text,
                    source=element.source,
                    confidence=element.confidence,
                    metadata={
                        "document_type": document.metadata.get("document_type"),
                        "task_type": document.metadata.get("task_type"),
                        "table_row": row,
                        "table_columns": columns,
                    },
                )
                chunks.append(
                    self._make_special_chunk(
                        document=document,
                        element=row_element,
                        chunk_index=len(chunks),
                        unit_type="table_row",
                        chunk_reason="table_row",
                        retrieval_prefix="IELTS Writing Task 1 table row.",
                        extra_metadata={
                            "table_row": row,
                            "table_columns": columns,
                        },
                    )
                )

        if not chunks and raw_elements:
            chunks.append(self._make_chunk(document, raw_elements, [], 0))
        return chunks

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
                "document_type": document.metadata.get("document_type"),
                "task_type": document.metadata.get("task_type"),
                "element_types": sorted({element.type for element in elements}),
                "sources": sorted({element.source for element in elements}),
            },
        )

    def _make_special_chunk(
        self,
        document: ProcessedDocument,
        element: DocumentElement,
        chunk_index: int,
        unit_type: str,
        chunk_reason: str,
        retrieval_prefix: str,
        extra_metadata: dict | None = None,
    ) -> DocumentChunk:
        metadata = {
            "mime_type": document.mime_type,
            "parser_version": document.parser_version,
            "document_type": document.metadata.get("document_type"),
            "task_type": document.metadata.get("task_type"),
            "unit_type": unit_type,
            "chunk_reason": chunk_reason,
            "element_types": [element.type],
            "sources": [element.source],
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return DocumentChunk(
            chunk_id=f"{document.document_id}-{unit_type}-{chunk_index + 1}",
            document_id=document.document_id,
            source_file=document.filename,
            pages=[element.page],
            element_ids=[element.element_id],
            heading_path=[unit_type],
            text=element.normalized_text,
            retrieval_text=f"{retrieval_prefix}\n\n{element.normalized_text}",
            display_text=element.normalized_text,
            token_count=estimate_tokens(element.normalized_text),
            min_confidence=element.confidence,
            chunk_index=chunk_index,
            metadata=metadata,
        )

    def _row_text(self, columns: list, row: list) -> str:
        if not columns or len(columns) != len(row):
            return " | ".join(str(cell) for cell in row)
        return "; ".join(f"{column}: {cell}" for column, cell in zip(columns, row))

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
