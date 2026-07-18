from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DOCUMENT_MARKERS = [
    "tài liệu",
    "file",
    "pdf",
    "docx",
    "trong bài",
    "bài đọc",
    "passage",
    "question",
    "questions",
    "câu hỏi",
    "trang",
    "page",
    "bảng",
    "table",
    "flowchart",
    "flow chart",
    "sơ đồ",
    "diagram",
    "ảnh",
    "hình",
    "image",
    "writing",
    "task 1",
    "task 2",
    "đề trên",
    "đã tải",
    "uploaded",
]

COLLECTION_MARKERS = [
    "những tài liệu",
    "các tài liệu",
    "toàn bộ tài liệu",
    "tài liệu đã tải",
    "uploaded documents",
]


@dataclass(frozen=True)
class DocumentScope:
    requested_document_ids: list[str]
    allowed_document_ids: list[str]
    resolved_document_ids: list[str]
    matched_files: list[str]
    method: str
    ambiguous: bool
    document_grounded: bool
    reason: str

    def to_debug(self) -> dict[str, Any]:
        return asdict(self)


def normalize_reference(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(character for character in normalized if not unicodedata.combining(character))
    return " ".join(re.findall(r"[a-z0-9]+", without_marks))


def is_document_grounded_query(message: str, catalog: list[dict[str, Any]] | None = None) -> bool:
    lowered = message.casefold()
    normalized = normalize_reference(message)
    if any(_contains_phrase(lowered, marker.casefold()) for marker in DOCUMENT_MARKERS):
        return True
    return any(_filename_match_score(normalized, item.get("source_file", "")) > 0 for item in catalog or [])


def resolve_document_scope(
    message: str,
    catalog: list[dict[str, Any]],
    requested_document_ids: list[str] | None = None,
) -> DocumentScope:
    requested = list(dict.fromkeys(requested_document_ids or []))
    available_ids = {
        str(document_id)
        for item in catalog
        for document_id in item.get("document_ids", [])
        if document_id
    }
    allowed = [document_id for document_id in requested if document_id in available_ids]
    if not requested:
        allowed = sorted(available_ids)

    allowed_entries = [
        item
        for item in catalog
        if any(document_id in allowed for document_id in item.get("document_ids", []))
    ]
    # An explicit client scope means the answer must stay grounded even when
    # the query itself only names a section title or topic.
    grounded = bool(requested) or is_document_grounded_query(message, allowed_entries or catalog)
    if requested and not allowed:
        return DocumentScope(
            requested,
            [],
            [],
            [],
            "invalid_requested_scope",
            False,
            grounded,
            "None of the requested document IDs exists in the current index.",
        )
    if len(allowed) == 1:
        entry = next(
            (item for item in allowed_entries if allowed[0] in item.get("document_ids", [])),
            None,
        )
        return DocumentScope(
            requested,
            allowed,
            allowed,
            [entry.get("source_file", "unknown")] if entry else [],
            "requested_single" if requested else "catalog_single",
            False,
            grounded,
            "A single document is available in the allowed scope.",
        )

    normalized_query = normalize_reference(message)
    scored: list[tuple[float, dict[str, Any]]] = []
    for entry in allowed_entries:
        score = _filename_match_score(normalized_query, entry.get("source_file", ""))
        score += _modality_match_score(message, entry)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda item: item[0], reverse=True)

    if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        entry = scored[0][1]
        resolved = [
            document_id
            for document_id in entry.get("document_ids", [])
            if document_id in allowed
        ]
        return DocumentScope(
            requested,
            allowed,
            resolved,
            [entry.get("source_file", "unknown")],
            "catalog_reference",
            False,
            grounded,
            "The query uniquely matched a file name or document modality.",
        )

    if any(normalize_reference(marker) in normalized_query for marker in COLLECTION_MARKERS):
        return DocumentScope(
            requested,
            allowed,
            allowed,
            [entry.get("source_file", "unknown") for entry in allowed_entries],
            "collection",
            False,
            grounded,
            "The query explicitly targets the uploaded document collection.",
        )

    return DocumentScope(
        requested,
        allowed,
        [],
        [entry.get("source_file", "unknown") for entry in allowed_entries],
        "ambiguous" if grounded and len(allowed_entries) > 1 else "unresolved",
        bool(grounded and len(allowed_entries) > 1),
        grounded,
        "Multiple documents remain possible and the query does not identify one uniquely.",
    )


def _filename_match_score(normalized_query: str, source_file: str) -> float:
    stem = normalize_reference(Path(source_file).stem)
    if not stem:
        return 0.0
    if stem in normalized_query:
        return 100.0
    query_sequence = normalized_query.split()
    file_sequence = stem.split()
    query_terms = set(query_sequence)
    file_terms = set(file_sequence)
    query_numbers = {term for term in query_terms if term.isdigit()}
    file_numbers = {term for term in file_terms if term.isdigit()}
    if query_numbers and file_numbers and not query_numbers.intersection(file_numbers):
        return 0.0
    overlap = query_terms.intersection(file_terms)
    if len(overlap) < 2:
        return 0.0
    longest_phrase = 0
    normalized_padded = f" {normalized_query} "
    for length in range(2, len(file_sequence) + 1):
        if any(
            f" {' '.join(file_sequence[start:start + length])} " in normalized_padded
            for start in range(0, len(file_sequence) - length + 1)
        ):
            longest_phrase = length
    return (
        float(len(overlap))
        + len(overlap) / max(1, len(file_terms))
        + longest_phrase * 20.0
    )


def _modality_match_score(message: str, entry: dict[str, Any]) -> float:
    lowered = message.casefold()
    mime_types = entry.get("mime_types", [])
    if any(_contains_phrase(lowered, marker) for marker in ["ảnh", "hình", "image", "screenshot"]):
        return 30.0 if any(str(mime).startswith("image/") for mime in mime_types) else 0.0
    return 0.0


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))
