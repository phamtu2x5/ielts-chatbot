from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DOCUMENT_REFERENCE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\btrong\s+(?:tài liệu|file|pdf|docx|bài đọc|ảnh|hình|bảng|sơ đồ)\b",
        r"\b(?:tài liệu|file|pdf|docx|bài đọc|ảnh|hình|bảng|sơ đồ)\s+(?:này|trên|đó|đã tải)\b",
        r"\b(?:uploaded|attached)\s+(?:material|document|file|pdf|image)\b",
        r"\b(?:in|from)\s+(?:the\s+)?(?:document|file|pdf|image|table|diagram)\b",
        r"\b(?:page|trang)\s*\d{1,4}\b",
        r"\b(?:nội dung|tóm tắt|tổng quan)\s+(?:của\s+)?(?:tài liệu|file|pdf)\b",
        r"\b(?:tài liệu|file|pdf)\s+(?:gồm|chứa|nhắc|nói về)\b",
    ]
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
    if any(pattern.search(message) for pattern in DOCUMENT_REFERENCE_PATTERNS):
        return True
    if re.search(
        r"\b(?:reading\s+passage|passage|questions?|câu(?:\s+hỏi)?)\s*\d{1,3}\b",
        lowered,
    ):
        return True
    matches = [
        item
        for item in catalog or []
        if _catalog_match_score(normalized, lowered, item) > 0
    ]
    return len(matches) == 1


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
    # Client-provided IDs constrain which documents may be searched. They do
    # not by themselves mean that a general chat message requires RAG.
    grounded = is_document_grounded_query(message)
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
        score = _catalog_match_score(normalized_query, message.casefold(), entry)
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
            True,
            "The query uniquely matched a file name, section title, or explicit document modality.",
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


def apply_document_affinity(
    scope: DocumentScope,
    catalog: list[dict[str, Any]],
    affinity_document_ids: list[str] | None,
) -> DocumentScope:
    """Use client-carried conversation context only when this query did not select a document."""
    if scope.method not in {"ambiguous", "unresolved"} or not affinity_document_ids:
        return scope

    allowed_affinity = [
        document_id
        for document_id in dict.fromkeys(affinity_document_ids)
        if document_id in scope.allowed_document_ids
    ]
    if len(allowed_affinity) != 1:
        return scope

    matched_files = [
        item.get("source_file", "unknown")
        for item in catalog
        if any(document_id in allowed_affinity for document_id in item.get("document_ids", []))
    ]
    return DocumentScope(
        requested_document_ids=scope.requested_document_ids,
        allowed_document_ids=scope.allowed_document_ids,
        resolved_document_ids=allowed_affinity,
        matched_files=matched_files,
        method="conversation_affinity",
        ambiguous=False,
        document_grounded=scope.document_grounded,
        reason="The current query did not select a document, so the previous turn's document scope was reused.",
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
    longest_phrase = 0
    normalized_padded = f" {normalized_query} "
    for length in range(2, len(file_sequence) + 1):
        if any(
            f" {' '.join(file_sequence[start:start + length])} " in normalized_padded
            for start in range(0, len(file_sequence) - length + 1)
        ):
            longest_phrase = length
    if longest_phrase < 3:
        return 0.0
    return (
        float(len(overlap))
        + len(overlap) / max(1, len(file_terms))
        + longest_phrase * 20.0
    )


def _section_title_match_score(normalized_query: str, titles: list[str]) -> float:
    query_terms = normalized_query.split()
    query_numbers = {term for term in query_terms if term.isdigit()}
    for title in titles:
        normalized_title = normalize_reference(str(title))
        title_terms = normalized_title.split()
        if not normalized_title:
            continue
        title_numbers = {term for term in title_terms if term.isdigit()}
        if query_numbers and title_numbers and not query_numbers.intersection(title_numbers):
            continue
        if normalized_title in normalized_query and (
            len(title_terms) >= 2 or len(normalized_title) >= 5
        ):
            return 120.0 + len(title_terms)
        longest_phrase = _longest_contiguous_match(query_terms, title_terms)
        if longest_phrase >= 3:
            return 100.0 + longest_phrase
    return 0.0


def _longest_contiguous_match(first: list[str], second: list[str]) -> int:
    longest = 0
    for first_start in range(len(first)):
        for second_start in range(len(second)):
            length = 0
            while (
                first_start + length < len(first)
                and second_start + length < len(second)
                and first[first_start + length] == second[second_start + length]
            ):
                length += 1
            longest = max(longest, length)
    return longest


def _catalog_match_score(
    normalized_query: str,
    lowered_message: str,
    entry: dict[str, Any],
) -> float:
    return (
        _filename_match_score(normalized_query, entry.get("source_file", ""))
        + _section_title_match_score(normalized_query, entry.get("section_titles") or [])
        + _modality_match_score(lowered_message, entry)
    )


def _modality_match_score(message: str, entry: dict[str, Any]) -> float:
    mime_types = entry.get("mime_types", [])
    image_reference = any(
        re.search(pattern, message, flags=re.IGNORECASE)
        for pattern in [
            r"\btrong\s+(?:ảnh|hình|image|screenshot)\b",
            r"\b(?:ảnh|hình|image|screenshot)\s+(?:này|trên|đó|đã tải)\b",
        ]
    )
    if image_reference:
        return 30.0 if any(str(mime).startswith("image/") for mime in mime_types) else 0.0
    return 0.0
