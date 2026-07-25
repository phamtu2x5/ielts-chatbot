from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
from typing import Any

from .config import settings


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
    request_mode: str = "auto"

    def to_debug(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankedDocumentCandidate:
    entry: dict[str, Any]
    score: float
    matched_fields: tuple[str, ...]
    recency_rank: int

    def to_debug(self) -> dict[str, Any]:
        return {
            "document_ids": list(self.entry.get("document_ids") or []),
            "source_file": self.entry.get("source_file", "unknown"),
            "score": round(self.score, 3),
            "matched_fields": list(self.matched_fields),
            "recency_rank": self.recency_rank,
        }


def normalize_reference(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(character for character in normalized if not unicodedata.combining(character))
    return " ".join(re.findall(r"[a-z0-9]+", without_marks))


def _requested_source_modalities(message: str) -> set[str]:
    token_sequence = re.findall(
        r"[\w]+",
        unicodedata.normalize("NFC", message.casefold()),
    )
    tokens = set(token_sequence)
    modalities: set[str] = set()
    mentions_vietnamese_image = any(
        token == "ảnh"
        and (
            index + 1 >= len(token_sequence)
            or token_sequence[index + 1] != "hưởng"
        )
        for index, token in enumerate(token_sequence)
    )
    if mentions_vietnamese_image or tokens.intersection(
        {"image", "photo", "picture", "screenshot"}
    ):
        modalities.add("image")
    if "pdf" in tokens:
        modalities.add("pdf")
    return modalities


def _source_modality_score(message: str, entry: dict[str, Any]) -> float:
    requested = _requested_source_modalities(message)
    if not requested:
        return 0.0
    mime_types = {
        str(value).casefold()
        for value in entry.get("mime_types") or []
    }
    suffix = Path(str(entry.get("source_file", ""))).suffix.casefold()
    available: set[str] = set()
    if any(value.startswith("image/") for value in mime_types) or suffix in {
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }:
        available.add("image")
    if "application/pdf" in mime_types or suffix == ".pdf":
        available.add("pdf")
    return 180.0 if requested.intersection(available) else 0.0


def resolve_document_scope(
    message: str,
    catalog: list[dict[str, Any]],
    requested_document_ids: list[str] | None = None,
    request_mode: str = "auto",
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
    # This layer only validates allowed IDs and exact catalog references.
    # The separate target resolver owns semantic document selection.
    grounded = False
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
            request_mode,
        )

    normalized_query = normalize_reference(message)
    title_weights = _title_token_weights(allowed_entries)
    scored: list[tuple[float, dict[str, Any]]] = []
    for entry in allowed_entries:
        score = _catalog_match_score(message, normalized_query, entry, title_weights)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda item: item[0], reverse=True)

    top_score = scored[0][0] if scored else 0.0
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if (
        scored
        and top_score >= settings.document_scope_min_match_score
        and (
            len(scored) == 1
            or top_score - second_score >= settings.document_scope_match_margin
        )
    ):
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
            request_mode,
        )

    if request_mode == "explicit" and len(allowed) == 1:
        return DocumentScope(
            requested,
            allowed,
            allowed,
            [entry.get("source_file", "unknown") for entry in allowed_entries],
            "explicit_single",
            False,
            False,
            "The client attached one document to the current request.",
            request_mode,
        )

    if request_mode == "explicit" and len(allowed) > 1:
        return DocumentScope(
            requested,
            allowed,
            [],
            [entry.get("source_file", "unknown") for entry in allowed_entries],
            "explicit_multiple",
            False,
            False,
            "The current request attached multiple documents; target resolution is still required.",
            request_mode,
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
            request_mode,
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
        request_mode,
    )

def _filename_match_score(normalized_query: str, source_file: str) -> float:
    stem = normalize_reference(Path(source_file).stem)
    if not stem:
        return 0.0
    if f" {stem} " in f" {normalized_query} ":
        return 240.0
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
        160.0
        + float(len(overlap))
        + len(overlap) / max(1, len(file_terms))
        + longest_phrase * 20.0
    )


def _title_token_weights(catalog: list[dict[str, Any]]) -> dict[str, float]:
    normalized_titles = [
        normalize_reference(str(title))
        for entry in catalog
        for title in entry.get("section_titles") or []
        if normalize_reference(str(title))
    ]
    if not normalized_titles:
        return {}
    frequencies: Counter[str] = Counter()
    for title in normalized_titles:
        frequencies.update(set(title.split()))
    total = len(normalized_titles)
    return {
        token: 1.0 + log((total + 1.0) / (frequency + 1.0))
        for token, frequency in frequencies.items()
    }


def _section_title_match_score(
    normalized_query: str,
    titles: list[str],
    token_weights: dict[str, float] | None = None,
) -> float:
    query_terms = normalized_query.split()
    query_term_set = set(query_terms)
    weights = token_weights or {}
    best_score = 0.0
    for title in titles:
        normalized_title = normalize_reference(str(title))
        title_terms = normalized_title.split()
        if not normalized_title:
            continue
        if normalized_title in normalized_query and (
            len(title_terms) >= 2 or len(normalized_title) >= 5
        ):
            best_score = max(best_score, 140.0 + len(title_terms))
            continue

        matched_terms = query_term_set.intersection(title_terms)
        if len(matched_terms) < 2:
            continue
        title_weight = sum(weights.get(term, 1.0) for term in set(title_terms))
        matched_weight = sum(weights.get(term, 1.0) for term in matched_terms)
        weighted_coverage = matched_weight / max(1.0, title_weight)
        longest_phrase = _longest_contiguous_match(query_terms, title_terms)
        distinctive_matches = sum(
            1 for term in matched_terms if weights.get(term, 1.0) > 1.15
        )
        qualifies = (
            longest_phrase >= 3
            or weighted_coverage >= 0.6
            or (
                longest_phrase >= 2
                and (weighted_coverage >= 0.28 or distinctive_matches >= 2)
            )
        )
        if not qualifies:
            continue
        matching_numbers = sum(
            1 for term in matched_terms if term.isdigit()
        )
        score = (
            35.0
            + weighted_coverage * 70.0
            + longest_phrase * 10.0
            + distinctive_matches * 4.0
            + matching_numbers * 4.0
        )
        best_score = max(best_score, score)
    return best_score


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
    message: str,
    normalized_query: str,
    entry: dict[str, Any],
    title_weights: dict[str, float] | None = None,
) -> float:
    return (
        _source_modality_score(message, entry)
        + _filename_match_score(normalized_query, entry.get("source_file", ""))
        + _section_title_match_score(
            normalized_query,
            entry.get("section_titles") or [],
            title_weights,
        )
    )


def _metadata_token_weights(catalog: list[dict[str, Any]]) -> dict[str, float]:
    frequencies: Counter[str] = Counter()
    for entry in catalog:
        tokens = {
            token
            for key in (
                "document_types",
                "task_types",
                "visual_types",
                "table_columns",
                "target_descriptors",
                "unit_types",
                "mime_types",
            )
            for value in entry.get(key) or []
            for token in normalize_reference(str(value)).split()
        }
        frequencies.update(tokens)
    total = max(1, len(catalog))
    return {
        token: 1.0 + log((total + 1.0) / (frequency + 1.0))
        for token, frequency in frequencies.items()
    }


def _metadata_field_score(
    normalized_query: str,
    values: list[Any],
    token_weights: dict[str, float],
    field_weight: float,
) -> float:
    query_terms = set(normalized_query.split())
    best = 0.0
    for value in values:
        normalized_value = normalize_reference(str(value))
        value_terms = set(normalized_value.split())
        if not value_terms:
            continue
        matched = query_terms.intersection(value_terms)
        if not matched:
            continue
        total_weight = sum(token_weights.get(term, 1.0) for term in value_terms)
        matched_weight = sum(token_weights.get(term, 1.0) for term in matched)
        coverage = matched_weight / max(1.0, total_weight)
        phrase_bonus = 0.35 if normalized_value in normalized_query else 0.0
        best = max(best, field_weight * (coverage + phrase_bonus))
    return best


def rank_document_candidates(
    message: str,
    catalog: list[dict[str, Any]],
    max_candidates: int,
    preferred_document_ids: list[str] | None = None,
) -> list[RankedDocumentCandidate]:
    """Rank target candidates by query relevance, using recency only for ties."""
    if max_candidates <= 0:
        return []
    normalized_query = normalize_reference(message)
    title_weights = _title_token_weights(catalog)
    metadata_weights = _metadata_token_weights(catalog)
    preferred = set(preferred_document_ids or [])
    ranked: list[RankedDocumentCandidate] = []
    field_weights = {
        "document_types": 18.0,
        "task_types": 24.0,
        "visual_types": 18.0,
        "table_columns": 28.0,
        "target_descriptors": 22.0,
        "unit_types": 8.0,
        "mime_types": 8.0,
    }
    for recency_rank, entry in enumerate(catalog):
        matched_fields: list[str] = []
        modality_score = _source_modality_score(message, entry)
        filename_score = _filename_match_score(
            normalized_query,
            entry.get("source_file", ""),
        )
        title_score = _section_title_match_score(
            normalized_query,
            entry.get("section_titles") or [],
            title_weights,
        )
        score = modality_score + filename_score + title_score
        if modality_score:
            matched_fields.append("source_modality")
        if filename_score:
            matched_fields.append("source_file")
        if title_score:
            matched_fields.append("section_titles")
        for key, weight in field_weights.items():
            field_score = _metadata_field_score(
                normalized_query,
                entry.get(key) or [],
                metadata_weights,
                weight,
            )
            if field_score:
                score += field_score
                matched_fields.append(key)
        ranked.append(
            RankedDocumentCandidate(
                entry=entry,
                score=score,
                matched_fields=tuple(matched_fields),
                recency_rank=recency_rank,
            )
        )

    ranked.sort(
        key=lambda candidate: (
            -candidate.score,
            -int(
                bool(
                    preferred.intersection(
                        candidate.entry.get("document_ids") or []
                    )
                )
            ),
            candidate.recency_rank,
        )
    )
    return ranked[:max_candidates]


def order_metadata_values(message: str, values: list[Any]) -> list[str]:
    """Put values most relevant to the current query first, preserving stable ties."""
    normalized_query = normalize_reference(message)
    query_terms = set(normalized_query.split())

    def relevance(value: str) -> tuple[int, int, int]:
        normalized_value = normalize_reference(value)
        value_terms = normalized_value.split()
        return (
            int(bool(normalized_value and normalized_value in normalized_query)),
            _longest_contiguous_match(normalized_query.split(), value_terms),
            len(query_terms.intersection(value_terms)),
        )

    scored = [
        (index, str(value), relevance(str(value)))
        for index, value in enumerate(values)
    ]
    scored.sort(
        key=lambda item: (
            -item[2][0],
            -item[2][1],
            -item[2][2],
            item[0],
        )
    )
    return [value for _, value, _ in scored]
