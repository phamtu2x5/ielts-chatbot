import re
from typing import Any


QUESTION_RANGE_RE = re.compile(
    r"(?:questions?|question|câu hỏi|câu)\s*(?:từ\s+)?(\d{1,2})(?:\s*(?:-|–|to|đến|tới)\s*(?:questions?|question|câu hỏi|câu)?\s*(\d{1,2}))?",
    re.IGNORECASE,
)


def parse_question_ranges(message: str) -> list[tuple[int, int]]:
    ranges = []
    for match in QUESTION_RANGE_RE.finditer(message):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start > end:
            start, end = end, start
        ranges.append((start, end))
    return ranges


def detect_query_intent(message: str, probe: dict[str, Any]) -> str:
    lowered = message.lower()
    question_ranges = parse_question_ranges(message)
    if probe.get("is_overview"):
        return "document_overview"
    if question_ranges:
        if any(marker in lowered for marker in ["dịch", "translate", "nghĩa tiếng việt", "nghĩa là gì"]):
            return "translate_questions"
        solve_markers = [
            "trả lời",
            "đáp án",
            "answer key",
            "answer question",
            "answer questions",
            "giải bài",
            "giải câu",
            "làm câu",
            "chọn đáp án",
            "tìm đáp án",
            "true false not given",
            "t/f/ng",
        ]
        if any(marker in lowered for marker in solve_markers):
            return "solve_questions"
        if any(marker in lowered for marker in ["giải thích", "explain", "phân tích"]):
            return "explain_questions"
        return "show_questions"
    return "semantic_qa" if probe.get("has_document_intent") else "direct"


def ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def filter_sources_for_intent(sources: list[dict[str, Any]], message: str, intent: str) -> list[dict[str, Any]]:
    question_ranges = parse_question_ranges(message)
    if not question_ranges or intent not in {"show_questions", "explain_questions", "translate_questions"}:
        return sources

    filtered = []
    for source in sources:
        metadata = source.get("metadata", {})
        question_range = metadata.get("question_range")
        if metadata.get("unit_type") not in {"question_group", "question"}:
            continue
        if not isinstance(question_range, list) or len(question_range) != 2:
            continue
        chunk_start, chunk_end = int(question_range[0]), int(question_range[1])
        if any(ranges_overlap(start, end, chunk_start, chunk_end) for start, end in question_ranges):
            filtered.append(source)
    return filtered or sources


def dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for source in sources:
        key = source.get("chunk_id") or (source.get("source_file"), tuple(source.get("pages") or []), source.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped
