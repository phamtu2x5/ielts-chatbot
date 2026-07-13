import re
from typing import Any


QUESTION_RANGE_RE = re.compile(
    r"(?:questions?|question|câu hỏi|câu)\s*(?:từ\s+)?(\d{1,2})(?:\s*(?:-|–|to|đến|tới)\s*(?:questions?|question|câu hỏi|câu)?\s*(\d{1,2}))?",
    re.IGNORECASE,
)

NO_SOLUTION_MARKERS = [
    "không giải",
    "chưa giải",
    "không đưa đáp án",
    "chưa đưa đáp án",
    "chưa điền đáp án",
    "không điền đáp án",
    "giữ nguyên ô trống",
    "chỉ hiển thị",
    "chỉ liệt kê",
    "chỉ trích xuất",
    "không tự giải",
]

SHOW_MARKERS = [
    "hiển thị",
    "liệt kê",
    "trích xuất",
    "show",
    "extract",
    "giữ đúng",
    "mô tả",
]

TABLE_MARKERS = ["bảng", "table", "hàng", "cột", "ô trống"]
FLOWCHART_MARKERS = ["flowchart", "flow chart", "flow-chart", "sơ đồ", "node", "hướng nối"]
PASSAGE_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}

SOLVE_MARKERS = [
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


def parse_question_ranges(message: str) -> list[tuple[int, int]]:
    ranges = []
    for match in QUESTION_RANGE_RE.finditer(message):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start > end:
            start, end = end, start
        ranges.append((start, end))
    return ranges


def parse_passage_number(message: str) -> int | None:
    match = re.search(
        r"(?:reading\s+)?passage\s+(\d{1,2}|one|two|three|four|five|six)\b",
        message.lower(),
    )
    if not match:
        return None
    value = match.group(1)
    return int(value) if value.isdigit() else PASSAGE_WORDS.get(value)


def _has_any(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)


def _looks_like_table_cell_lookup(text: str) -> bool:
    asks_value = any(marker in text for marker in ["bao nhiêu", "giá trị", "tỷ lệ", "số liệu", "value", "figure"])
    has_year_or_number = bool(re.search(r"\b\d{4}\b", text))
    has_row_reference = bool(
        re.search(r"(?:hàng|row|dòng|nước)\s+[a-z0-9][\w-]*", text, flags=re.IGNORECASE)
    )
    return asks_value and has_year_or_number and has_row_reference


def detect_query_intent(message: str, probe: dict[str, Any]) -> str:
    lowered = message.lower()
    question_ranges = parse_question_ranges(message)
    forbid_solution = _has_any(lowered, NO_SOLUTION_MARKERS)
    targets_table = _has_any(lowered, TABLE_MARKERS)
    targets_flowchart = _has_any(lowered, FLOWCHART_MARKERS)
    asks_show = _has_any(lowered, SHOW_MARKERS)

    if probe.get("is_overview"):
        return "document_overview"

    if targets_flowchart and (asks_show or forbid_solution):
        return "show_flowchart"
    if _looks_like_table_cell_lookup(lowered):
        return "show_table"
    if targets_table and (asks_show or forbid_solution):
        return "show_table"

    if question_ranges:
        if any(marker in lowered for marker in ["dịch", "translate", "nghĩa tiếng việt", "nghĩa là gì"]):
            return "translate_questions"
        if any(marker in lowered for marker in ["giải thích", "explain", "phân tích"]):
            return "explain_questions"
        if forbid_solution:
            return "show_questions"
        if any(marker in lowered for marker in SOLVE_MARKERS):
            return "solve_questions"
        return "show_questions"
    return "semantic_qa" if probe.get("has_document_intent") else "direct"


def ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def filter_sources_for_intent(sources: list[dict[str, Any]], message: str, intent: str) -> list[dict[str, Any]]:
    question_ranges = parse_question_ranges(message)
    passage_number = parse_passage_number(message)
    if passage_number is not None:
        sources = [
            source
            for source in sources
            if source.get("metadata", {}).get("passage_number") == passage_number
            or source.get("metadata", {}).get("unit_type") == "document_outline"
        ]

    if intent in {"show_table", "extract_table"}:
        table_sources = []
        for source in sources:
            metadata = source.get("metadata", {})
            if metadata.get("unit_type") in {"writing_table", "table", "table_row"}:
                table_sources.append(source)
            elif metadata.get("question_type") == "table_completion":
                table_sources.append(source)
        return table_sources

    if intent == "show_flowchart":
        flowchart_sources = []
        for source in sources:
            metadata = source.get("metadata", {})
            if metadata.get("unit_type") == "flowchart" or metadata.get("question_type") == "flowchart_completion":
                flowchart_sources.append(source)
        return flowchart_sources

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
    return filtered


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
