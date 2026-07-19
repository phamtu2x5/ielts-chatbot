import re
from dataclasses import asdict, dataclass
from typing import Any


QUESTION_RANGE_RE = re.compile(
    r"(?:questions?|question|câu hỏi|câu)\s*(?:từ\s+)?(\d{1,2})(?:\s*(?:-|–|to|đến|tới)\s*(?:questions?|question|câu hỏi|câu)?\s*(\d{1,2}))?",
    re.IGNORECASE,
)

NO_SOLUTION_PATTERNS = [
    re.compile(r"\b(?:không|chưa)\s+(?:tự\s+)?giải(?!\s*thích)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:không|chưa)\s+(?:đưa|điền)\s+(?:ra\s+)?đáp\s+án\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:không|chưa)\s+chọn\s+(?:\S+\s+){0,3}?đáp\s+án\b", re.IGNORECASE),
    re.compile(r"\b(?:không|chưa)\s+ghép\b", re.IGNORECASE),
    re.compile(r"\bgiữ\s+nguyên\s+ô\s+trống\b", re.IGNORECASE),
]

NO_WRITING_PATTERNS = [
    re.compile(r"\b(?:không|chưa)\s+viết\s+(?:bài|đoạn|report|essay)\b", re.IGNORECASE),
    re.compile(r"\b(?:không|chưa)\s+viết\s*(?:[.!?]|$)", re.IGNORECASE),
    re.compile(
        r"\bchỉ\s+(?:trình\s+bày|nêu|giải\s+thích|mô\s+tả)\s+(?:phần\s+)?(?:yêu\s+cầu|đề\s+bài|prompt)\b",
        re.IGNORECASE,
    ),
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

TABLE_MARKERS = ["bảng", "table", "hàng", "cột"]
FLOWCHART_MARKERS = ["flowchart", "flow chart", "flow-chart", "sơ đồ", "node", "hướng nối"]
DIAGRAM_MARKERS = ["diagram", "biểu đồ cấu tạo", "nhãn trống"]
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
]

CALCULATION_MARKERS = ["tăng nhiều nhất", "giảm nhiều nhất", "phép tính", "chênh lệch", "difference"]
COMPARISON_MARKERS = ["so sánh", "compare", "comparison"]
WRITING_GENERATION_MARKERS = [
    "viết bài",
    "viết đoạn",
    "write an essay",
    "write a report",
    "write a paragraph",
    "write an introduction",
    "write a body paragraph",
    "170-190",
    "170–190",
]


@dataclass(frozen=True)
class QueryIntentDecision:
    intent: str
    confidence: float
    reason: str
    allow_solution: bool
    question_ranges: list[tuple[int, int]]
    passage_number: int | None
    visual_target: str | None = None

    def to_debug(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["question_ranges"] = [list(item) for item in self.question_ranges]
        return payload


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


def has_explicit_no_solution_constraint(message: str) -> bool:
    return any(pattern.search(message) for pattern in NO_SOLUTION_PATTERNS)


def has_explicit_no_writing_constraint(message: str) -> bool:
    return any(pattern.search(message) for pattern in NO_WRITING_PATTERNS)


def _looks_like_table_cell_lookup(text: str) -> bool:
    asks_value = any(marker in text for marker in ["bao nhiêu", "giá trị", "tỷ lệ", "số liệu", "value", "figure"])
    has_year_or_number = bool(re.search(r"\b\d{4}\b", text))
    has_row_reference = bool(
        re.search(r"(?:hàng|row|dòng|nước|country)\s+[a-z0-9][\w-]*", text, flags=re.IGNORECASE)
    )
    return asks_value and has_year_or_number and has_row_reference


def looks_like_document_overview(message: str) -> bool:
    lowered = message.lower()
    markers = [
        "nội dung tài liệu",
        "nội dung của tài liệu",
        "tóm tắt tài liệu",
        "tổng quan tài liệu",
        "gồm những passage",
        "gồm các passage",
        "ba passage",
        "cấu trúc reading test",
        "chứa những đề và bài mẫu",
        "những tài liệu nào",
        "các tài liệu đã tải",
        "summary of the document",
        "summarize the document",
    ]
    return any(marker in lowered for marker in markers)


def detect_query_intent_decision(
    message: str,
    probe: dict[str, Any],
    document_grounded: bool | None = None,
) -> QueryIntentDecision:
    lowered = message.lower()
    question_ranges = parse_question_ranges(message)
    passage_number = parse_passage_number(message)
    forbid_solution = has_explicit_no_solution_constraint(message)
    targets_table = _has_any(lowered, TABLE_MARKERS)
    targets_flowchart = _has_any(lowered, FLOWCHART_MARKERS)
    targets_diagram = _has_any(lowered, DIAGRAM_MARKERS)
    asks_show = _has_any(lowered, SHOW_MARKERS)
    asks_translate = any(marker in lowered for marker in ["dịch", "translate", "nghĩa tiếng việt", "nghĩa là gì"])
    asks_explain = any(marker in lowered for marker in ["giải thích", "explain", "phân tích"])
    asks_solve = any(marker in lowered for marker in SOLVE_MARKERS)
    is_grounded = probe.get("has_document_intent") if document_grounded is None else document_grounded
    tabular_coordinates = len(re.findall(r"\b\d{4}\b", lowered)) >= 2 or any(
        marker in lowered
        for marker in ["tỷ lệ", "số liệu", "phần trăm", "percent", "country", "quốc gia"]
    )

    def decision(
        intent: str,
        confidence: float,
        reason: str,
        allow_solution: bool = False,
        visual_target: str | None = None,
    ) -> QueryIntentDecision:
        return QueryIntentDecision(
            intent=intent,
            confidence=confidence,
            reason=reason,
            allow_solution=allow_solution,
            question_ranges=question_ranges,
            passage_number=passage_number,
            visual_target=visual_target,
        )

    if targets_flowchart and (asks_show or forbid_solution):
        return decision("show_flowchart", 0.98, "flowchart target with show/no-solve constraint", visual_target="flowchart")
    if targets_diagram and (asks_show or forbid_solution):
        return decision("show_diagram", 0.98, "diagram target with show/no-solve constraint", visual_target="diagram")
    if _has_any(lowered, CALCULATION_MARKERS) and (
        targets_table or bool(is_grounded and tabular_coordinates)
    ):
        return decision("table_calculation", 0.98, "table target with calculation operation", visual_target="table")
    if _has_any(lowered, COMPARISON_MARKERS) and (
        targets_table or bool(is_grounded and tabular_coordinates)
    ):
        return decision("table_comparison", 0.98, "table target with comparison operation", visual_target="table")
    if _looks_like_table_cell_lookup(lowered):
        return decision("table_cell", 0.98, "single table row/column value lookup", visual_target="table")
    if targets_table and (asks_show or forbid_solution):
        return decision("show_table", 0.98, "table target with show/no-solve constraint", visual_target="table")

    writing_reference = any(
        marker in lowered
        for marker in ["writing", "task 1", "task 2", "ảnh", "hình", "image"]
    )
    asks_writing_prompt = any(marker in lowered for marker in ["yêu cầu", "đề bài", "prompt"])
    forbid_writing = has_explicit_no_writing_constraint(message)
    asks_write_overview = "overview" in lowered and any(marker in lowered for marker in ["viết", "write"])
    if writing_reference and asks_writing_prompt and forbid_writing:
        return decision("show_writing_prompt", 0.99, "writing prompt request with explicit no-writing constraint")
    if writing_reference and not forbid_solution and not forbid_writing and (
        _has_any(lowered, WRITING_GENERATION_MARKERS) or asks_write_overview
    ):
        return decision("writing_generation", 0.98, "explicit writing generation request", allow_solution=True)
    if writing_reference and asks_writing_prompt:
        return decision("show_writing_prompt", 0.95, "writing prompt request without generation")

    if question_ranges:
        if forbid_solution:
            if asks_translate:
                return decision("translate_questions", 0.99, "translation request with no-solution constraint")
            if asks_explain:
                return decision("explain_questions", 0.99, "explanation request with no-solution constraint")
            return decision("show_questions", 0.99, "question range with no-solution constraint")
        if asks_translate:
            return decision("translate_questions", 0.98, "explicit translation request")
        if asks_solve:
            return decision("solve_questions", 0.99, "explicit answer/solve request", allow_solution=True)
        if asks_explain:
            return decision("explain_questions", 0.95, "question explanation request without solve marker")
        return decision("show_questions", 0.9, "question range defaults to show-only")
    if probe.get("is_overview") or looks_like_document_overview(message):
        return decision("document_overview", 0.98, "overview marker or overview probe")
    if is_grounded:
        return decision("semantic_qa", 0.85, "document-grounded semantic question", allow_solution=True)
    return decision("direct", 0.9, "question is independent from uploaded documents")


def detect_query_intent(message: str, probe: dict[str, Any]) -> str:
    return detect_query_intent_decision(message, probe).intent


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

    if intent in {"show_table", "extract_table", "table_cell", "table_calculation", "table_comparison"}:
        table_sources = []
        for source in sources:
            metadata = source.get("metadata", {})
            if metadata.get("unit_type") in {"writing_table", "table", "table_row"}:
                table_sources.append(source)
            elif metadata.get("question_type") == "table_completion":
                table_sources.append(source)
        return table_sources

    if intent in {"show_flowchart", "show_diagram"}:
        flowchart_sources = []
        for source in sources:
            metadata = source.get("metadata", {})
            visual_type = "flowchart" if intent == "show_flowchart" else "diagram"
            question_type = "flowchart_completion" if intent == "show_flowchart" else "diagram_completion"
            if metadata.get("unit_type") == visual_type or metadata.get("question_type") == question_type:
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
