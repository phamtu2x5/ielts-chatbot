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

PASSAGE_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


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


SOLUTION_INTENTS = {"solve_questions", "writing_generation"}
VISUAL_TARGETS = {
    "show_table": "table",
    "extract_table": "table",
    "table_cell": "table",
    "table_calculation": "table",
    "table_comparison": "table",
    "show_flowchart": "flowchart",
    "show_diagram": "diagram",
}


def semantic_intent_decision(
    message: str,
    intent: str,
    confidence: float,
    reason: str,
) -> QueryIntentDecision:
    """Validate a semantic router decision and apply only explicit safety constraints."""
    forbid_solution = has_explicit_no_solution_constraint(message)
    forbid_writing = has_explicit_no_writing_constraint(message)
    validated_intent = intent
    constraint_reason = ""
    if forbid_solution and intent == "solve_questions":
        validated_intent = "show_questions"
        constraint_reason = " Explicit no-solution constraint enforced."
    elif forbid_writing and intent == "writing_generation":
        validated_intent = "show_writing_prompt"
        constraint_reason = " Explicit no-writing constraint enforced."

    return QueryIntentDecision(
        intent=validated_intent,
        confidence=max(0.0, min(float(confidence), 1.0)),
        reason=reason + constraint_reason,
        allow_solution=validated_intent in SOLUTION_INTENTS,
        question_ranges=parse_question_ranges(message),
        passage_number=parse_passage_number(message),
        visual_target=VISUAL_TARGETS.get(validated_intent),
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


def parse_passage_number(message: str) -> int | None:
    match = re.search(
        r"(?:reading\s+)?passage\s+(\d{1,2}|one|two|three|four|five|six)\b",
        message.lower(),
    )
    if not match:
        return None
    value = match.group(1)
    return int(value) if value.isdigit() else PASSAGE_WORDS.get(value)


def has_explicit_no_solution_constraint(message: str) -> bool:
    return any(pattern.search(message) for pattern in NO_SOLUTION_PATTERNS)


def has_explicit_no_writing_constraint(message: str) -> bool:
    return any(pattern.search(message) for pattern in NO_WRITING_PATTERNS)


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
