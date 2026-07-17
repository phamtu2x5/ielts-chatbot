import re
from dataclasses import dataclass, field
from typing import Any

from .chunking import estimate_tokens
from .config import DocumentPipelineConfig
from .models import DocumentChunk, DocumentElement, ProcessedDocument
from .visual import IELTSQuestionVisualParser
from .writing import WritingCollectionParser, WritingSection


QUESTION_HEADER_RE = re.compile(
    r"Questions?\s+(\d{1,2})(?:\s*(?:-|–|to)\s*(\d{1,2}))?",
    re.IGNORECASE,
)
NUMBERED_QUESTION_RE = re.compile(
    r"(?<!\d)(\d{1,2})\.\s*(.*?)(?=(?<!\d)\d{1,2}\.\s|Questions?\s+\d{1,2}(?:\s*(?:-|–|to)\s*\d{1,2})?|$)",
    re.IGNORECASE | re.DOTALL,
)
PAGE_MARKER_RE = re.compile(r"\n{1,3}\[Page\s+\d+\]", re.IGNORECASE)
FOOTER_RE = re.compile(r"https?://\S+|Page\s+\d+", re.IGNORECASE)
TITLE_WORD_RE = re.compile(r"[A-Z][A-Za-z'’.-]+")
PASSAGE_MARKER_LINE_RE = re.compile(
    r"^[ \t]*(?:reading\s+)?passage\s+(?:\d{1,2}|one|two|three)[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
INSTRUCTION_RE = re.compile(
    r"\b("
    r"questions?\s+\d{1,2}\s*(?:-|–|to)\s*\d{1,2}|"
    r"choose\s+no\s+more\s+than|"
    r"no\s+more\s+than\s+\w+\s+words?|"
    r"complete\s+the\s+(?:table|flow\s*-?\s*chart|flowchart|summary|sentence|notes?)|"
    r"choose\s+the\s+correct\s+letter|"
    r"give\s+two\s+examples|"
    r"write\s+true|"
    r"false\s+if|"
    r"not\s+given|"
    r"answer\s+the\s+questions?|"
    r"match\s+each|"
    r"do\s+the\s+following\s+statements|"
    r"use\s+no\s+more\s+than"
    r")\b",
    re.IGNORECASE,
)
PASSAGE_TITLE_BLACKLIST = {
    "reading passage one",
    "reading passage two",
    "reading passage three",
    "questions",
    "ielts reading test",
}


@dataclass
class IELTSQuestion:
    question_number: int
    text: str
    question_type: str | None
    options: list[str]
    source_element_ids: list[str]
    page_numbers: list[int]
    visual_asset_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_number": self.question_number,
            "text": self.text,
            "question_type": self.question_type,
            "options": self.options,
            "source_element_ids": self.source_element_ids,
            "page_numbers": self.page_numbers,
            "visual_asset_id": self.visual_asset_id,
        }


@dataclass
class IELTSQuestionGroup:
    question_start: int
    question_end: int
    instructions: str
    question_type: str | None
    questions: list[IELTSQuestion]
    page_numbers: list[int]
    source_element_ids: list[str]
    text: str
    visual_element_id: str | None = None
    visual_element: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_start": self.question_start,
            "question_end": self.question_end,
            "instructions": self.instructions,
            "question_type": self.question_type,
            "questions": [question.to_dict() for question in self.questions],
            "page_numbers": self.page_numbers,
            "source_element_ids": self.source_element_ids,
            "visual_element_id": self.visual_element_id,
            "visual_element": self.visual_element,
        }


@dataclass
class IELTSPassage:
    passage_number: int
    title: str | None
    text: str
    question_groups: list[IELTSQuestionGroup]
    page_numbers: list[int]
    source_element_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passage_number": self.passage_number,
            "title": self.title,
            "page_numbers": self.page_numbers,
            "source_element_ids": self.source_element_ids,
            "question_groups": [group.to_dict() for group in self.question_groups],
        }


@dataclass
class IELTSDocument:
    document_id: str
    filename: str
    passages: list[IELTSPassage]
    outline: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    sections: list[WritingSection] = field(default_factory=list)

    def has_structure(self) -> bool:
        return bool(self.sections or (self.passages and any(passage.question_groups for passage in self.passages)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "passages": [passage.to_dict() for passage in self.passages],
            "outline": self.outline,
            "diagnostics": self.diagnostics,
            "sections": [section.to_dict() for section in self.sections],
        }


@dataclass
class _Span:
    start: int
    end: int
    element: DocumentElement


@dataclass
class _ParsedGroup:
    start_offset: int
    end_offset: int
    group: IELTSQuestionGroup


@dataclass
class _TitleCandidate:
    title: str
    start: int
    end: int
    score: tuple[int, int, int, int]


class IELTSStructureParser:
    def __init__(self, config: DocumentPipelineConfig) -> None:
        self.config = config
        self.visual_parser = IELTSQuestionVisualParser(config.connector_direction_min_confidence)
        self.writing_parser = WritingCollectionParser(config)

    def parse(self, document: ProcessedDocument) -> IELTSDocument:
        if document.metadata.get("document_type") == "ielts_writing_task_1":
            structured = self._writing_structure(document)
            document.metadata["ielts_structure"] = structured.to_dict()
            return structured

        writing_sections = self.writing_parser.parse(document)
        if writing_sections:
            writing_tasks = [section for section in writing_sections if section.type == "writing_task_1"]
            sample_answers = [section for section in writing_sections if section.type == "sample_answer"]
            outline = {
                "document_type": "ielts_writing_collection",
                "tasks": [
                    {
                        "task_index": section.task_index,
                        "title": section.title,
                        "visual_type": section.visual_type,
                        "pages": section.pages,
                    }
                    for section in writing_sections
                    if section.type == "writing_task_1"
                ],
            }
            structured = IELTSDocument(
                document_id=document.document_id,
                filename=document.filename,
                passages=[],
                outline=outline,
                diagnostics={
                    "writing_tasks_found": len(writing_tasks),
                    "sample_answers_found": len(sample_answers),
                    "unpaired_tasks": sorted(
                        {section.task_index for section in writing_tasks}
                        - {section.task_index for section in sample_answers}
                    ),
                },
                sections=writing_sections,
            )
            document.metadata["document_type"] = "ielts_writing_collection"
            document.metadata["sections"] = [section.to_dict() for section in writing_sections]
            document.metadata["ielts_structure"] = structured.to_dict()
            return structured

        full_text, spans = self._linearize(document)
        parsed_groups = self._dedupe_groups(self._parse_question_groups(document, full_text, spans))
        passages = self._assign_passages(document, full_text, spans, parsed_groups)
        outline = self._build_outline(document, passages)
        diagnostics = self._diagnostics(passages)
        structured = IELTSDocument(
            document_id=document.document_id,
            filename=document.filename,
            passages=passages,
            outline=outline,
            diagnostics=diagnostics,
        )
        document.metadata["ielts_structure"] = structured.to_dict()
        return structured

    def _linearize(self, document: ProcessedDocument) -> tuple[str, list[_Span]]:
        parts: list[str] = []
        spans: list[_Span] = []
        offset = 0
        current_page = None
        for element in document.elements:
            text = self._structure_text(element)
            if not text or self._is_footer_only(text):
                continue
            if element.page != current_page:
                marker = f"\n\n[Page {element.page}]\n"
                parts.append(marker)
                offset += len(marker)
                current_page = element.page
            if parts and not parts[-1].endswith("\n\n"):
                separator = "\n" if parts[-1].endswith("\n") else "\n\n"
                parts.append(separator)
                offset += len(separator)
            start = offset
            parts.append(text)
            offset += len(text)
            spans.append(_Span(start=start, end=offset, element=element))
            parts.append("\n\n")
            offset += 2
        return "".join(parts), spans

    def _structure_text(self, element: DocumentElement) -> str:
        text = element.raw_text or element.normalized_text
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        compact: list[str] = []
        for line in lines:
            if line or (compact and compact[-1]):
                compact.append(line)
        return "\n".join(compact).strip()

    def _parse_question_groups(
        self,
        document: ProcessedDocument,
        full_text: str,
        spans: list[_Span],
    ) -> list[_ParsedGroup]:
        headers = list(QUESTION_HEADER_RE.finditer(full_text))
        groups: list[_ParsedGroup] = []
        for index, header in enumerate(headers):
            start, end = self._header_range(header)
            next_start = headers[index + 1].start() if index + 1 < len(headers) else len(full_text)
            logical_end = self._logical_group_end(full_text, header.start(), next_start, start, end)
            raw_section = full_text[header.start() : logical_end]
            raw_text = self._clean_section_text(raw_section)
            instructions = self._instructions(raw_text, start)
            question_type = self._infer_question_type(raw_text)
            element_ids, pages = self._span_metadata(spans, header.start(), logical_end)
            questions = self._parse_questions(raw_text, start, end, question_type, element_ids, pages)
            text = self._group_display_text(raw_text, instructions, questions)
            visual = self.visual_parser.parse(
                raw_section,
                start,
                end,
                question_type,
                pages,
                element_ids,
                spatial_pages=self._spatial_pages(document, pages),
            )
            groups.append(
                _ParsedGroup(
                    start_offset=header.start(),
                    end_offset=logical_end,
                    group=IELTSQuestionGroup(
                        question_start=start,
                        question_end=end,
                        instructions=instructions,
                        question_type=question_type,
                        questions=questions,
                        page_numbers=pages,
                        source_element_ids=element_ids,
                        text=text,
                        visual_element_id=visual.element_id if visual else None,
                        visual_element=visual.payload if visual else None,
                    ),
                )
            )
        return groups

    def _spatial_pages(self, document: ProcessedDocument, page_numbers: list[int]) -> list[dict[str, Any]]:
        selected = set(page_numbers)
        return [
            {
                "page": page.page_number,
                "layout_regions": page.metadata.get("layout_regions") or [],
                "ocr_lines": (page.metadata.get("ocr_metadata") or {}).get("lines") or [],
                "connector_regions": page.metadata.get("connector_regions") or [],
            }
            for page in document.pages
            if page.page_number in selected
        ]

    def _dedupe_groups(self, groups: list[_ParsedGroup]) -> list[_ParsedGroup]:
        selected: dict[tuple[int, int], _ParsedGroup] = {}
        for parsed in groups:
            key = (parsed.group.question_start, parsed.group.question_end)
            existing = selected.get(key)
            if existing is None:
                selected[key] = parsed
                continue

            winner = parsed if self._group_quality(parsed) > self._group_quality(existing) else existing
            selected[key] = _ParsedGroup(
                start_offset=min(parsed.start_offset, existing.start_offset),
                end_offset=max(parsed.end_offset, existing.end_offset),
                group=winner.group,
            )
        return sorted(selected.values(), key=lambda item: item.start_offset)

    def _writing_structure(self, document: ProcessedDocument) -> IELTSDocument:
        visual_elements = document.metadata.get("visual_structure", {}).get("visual_elements") or []
        outline = {
            "document_type": document.metadata.get("document_type"),
            "task_type": document.metadata.get("task_type"),
            "filename": document.filename,
            "pages": [page.page_number for page in document.pages],
            "visual_elements": [
                {
                    "type": element.get("type"),
                    "columns": len(element.get("columns") or []),
                    "rows": len(element.get("rows") or []),
                    "confidence": element.get("confidence"),
                }
                for element in visual_elements
            ],
        }
        diagnostics = {
            "passages_found": 0,
            "question_groups_found": 0,
            "questions_found": 0,
            "individual_questions_found": 0,
            "missing_questions": [],
            "duplicate_questions": [],
            "unassigned_questions": [],
            "overlapping_question_groups": [],
            "instruction_as_title": [],
            "suspicious_boundaries": [],
            "visual_elements_found": len(visual_elements),
            "tables_found": sum(1 for element in visual_elements if element.get("type") == "table"),
            "flowcharts_found": sum(1 for element in visual_elements if element.get("type") == "flowchart"),
            "low_confidence_visual_elements": [
                {
                    "type": element.get("type"),
                    "confidence": element.get("confidence"),
                }
                for element in visual_elements
                if float(element.get("confidence") or 0.0) < 0.6
            ],
        }
        return IELTSDocument(
            document_id=document.document_id,
            filename=document.filename,
            passages=[],
            outline=outline,
            diagnostics=diagnostics,
        )

    def _group_quality(self, parsed: _ParsedGroup) -> tuple[int, int, int]:
        group = parsed.group
        layout_bonus = 1 if group.question_type in {"table_completion", "flowchart_completion", "diagram_labeling"} else 0
        return (len(group.questions), layout_bonus, len(group.text))

    def _logical_group_end(self, full_text: str, header_start: int, next_header_start: int, start: int, end: int) -> int:
        section = full_text[header_start:next_header_start]
        end_question_match = None
        for match in re.finditer(rf"(?<!\d){end}\.\s*", section):
            end_question_match = match
        stop_candidates = [next_header_start]
        task_match = re.search(r"\bTask\s+[12]\b|\bWriting\s+Task\b", section, flags=re.IGNORECASE)
        if task_match:
            stop_candidates.append(header_start + task_match.start())
        if end_question_match:
            after_last_question = header_start + end_question_match.start()
            title_offset = self._passage_title_offset(
                section,
                start_at=end_question_match.end(),
                require_line_boundary=True,
            )
            if title_offset is not None:
                stop_candidates.append(header_start + title_offset)
            page_match = PAGE_MARKER_RE.search(full_text, after_last_question, next_header_start)
            if page_match:
                stop_candidates.append(page_match.start())
                title_match = re.search(
                    r"\n\s*([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,5})\s*"
                    r"(?=\n\s*\[Page\s+\d+\])",
                    full_text[after_last_question : page_match.end()],
                )
                if title_match:
                    stop_candidates.append(after_last_question + title_match.start())
            if self._infer_question_type(section) == "multiple_choice":
                multiple_choice_end = self._multiple_choice_group_end(section, end_question_match.start())
                if multiple_choice_end:
                    stop_candidates.append(header_start + multiple_choice_end)
        else:
            title_offset = self._passage_title_offset(section)
            if title_offset is not None:
                question_type = self._infer_question_type(section)
                if question_type not in {"table_completion", "flowchart_completion", "diagram_labeling"} or self._visual_tail_is_passage_body(section[title_offset:]):
                    stop_candidates.append(header_start + title_offset)
        return min(stop_candidates)

    def _passage_title_offset(
        self,
        text: str,
        start_at: int = 20,
        require_line_boundary: bool = False,
    ) -> int | None:
        window = text[:2000]
        candidate = self._best_title_candidate(
            window,
            min_start=max(20, start_at),
            require_line_boundary=require_line_boundary,
        )
        marker_offsets = [
            match.start()
            for match in PASSAGE_MARKER_LINE_RE.finditer(window)
            if match.start() >= max(20, start_at)
        ]
        offsets = marker_offsets + ([candidate.start] if candidate else [])
        return min(offsets) if offsets else None

    def _multiple_choice_group_end(self, section: str, last_question_start: int) -> int | None:
        tail = section[last_question_start:]
        option_d = re.search(r"\bD\s+", tail)
        if not option_d:
            return None
        sentence_end = re.search(r"[.!?](?:\s|$)", tail[option_d.end() :])
        if not sentence_end:
            return None
        return last_question_start + option_d.end() + sentence_end.end()

    def _parse_questions(
        self,
        text: str,
        start: int,
        end: int,
        question_type: str | None,
        element_ids: list[str],
        pages: list[int],
    ) -> list[IELTSQuestion]:
        questions: list[IELTSQuestion] = []
        for match in NUMBERED_QUESTION_RE.finditer(text):
            number = int(match.group(1))
            if not start <= number <= end:
                continue
            body = self._trim_question_body(self._clean_section_text(match.group(2)), question_type)
            options = re.findall(r"\b([A-D])\s+([^A-D]+?)(?=\s+[A-D]\s+|$)", body)
            questions.append(
                IELTSQuestion(
                    question_number=number,
                    text=body,
                    question_type=question_type,
                    options=[f"{label}. {option.strip()}" for label, option in options],
                    source_element_ids=element_ids,
                    page_numbers=pages,
                )
            )
        return questions

    def _trim_question_body(self, body: str, question_type: str | None) -> str:
        body = self._strip_noise_tail(body)
        if question_type == "multiple_choice":
            option_d = re.search(r"\bD\s+", body)
            if option_d:
                sentence_end = re.search(r"[.!?](?:\s|$)", body[option_d.end() :])
                if sentence_end:
                    return body[: option_d.end() + sentence_end.end()].strip()
        return body

    def _group_display_text(
        self,
        raw_text: str,
        instructions: str,
        questions: list[IELTSQuestion],
    ) -> str:
        if questions:
            question_text = " ".join(
                f"{question.question_number}. {question.text}".strip() for question in questions
            )
            return self._clean_section_text(f"{instructions} {question_text}")
        return self._strip_noise_tail(raw_text)

    def _assign_passages(
        self,
        document: ProcessedDocument,
        full_text: str,
        spans: list[_Span],
        parsed_groups: list[_ParsedGroup],
    ) -> list[IELTSPassage]:
        if not parsed_groups:
            return []

        passages: list[IELTSPassage] = []
        state = "SEARCHING_PASSAGE"
        current_text = full_text[: parsed_groups[0].start_offset].strip()
        current_start = 0
        current_groups: list[IELTSQuestionGroup] = []
        previous_group_end = parsed_groups[0].start_offset

        for parsed in parsed_groups:
            gap = full_text[previous_group_end : parsed.start_offset].strip()
            gap_kind = self._classify_inter_group_gap(gap)
            if current_groups and gap_kind == "new_passage":
                passages.append(
                    self._make_passage(len(passages) + 1, current_text, current_groups, spans, current_start, previous_group_end)
                )
                current_text = gap
                current_start = previous_group_end
                current_groups = []
                state = "READING_PASSAGE_BODY"
            elif gap and gap_kind == "passage_body":
                current_text = f"{current_text}\n\n{gap}".strip()
                state = "READING_PASSAGE_BODY"
            elif gap_kind == "instruction":
                state = "READING_INSTRUCTIONS"

            state = "READING_QUESTIONS"
            current_groups.append(parsed.group)
            previous_group_end = parsed.end_offset

        state = "WAITING_NEXT_PASSAGE"
        passages.append(
            self._make_passage(len(passages) + 1, current_text, current_groups, spans, current_start, previous_group_end)
        )
        _ = state
        return [passage for passage in passages if passage.text or passage.question_groups]

    def _make_passage(
        self,
        passage_number: int,
        text: str,
        groups: list[IELTSQuestionGroup],
        spans: list[_Span],
        start: int,
        end: int,
    ) -> IELTSPassage:
        title = self._infer_title(text)
        cleaned = self._clean_passage_text(text)
        cleaned = self._strip_leading_title(cleaned, title)
        element_ids, pages = self._span_metadata(spans, start, end)
        group_pages = {page for group in groups for page in group.page_numbers}
        return IELTSPassage(
            passage_number=passage_number,
            title=title,
            text=cleaned,
            question_groups=groups,
            page_numbers=sorted(set(pages) | group_pages),
            source_element_ids=element_ids,
        )

    def _build_outline(self, document: ProcessedDocument, passages: list[IELTSPassage]) -> dict[str, Any]:
        return {
            "document_type": "IELTS Reading" if passages else "unknown",
            "filename": document.filename,
            "passages": [
                {
                    "number": passage.passage_number,
                    "title": passage.title,
                    "pages": passage.page_numbers,
                    "question_groups": [
                        {
                            "range": [group.question_start, group.question_end],
                            "type": group.question_type,
                            "pages": group.page_numbers,
                            "visual_type": group.visual_element.get("type") if group.visual_element else None,
                            "visual_confidence": group.visual_element.get("confidence") if group.visual_element else None,
                        }
                        for group in passage.question_groups
                    ],
                }
                for passage in passages
            ],
        }

    def _diagnostics(self, passages: list[IELTSPassage]) -> dict[str, Any]:
        parsed_questions = [
            question.question_number
            for passage in passages
            for group in passage.question_groups
            for question in group.questions
        ]
        covered_questions = {
            number
            for passage in passages
            for group in passage.question_groups
            for number in range(group.question_start, group.question_end + 1)
        }
        expected = (
            set(range(min(covered_questions), max(covered_questions) + 1))
            if covered_questions
            else set()
        )
        duplicates = sorted(
            {number for number in parsed_questions if parsed_questions.count(number) > 1}
        )
        group_ranges = [
            (group.question_start, group.question_end, passage.passage_number)
            for passage in passages
            for group in passage.question_groups
        ]
        overlapping_groups = []
        for index, (start, end, passage_number) in enumerate(group_ranges):
            for other_start, other_end, other_passage_number in group_ranges[index + 1 :]:
                if max(start, other_start) <= min(end, other_end):
                    overlapping_groups.append(
                        {
                            "range": [start, end],
                            "passage_number": passage_number,
                            "overlaps": [other_start, other_end],
                            "overlap_passage_number": other_passage_number,
                        }
                    )
        instruction_as_title = [
            {
                "passage_number": passage.passage_number,
                "title": passage.title,
            }
            for passage in passages
            if passage.title and self._is_instruction_like_text(passage.title)
        ]
        suspicious_boundaries = []
        for passage in passages:
            if not passage.title and len(passage.question_groups) > 1:
                suspicious_boundaries.append(
                    {
                        "passage_number": passage.passage_number,
                        "reason": "missing_title_with_multiple_question_groups",
                    }
                )
        visual_elements = [
            group.visual_element
            for passage in passages
            for group in passage.question_groups
            if group.visual_element
        ]
        return {
            "passages_found": len(passages),
            "question_groups_found": sum(len(passage.question_groups) for passage in passages),
            "questions_found": len(covered_questions),
            "individual_questions_found": len(set(parsed_questions)),
            "missing_questions": sorted(expected - covered_questions),
            "duplicate_questions": duplicates,
            "unassigned_questions": [],
            "overlapping_question_groups": overlapping_groups,
            "instruction_as_title": instruction_as_title,
            "suspicious_boundaries": suspicious_boundaries,
            "visual_elements_found": len(visual_elements),
            "tables_found": sum(1 for element in visual_elements if element.get("type") == "table"),
            "flowcharts_found": sum(1 for element in visual_elements if element.get("type") == "flowchart"),
            "low_confidence_visual_elements": [
                {
                    "type": element.get("type"),
                    "question_range": element.get("question_range"),
                    "confidence": element.get("confidence"),
                }
                for element in visual_elements
                if float(element.get("confidence") or 0.0) < 0.6
            ],
        }

    def _span_metadata(self, spans: list[_Span], start: int, end: int) -> tuple[list[str], list[int]]:
        elements = [span.element for span in spans if span.end > start and span.start < end]
        return [element.element_id for element in elements], sorted({element.page for element in elements})

    def _instructions(self, text: str, start: int) -> str:
        parts = re.split(rf"(?<!\d){start}\.\s*", text, maxsplit=1)
        return self._clean_section_text(parts[0]) if parts else ""

    def _infer_question_type(self, text: str) -> str | None:
        lowered = text.lower()
        if "true" in lowered and "false" in lowered and "not given" in lowered:
            return "true_false_not_given"
        if "complete the table" in lowered:
            return "table_completion"
        if "flow chart" in lowered or "flow-chart" in lowered or "flowchart" in lowered:
            return "flowchart_completion"
        if "label the diagram" in lowered or "label the figure" in lowered:
            return "diagram_labeling"
        if "choose the correct letter" in lowered:
            return "multiple_choice"
        if "give two examples" in lowered:
            return "short_answer_examples"
        if "choose no more than" in lowered:
            return "short_answer"
        if "match" in lowered:
            return "matching"
        return None

    def _classify_inter_group_gap(self, text: str) -> str:
        cleaned = self._clean_passage_text(text)
        if not cleaned:
            return "empty"
        if self._is_instruction_like_text(cleaned):
            return "instruction"
        if self._is_passage_boundary_candidate(text):
            return "new_passage"
        return "passage_body"

    def _looks_like_new_passage(self, text: str) -> bool:
        cleaned = self._clean_passage_text(text)
        return self._is_passage_boundary_candidate(cleaned)

    def _is_passage_boundary_candidate(self, text: str) -> bool:
        cleaned = self._clean_passage_text(text)
        if not cleaned or self._is_instruction_like_text(cleaned):
            return False
        if PASSAGE_MARKER_LINE_RE.search(text):
            return True
        title = self._infer_title(text)
        if title and not self._is_instruction_like_text(title):
            return True
        return estimate_tokens(cleaned) >= 120

    def _infer_title(self, text: str) -> str | None:
        marker = PASSAGE_MARKER_LINE_RE.search(text)
        if marker:
            tail = text[marker.end() :]
            candidate = self._best_title_candidate(tail[:2000])
            first_content = self._first_content_line_offset(tail)
            if candidate and first_content is not None and candidate.start == first_content:
                return candidate.title
            return None
        candidate = self._best_title_candidate(text[:2000])
        return candidate.title if candidate else None

    def _best_title_candidate(
        self,
        text: str,
        min_start: int = 0,
        require_line_boundary: bool = False,
    ) -> _TitleCandidate | None:
        candidates: list[_TitleCandidate] = []
        substantive_lines = [
            line
            for line in text.splitlines()
            if line.strip() and not self._is_footer_only(line)
        ]
        for line_match in re.finditer(r"(?m)^[^\n]*\S[^\n]*$", text):
            raw_line = line_match.group(0)
            if re.search(r"https?://", raw_line, flags=re.IGNORECASE):
                continue
            leading = len(raw_line) - len(raw_line.lstrip())
            start = line_match.start() + leading
            if start < min_start:
                continue
            title = self._clean_title_line(raw_line)
            if not title or PASSAGE_MARKER_LINE_RE.fullmatch(title):
                continue
            block_boundary = self._has_block_boundary_before(text, start)
            if not block_boundary and len(substantive_lines) > 3:
                continue
            if not self._is_title_line(title, block_boundary=block_boundary):
                continue
            if self._is_list_sequence_line(raw_line, text[line_match.end() :]):
                continue
            if require_line_boundary and not self._has_line_boundary_before(text, start):
                continue
            after_title = text[line_match.end() : line_match.end() + 320]
            body_score = self._body_likeness(after_title)
            if body_score <= 0 or len(re.findall(r"[A-Za-z]+", after_title)) < 8:
                continue
            title_len = len(title.split())
            candidates.append(
                _TitleCandidate(
                    title=title,
                    start=start,
                    end=line_match.end(),
                    score=(
                        1 if block_boundary else 0,
                        -start,
                        body_score,
                        -abs(title_len - 4),
                    ),
                )
            )

            if block_boundary:
                inline = self._inline_title_candidate(raw_line, start, scan=False)
                if inline:
                    candidates.append(inline)

        if not PASSAGE_MARKER_LINE_RE.search(text) and len(substantive_lines) <= 3:
            inline = self._inline_title_candidate(text[min_start:], min_start, scan=True)
            if inline:
                candidates.append(inline)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.score)

    def _inline_title_candidate(self, text: str, base_offset: int, scan: bool) -> _TitleCandidate | None:
        words = list(TITLE_WORD_RE.finditer(text))
        if not words:
            return None
        candidates: list[_TitleCandidate] = []
        for index, first_word in enumerate(words):
            if not scan and first_word.start() != len(text) - len(text.lstrip()):
                break
            if scan:
                prefix = text[: first_word.start()]
                if (
                    prefix.strip()
                    and not re.search(r"[.!?]\s*$", prefix)
                    and not self._has_block_boundary_before(text, first_word.start())
                ):
                    continue
            for end_index in range(index + 1, min(index + 6, len(words))):
                sequence = words[index : end_index + 1]
                if any(sequence[pos].end() + 1 < sequence[pos + 1].start() for pos in range(len(sequence) - 1)):
                    break
                title_end = sequence[-1].end()
                if title_end < len(text) and text[title_end] in "!?":
                    title_end += 1
                title = text[first_word.start() : title_end].strip()
                if not self._is_plausible_title(title):
                    continue
                after_title = text[title_end : title_end + 240]
                body_score = self._body_likeness(after_title)
                if body_score <= 0:
                    continue
                start = base_offset + first_word.start()
                candidates.append(
                    _TitleCandidate(
                        title=title,
                        start=start,
                        end=base_offset + title_end,
                        score=(0, body_score, -start, -abs(len(title.split()) - 4)),
                    )
                )
        return max(candidates, key=lambda item: item.score) if candidates else None

    def _clean_title_line(self, line: str) -> str:
        line = FOOTER_RE.sub(" ", line)
        return re.sub(r"\s+", " ", line).strip()

    def _first_content_line_offset(self, text: str) -> int | None:
        for match in re.finditer(r"(?m)^[^\n]*\S[^\n]*$", text):
            raw_line = match.group(0)
            if re.search(r"https?://", raw_line, flags=re.IGNORECASE):
                continue
            line = self._clean_title_line(raw_line)
            if not line or re.fullmatch(r"IELTS\s+READING\s+TEST\s+\d+", line, flags=re.IGNORECASE):
                continue
            return match.start() + len(raw_line) - len(raw_line.lstrip())
        return None

    def _is_list_sequence_line(self, line: str, following_text: str) -> bool:
        label = re.match(r"^\s*([A-H]|[ivxlcdm]+)[.)]?\s+", line, flags=re.IGNORECASE)
        if not label:
            return False
        next_line = next((item.strip() for item in following_text.splitlines() if item.strip()), "")
        return bool(re.match(r"^(?:[A-H]|[ivxlcdm]+)[.)]?\s+", next_line, flags=re.IGNORECASE))

    def _is_title_line(self, title: str, block_boundary: bool) -> bool:
        if len(title) > 160 or not 1 <= len(title.split()) <= 12:
            return False
        if len(re.findall(r"[A-Za-z]", title)) < 2:
            return False
        if "|" in title:
            return False
        if not block_boundary and re.match(r"^(?:[A-H]|[ivxlcdm]+)[.)]?\s+", title, flags=re.IGNORECASE):
            return False
        normalized = title.lower().strip(" {}[]()")
        if re.fullmatch(r"ielts\s+reading\s+test\s+\d+", normalized):
            return False
        if normalized in {"list of headings", "notes", "list of people", "list of statements"}:
            return False
        return self._is_plausible_title(title)

    def _has_block_boundary_before(self, text: str, offset: int) -> bool:
        prefix = text[:offset]
        return not prefix.strip() or bool(re.search(r"\n[ \t]*\n[ \t]*$", prefix))

    def _visual_tail_is_passage_body(self, text: str) -> bool:
        snippet = self._clean_section_text(text[:240])
        word_count = len(re.findall(r"[A-Za-z]+", snippet))
        return word_count >= 16 and bool(re.search(r"[.!?]", snippet))

    def _has_line_boundary_before(self, text: str, offset: int) -> bool:
        prefix = text[max(0, offset - 8) : offset]
        return "\n" in prefix

    def _body_likeness(self, text: str) -> int:
        snippet = text.strip()
        if not snippet:
            return 0
        first_words = snippet.split()[:3]
        score = 0
        if re.match(r"^(?:In|The|A|An|This|These|Those|There|It|They|On|For|With|By|From|After|Before)\b", snippet):
            score += 2
        if len(first_words) >= 2 and re.match(
            r"^[A-Z][A-Za-z'’.-]+$",
            first_words[0],
        ) and re.match(
            r"^(?:is|are|was|were|has|have|had|can|could|will|would|should|may|might|must|seems?|looks?|becomes?|became|contains?|includes?|provides?|uses?)\b",
            first_words[1],
            flags=re.IGNORECASE,
        ):
            score += 2
        letters = re.findall(r"[A-Za-z]", snippet[:160])
        lowercase = re.findall(r"[a-z]", snippet[:160])
        if letters and len(lowercase) / len(letters) >= 0.45:
            score += 1
        if re.search(r"[.!?]", snippet[:180]):
            score += 1
        return score

    def _clean_passage_text(self, text: str) -> str:
        text = re.sub(r"\[Page\s+\d+\]", " ", text)
        text = FOOTER_RE.sub(" ", text)
        text = self._trim_to_passage_title(text)
        text = self._remove_question_sections(text)
        text = PASSAGE_MARKER_LINE_RE.sub(" ", text)
        text = re.sub(r"(?im)^\s*IELTS\s+READING\s+TEST\s+\d+\s*$", " ", text)
        return self._clean_section_text(text)

    def _trim_to_passage_title(self, text: str) -> str:
        if not re.search(r"Questions?\s+\d{1,2}\s*(?:-|–|to)\s*\d{1,2}", text, flags=re.IGNORECASE):
            return text
        candidate = self._best_title_candidate(text[:2000])
        if candidate:
            return text[candidate.start :]
        return text

    def _remove_question_sections(self, text: str) -> str:
        headers = list(QUESTION_HEADER_RE.finditer(text))
        if not headers:
            return text

        cleaned_parts: list[str] = []
        cursor = 0
        for index, header in enumerate(headers):
            cleaned_parts.append(text[cursor : header.start()])
            next_header_start = headers[index + 1].start() if index + 1 < len(headers) else len(text)
            start, end = self._header_range(header)
            logical_end = self._logical_group_end(text, header.start(), next_header_start, start, end)
            cursor = max(cursor, logical_end)
        cleaned_parts.append(text[cursor:])
        return " ".join(part for part in cleaned_parts if part.strip())

    def _is_plausible_title(self, title: str) -> bool:
        normalized = re.sub(r"\s+", " ", title).strip().lower().rstrip(".?!")
        if normalized in PASSAGE_TITLE_BLACKLIST:
            return False
        if normalized in {"no", "more", "than", "one", "two", "three", "four", "word", "words"}:
            return False
        if (
            "ielts" in normalized
            or "question" in normalized
            or "passage" in normalized
            or "write true" in normalized
            or "no more" in normalized
            or "no more than" in normalized
            or "more than" in normalized
            or "than " in normalized
            or normalized.endswith("words")
            or "complete the" in normalized
            or normalized.startswith("choose")
            or "from the passage" in normalized
        ):
            return False
        if self._is_instruction_like_text(title):
            return False
        return 1 <= len(title.split()) <= 6

    def _is_instruction_like_text(self, text: str) -> bool:
        normalized = self._clean_section_text(text).lower()
        if not normalized:
            return False
        if INSTRUCTION_RE.search(normalized):
            return True
        first_words = " ".join(normalized.split()[:5])
        return first_words in {
            "choose no more",
            "choose no more than",
            "complete the table",
            "complete the flow chart",
            "complete the flow-chart",
            "choose the correct letter",
            "give two examples",
        }

    def _strip_leading_title(self, text: str, title: str | None) -> str:
        if not title:
            return text
        match = re.search(re.escape(title), text[:140])
        if match:
            return text[match.end() :].strip()
        return text

    def _clean_section_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _strip_noise_tail(self, text: str) -> str:
        return re.split(r"\bTask\s+[12]\b|\bWriting\s+Task\b", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    def _is_footer_only(self, text: str) -> bool:
        without_footer = FOOTER_RE.sub("", text).strip()
        return not without_footer

    def _header_range(self, header: re.Match[str]) -> tuple[int, int]:
        start = int(header.group(1))
        end = int(header.group(2) or start)
        return (end, start) if start > end else (start, end)


class StructuredChunker:
    def __init__(self, config: DocumentPipelineConfig) -> None:
        self.config = config

    def chunk(self, document: ProcessedDocument, structured: IELTSDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        if not structured.has_structure():
            return chunks

        if structured.sections:
            return self._writing_chunks(document, structured.sections)

        chunks.append(self._outline_chunk(document, structured, len(chunks)))
        for passage in structured.passages:
            if passage.text:
                chunks.extend(self._passage_chunks(document, passage, len(chunks)))
            for group in passage.question_groups:
                chunks.append(self._question_group_chunk(document, passage, group, len(chunks)))
                if group.visual_element:
                    chunks.append(self._visual_chunk(document, passage, group, len(chunks)))
                for question in group.questions:
                    chunks.append(self._question_chunk(document, passage, group, question, len(chunks)))

        return chunks

    def _writing_chunks(
        self,
        document: ProcessedDocument,
        sections: list[WritingSection],
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for section in sections:
            parts = self._split_text(section.text, self.config.chunk_max_tokens)
            for part_index, part in enumerate(parts, 1):
                unit_type = "writing_task" if section.type == "writing_task_1" else "sample_answer"
                title = section.title or f"Writing Task {section.task_index}"
                retrieval = (
                    f"IELTS Writing Task 1. Task {section.task_index}. "
                    f"Section: {unit_type}. Visual type: {section.visual_type or 'unknown'}. "
                    f"Title: {title}.\n\n{part}"
                )
                chunks.append(
                    self._make_chunk(
                        document=document,
                        index=len(chunks),
                        chunk_id=(
                            f"{document.document_id}-writing-{section.task_index}-{unit_type}-{part_index}"
                        ),
                        text=part,
                        retrieval_text=retrieval,
                        pages=section.pages,
                        element_ids=section.element_ids,
                        heading_path=[title, unit_type],
                        min_confidence=section.confidence,
                        metadata={
                            "unit_type": unit_type,
                            "chunk_reason": unit_type,
                            "parent_id": f"writing-task-{section.task_index}",
                            "task_index": section.task_index,
                            "task_title": section.title,
                            "visual_type": section.visual_type,
                            "section_id": section.section_id,
                        },
                    )
                )
        return chunks

    def _outline_chunk(self, document: ProcessedDocument, structured: IELTSDocument, index: int) -> DocumentChunk:
        lines = [f"Document: {document.filename}", "IELTS Reading document outline."]
        for passage in structured.passages:
            title = f": {passage.title}" if passage.title else ""
            groups = ", ".join(
                f"Questions {group.question_start}-{group.question_end} ({group.question_type or 'unknown'})"
                for group in passage.question_groups
            )
            lines.append(f"Passage {passage.passage_number}{title}. Pages {passage.page_numbers}. {groups}.")
        text = "\n".join(lines)
        return self._make_chunk(
            document=document,
            index=index,
            chunk_id=f"{document.document_id}-outline",
            text=text,
            retrieval_text=text,
            pages=sorted({page for passage in structured.passages for page in passage.page_numbers}),
            element_ids=[],
            heading_path=["Document outline"],
            min_confidence=1.0,
            metadata={
                "unit_type": "document_outline",
                "chunk_reason": "document_outline",
                "outline": structured.outline,
                "structure_diagnostics": structured.diagnostics,
            },
        )

    def _passage_chunks(self, document: ProcessedDocument, passage: IELTSPassage, start_index: int) -> list[DocumentChunk]:
        chunks = []
        parts = self._split_text(passage.text, self.config.chunk_max_tokens)
        for offset, part in enumerate(parts):
            title = f": {passage.title}" if passage.title else ""
            display = f"Passage {passage.passage_number}{title}\n\n{part}"
            retrieval = (
                f"IELTS Reading. Passage {passage.passage_number}{title}. "
                f"Question groups: {self._group_ranges(passage)}.\n\n{part}"
            )
            chunks.append(
                self._make_chunk(
                    document=document,
                    index=start_index + offset,
                    chunk_id=f"{document.document_id}-passage-{passage.passage_number}-{offset + 1}",
                    text=display,
                    retrieval_text=retrieval,
                    pages=passage.page_numbers,
                    element_ids=passage.source_element_ids,
                    heading_path=[f"Passage {passage.passage_number}"],
                    min_confidence=self._min_confidence(document, passage.source_element_ids),
                    metadata={
                        "unit_type": "passage",
                        "chunk_reason": "passage_paragraph",
                        "passage_number": passage.passage_number,
                        "passage_title": passage.title,
                        "parent_id": f"passage-{passage.passage_number}",
                    },
                )
            )
        return chunks

    def _question_group_chunk(
        self,
        document: ProcessedDocument,
        passage: IELTSPassage,
        group: IELTSQuestionGroup,
        index: int,
    ) -> DocumentChunk:
        display = group.text
        retrieval = (
            f"IELTS Reading. Passage {passage.passage_number}. "
            f"Question Group {group.question_start}-{group.question_end}. "
            f"Question Type: {group.question_type or 'unknown'}.\n\n{display}"
        )
        return self._make_chunk(
            document=document,
            index=index,
            chunk_id=f"{document.document_id}-questions-{group.question_start}-{group.question_end}",
            text=display,
            retrieval_text=retrieval,
            pages=group.page_numbers,
            element_ids=group.source_element_ids,
            heading_path=[f"Passage {passage.passage_number}", f"Questions {group.question_start}-{group.question_end}"],
            min_confidence=self._min_confidence(document, group.source_element_ids),
            metadata={
                "unit_type": "question_group",
                "chunk_reason": "question_group",
                "parent_id": f"passage-{passage.passage_number}",
                "passage_number": passage.passage_number,
                "passage_title": passage.title,
                "question_range": [group.question_start, group.question_end],
                "question_start": group.question_start,
                "question_end": group.question_end,
                "question_type": group.question_type,
                "instructions": group.instructions,
            },
        )

    def _question_chunk(
        self,
        document: ProcessedDocument,
        passage: IELTSPassage,
        group: IELTSQuestionGroup,
        question: IELTSQuestion,
        index: int,
    ) -> DocumentChunk:
        display = f"{question.question_number}. {question.text}"
        retrieval = (
            f"IELTS Reading. Passage {passage.passage_number}. "
            f"Question {question.question_number}. "
            f"Question Group {group.question_start}-{group.question_end}. "
            f"Question Type: {question.question_type or 'unknown'}.\n\n{display}"
        )
        return self._make_chunk(
            document=document,
            index=index,
            chunk_id=f"{document.document_id}-question-{question.question_number}",
            text=display,
            retrieval_text=retrieval,
            pages=question.page_numbers,
            element_ids=question.source_element_ids,
            heading_path=[
                f"Passage {passage.passage_number}",
                f"Questions {group.question_start}-{group.question_end}",
            ],
            min_confidence=self._min_confidence(document, question.source_element_ids),
            metadata={
                "unit_type": "question",
                "chunk_reason": "individual_question",
                "parent_id": f"questions-{group.question_start}-{group.question_end}",
                "passage_number": passage.passage_number,
                "passage_title": passage.title,
                "question_range": [question.question_number, question.question_number],
                "question_start": question.question_number,
                "question_end": question.question_number,
                "question_type": question.question_type,
            },
        )

    def _visual_chunk(
        self,
        document: ProcessedDocument,
        passage: IELTSPassage,
        group: IELTSQuestionGroup,
        index: int,
    ) -> DocumentChunk:
        visual = group.visual_element or {}
        visual_type = str(visual.get("type") or "visual")
        display = self._visual_display_text(visual)
        retrieval = (
            f"IELTS Reading visual structure. Passage {passage.passage_number}. "
            f"Questions {group.question_start}-{group.question_end}. "
            f"Question Type: {group.question_type or 'unknown'}. Visual Type: {visual_type}.\n\n{display}"
        )
        return self._make_chunk(
            document=document,
            index=index,
            chunk_id=f"{document.document_id}-{visual_type}-{group.question_start}-{group.question_end}",
            text=display,
            retrieval_text=retrieval,
            pages=group.page_numbers,
            element_ids=group.source_element_ids,
            heading_path=[
                f"Passage {passage.passage_number}",
                f"Questions {group.question_start}-{group.question_end}",
                visual_type,
            ],
            min_confidence=float(visual.get("confidence") or self._min_confidence(document, group.source_element_ids)),
            metadata={
                "unit_type": visual_type,
                "chunk_reason": visual_type,
                "parent_id": f"questions-{group.question_start}-{group.question_end}",
                "passage_number": passage.passage_number,
                "passage_title": passage.title,
                "question_range": [group.question_start, group.question_end],
                "question_start": group.question_start,
                "question_end": group.question_end,
                "question_type": group.question_type,
                visual_type: visual,
            },
        )

    def _visual_display_text(self, visual: dict[str, Any]) -> str:
        visual_type = visual.get("type")
        if visual_type == "table":
            columns = visual.get("columns") or []
            rows = visual.get("rows") or []
            if columns and rows:
                lines = [
                    "| " + " | ".join(str(column) for column in columns) + " |",
                    "| " + " | ".join("---" for _ in columns) + " |",
                ]
                for row in rows:
                    lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
                return "\n".join(lines)
            return self._visual_fallback_text("Table", visual)
        if visual_type == "flowchart":
            nodes = visual.get("nodes") or []
            edges = visual.get("edges") or []
            if nodes:
                lines = [
                    f"Flowchart Questions {visual['question_range'][0]}-{visual['question_range'][1]}"
                ]
                for node in nodes:
                    label = node.get("text", "")
                    numbers = node.get("question_numbers") or (
                        [node["question_number"]] if node.get("question_number") else []
                    )
                    if numbers:
                        label = f"{label} [blanks: {', '.join(str(number) for number in numbers)}]".strip()
                    lines.append(f"- {node['id']}: {label}")
                for edge in edges:
                    lines.append(f"- edge: {edge['from']} -> {edge['to']}")
                return "\n".join(lines)
            return self._visual_fallback_text("Flowchart", visual)
        if visual_type == "diagram":
            labels = visual.get("labels") or []
            if labels:
                lines = [f"Diagram Questions {visual['question_range'][0]}-{visual['question_range'][1]}"]
                for label in labels:
                    lines.append(f"- {label['id']}: {label.get('text', '')}")
                lines.append(f"- detected connectors: {len(visual.get('connectors') or [])}")
                return "\n".join(lines)
            return self._visual_fallback_text("Diagram", visual)
        return str(visual)

    def _visual_fallback_text(self, label: str, visual: dict[str, Any]) -> str:
        start, end = visual.get("question_range") or ["?", "?"]
        blanks = ", ".join(str(number) for number in visual.get("blank_question_numbers") or [])
        raw_text = visual.get("raw_text") or ""
        return (
            f"{label} Questions {start}-{end}\n"
            f"Blank question numbers: {blanks}\n"
            "Structured rows/nodes could not be inferred reliably from the extracted text.\n"
            f"Raw visual text: {raw_text}"
        )

    def _make_chunk(
        self,
        document: ProcessedDocument,
        index: int,
        chunk_id: str,
        text: str,
        retrieval_text: str,
        pages: list[int],
        element_ids: list[str],
        heading_path: list[str],
        min_confidence: float,
        metadata: dict[str, Any],
    ) -> DocumentChunk:
        metadata.update(
            {
                "mime_type": document.mime_type,
                "parser_version": document.parser_version,
                "structured": True,
            }
        )
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            source_file=document.filename,
            pages=pages,
            element_ids=element_ids,
            heading_path=heading_path,
            text=text,
            retrieval_text=retrieval_text,
            display_text=text,
            token_count=estimate_tokens(text),
            min_confidence=min_confidence,
            chunk_index=index,
            metadata=metadata,
        )

    def _split_text(self, text: str, max_tokens: int) -> list[str]:
        if estimate_tokens(text) <= max_tokens:
            return [text]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            if estimate_tokens(sentence) > max_tokens:
                if current:
                    chunks.append(" ".join(current).strip())
                    current = []
                chunks.extend(self._split_oversized_sentence(sentence, max_tokens))
                continue
            projected = " ".join(current + [sentence])
            if current and estimate_tokens(projected) > max_tokens:
                chunks.append(" ".join(current).strip())
                current = []
            current.append(sentence)
        if current:
            chunks.append(" ".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _split_oversized_sentence(self, sentence: str, max_tokens: int) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        for word in sentence.split():
            projected = " ".join(current + [word])
            if current and estimate_tokens(projected) > max_tokens:
                chunks.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _group_ranges(self, passage: IELTSPassage) -> str:
        return ", ".join(f"{group.question_start}-{group.question_end}" for group in passage.question_groups)

    def _min_confidence(self, document: ProcessedDocument, element_ids: list[str]) -> float:
        by_id = {element.element_id: element for element in document.elements}
        confidences = [by_id[element_id].confidence for element_id in element_ids if element_id in by_id]
        return min(confidences) if confidences else 1.0
