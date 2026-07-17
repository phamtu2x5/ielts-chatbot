import re
from dataclasses import dataclass
from typing import Any

from .config import DocumentPipelineConfig
from .models import DocumentElement, ProcessedDocument


TASK_TITLE_RE = re.compile(r"\b(?:ielts\s+)?(?:essay\s+)?task\s*1\b", re.IGNORECASE)
PROMPT_RE = re.compile(
    r"\b(?:the\s+)?(?:pie\s+|line\s+|bar\s+)?(?:chart|graph|table|map|diagram)s?\b.*\b"
    r"(?:show|shows|illustrate|illustrates|describe|describes|give|gives|compare|compares)\b",
    re.IGNORECASE | re.DOTALL,
)
VISUAL_INTRO_RE = re.compile(
    r"\b(?:pie\s+chart|line\s+chart|bar\s+chart|chart|graph|table|map|diagram)s?\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


@dataclass
class WritingSection:
    section_id: str
    type: str
    task_index: int
    title: str | None
    visual_type: str | None
    text: str
    pages: list[int]
    element_ids: list[str]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "type": self.type,
            "task_index": self.task_index,
            "title": self.title,
            "visual_type": self.visual_type,
            "text": self.text,
            "pages": self.pages,
            "element_ids": self.element_ids,
            "confidence": self.confidence,
        }


@dataclass
class _TaskAnchor:
    element_index: int
    page: int
    title: str | None
    inferred: bool = False


class WritingCollectionParser:
    """Find repeated Writing Task 1 and sample-answer sections from document structure."""

    def __init__(self, config: DocumentPipelineConfig) -> None:
        self.config = config

    def parse(self, document: ProcessedDocument) -> list[WritingSection]:
        elements = [
            element
            for element in document.elements
            if self._is_primary_content(element)
        ]
        if not any(TASK_TITLE_RE.search(self._canonical_text(element)) for element in elements):
            return []
        anchors = self._explicit_anchors(elements)
        anchors.extend(self._visual_only_anchors(document, elements, anchors))
        anchors = self._dedupe_anchors(anchors)
        if len(anchors) < self.config.writing_collection_min_tasks:
            return []

        sections: list[WritingSection] = []
        for task_index, anchor in enumerate(anchors, 1):
            block_end = anchors[task_index].element_index if task_index < len(anchors) else len(elements)
            block = elements[anchor.element_index:block_end]
            task_elements, answer_elements = self._split_task_and_answer(block, anchor)
            evidence_text = self._section_text(task_elements + answer_elements[:2])
            visual_type = self._visual_type(evidence_text, document, anchor.page)
            title = anchor.title
            task_text = self._section_text(task_elements)
            if not task_text:
                task_text = self._task_placeholder(visual_type)

            sections.append(
                self._section(
                    section_type="writing_task_1",
                    task_index=task_index,
                    title=title,
                    visual_type=visual_type,
                    text=task_text,
                    elements=task_elements,
                    fallback_page=anchor.page,
                    confidence=0.75 if anchor.inferred else 0.95,
                )
            )
            if answer_elements:
                sections.append(
                    self._section(
                        section_type="sample_answer",
                        task_index=task_index,
                        title=title,
                        visual_type=visual_type,
                        text=self._section_text(answer_elements),
                        elements=answer_elements,
                        fallback_page=anchor.page,
                        confidence=min(element.confidence for element in answer_elements),
                    )
                )
        return sections

    def _explicit_anchors(self, elements: list[DocumentElement]) -> list[_TaskAnchor]:
        anchors: list[_TaskAnchor] = []
        pages_with_anchor: set[int] = set()
        for index, element in enumerate(elements):
            text = self._canonical_text(element)
            if TASK_TITLE_RE.search(text):
                anchors.append(_TaskAnchor(index, element.page, self._title(element.raw_text or text)))
                pages_with_anchor.add(element.page)

        for index, element in enumerate(elements):
            if element.page in pages_with_anchor:
                continue
            text = self._canonical_text(element)
            if not PROMPT_RE.search(text):
                continue
            title = None
            anchor_index = index
            if index > 0 and elements[index - 1].page == element.page:
                candidate = self._canonical_text(elements[index - 1])
                if self._is_short_title(candidate):
                    title = self._title(candidate)
                    anchor_index = index - 1
            anchors.append(_TaskAnchor(anchor_index, element.page, title))
            pages_with_anchor.add(element.page)
        return anchors

    def _visual_only_anchors(
        self,
        document: ProcessedDocument,
        elements: list[DocumentElement],
        anchors: list[_TaskAnchor],
    ) -> list[_TaskAnchor]:
        anchored_pages = {anchor.page for anchor in anchors}
        inferred: list[_TaskAnchor] = []
        index_by_id = {element.element_id: index for index, element in enumerate(elements)}
        for page in document.pages:
            if page.page_number in anchored_pages or not page.metadata.get("requires_layout"):
                continue
            page_elements = [
                element
                for element in page.elements
                if element.element_id in index_by_id and element.bbox and element.source != "pdf_page_ocr"
            ]
            page_elements.sort(key=lambda element: (element.bbox[1], element.bbox[0]))
            if len(page_elements) < 2:
                continue
            page_span = max(element.bbox[3] for element in page_elements) - min(
                element.bbox[1] for element in page_elements
            )
            if page_span <= 0:
                continue
            gap, candidate = max(
                (
                    max(0.0, current.bbox[1] - previous.bbox[3]),
                    current,
                )
                for previous, current in zip(page_elements, page_elements[1:])
            )
            if gap / page_span < self.config.writing_visual_gap_ratio:
                continue
            if not VISUAL_INTRO_RE.search(self._canonical_text(candidate)):
                continue
            inferred.append(
                _TaskAnchor(
                    element_index=index_by_id[candidate.element_id],
                    page=page.page_number,
                    title=None,
                    inferred=True,
                )
            )
        return inferred

    def _dedupe_anchors(self, anchors: list[_TaskAnchor]) -> list[_TaskAnchor]:
        result: list[_TaskAnchor] = []
        for anchor in sorted(anchors, key=lambda item: item.element_index):
            if result and result[-1].element_index == anchor.element_index:
                if result[-1].title is None and anchor.title is not None:
                    result[-1] = anchor
                continue
            result.append(anchor)
        return result

    def _split_task_and_answer(
        self,
        block: list[DocumentElement],
        anchor: _TaskAnchor,
    ) -> tuple[list[DocumentElement], list[DocumentElement]]:
        content = [element for element in block if element.source != "pdf_page_ocr"]
        if anchor.inferred:
            return [], content
        split = 1
        while split < len(content) and self._is_adjacent_prompt(content[split - 1], content[split]):
            split += 1
        return content[:split], content[split:]

    def _is_adjacent_prompt(self, previous: DocumentElement, candidate: DocumentElement) -> bool:
        if not PROMPT_RE.search(self._canonical_text(candidate)):
            return False
        if previous.page != candidate.page or not previous.bbox or not candidate.bbox:
            return True
        gap = max(0.0, candidate.bbox[1] - previous.bbox[3])
        previous_height = max(1.0, previous.bbox[3] - previous.bbox[1])
        candidate_height = max(1.0, candidate.bbox[3] - candidate.bbox[1])
        return gap <= max(previous_height, candidate_height) * self.config.writing_adjacent_block_gap_factor

    def _section(
        self,
        section_type: str,
        task_index: int,
        title: str | None,
        visual_type: str | None,
        text: str,
        elements: list[DocumentElement],
        fallback_page: int,
        confidence: float,
    ) -> WritingSection:
        pages = sorted({element.page for element in elements}) or [fallback_page]
        suffix = "task" if section_type == "writing_task_1" else "answer"
        return WritingSection(
            section_id=f"writing-{task_index}-{suffix}",
            type=section_type,
            task_index=task_index,
            title=title,
            visual_type=visual_type,
            text=text,
            pages=pages,
            element_ids=[element.element_id for element in elements],
            confidence=confidence,
        )

    def _visual_type(self, text: str, document: ProcessedDocument, page_number: int) -> str | None:
        lowered = text.lower()
        if "bar chart" in lowered and "pie chart" in lowered:
            return "combined_bar_pie_chart"
        for phrase, visual_type in (
            ("pie chart", "pie_chart"),
            ("line chart", "line_chart"),
            ("bar chart", "bar_chart"),
            ("map", "map"),
            ("table", "table"),
            ("diagram", "diagram"),
            ("graph", "graph"),
        ):
            if phrase in lowered:
                return visual_type
        page = next((item for item in document.pages if item.page_number == page_number), None)
        if page:
            types = {
                str(region.get("type") or "").lower()
                for region in page.metadata.get("layout_regions") or []
            }
            if "table" in types:
                return "table"
        return None

    def _title(self, text: str) -> str | None:
        for line in text.splitlines():
            line = URL_RE.sub("", line).strip()
            if line:
                return line.rstrip(":")
        return None

    def _is_short_title(self, text: str) -> bool:
        clean = URL_RE.sub("", text).strip()
        return bool(clean and len(clean.split()) <= 12 and not clean.endswith((".", "?")))

    def _canonical_text(self, element: DocumentElement) -> str:
        return (element.normalized_text or element.raw_text).strip()

    def _is_primary_content(self, element: DocumentElement) -> bool:
        if element.source == "pdf_page_ocr" or element.type == "ocr_supplement":
            return False
        text = self._canonical_text(element)
        return bool(text and not URL_RE.fullmatch(text))

    def _section_text(self, elements: list[DocumentElement]) -> str:
        return "\n\n".join(self._canonical_text(element) for element in elements if self._canonical_text(element))

    def _task_placeholder(self, visual_type: str | None) -> str:
        label = visual_type.replace("_", " ") if visual_type else "visual"
        return f"IELTS Writing Task 1 ({label})."
