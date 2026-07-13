import re
from dataclasses import dataclass
from typing import Any

from .normalization import normalize_text


@dataclass
class ParsedWritingTable:
    document_type: str
    task_type: str
    prompt: dict[str, Any]
    table: dict[str, Any]

    def prompt_text(self) -> str:
        parts = []
        if self.prompt.get("time_minutes"):
            parts.append(f"Time: {self.prompt['time_minutes']} minutes.")
        if self.prompt.get("description"):
            parts.append(str(self.prompt["description"]))
        if self.prompt.get("instruction"):
            parts.append(str(self.prompt["instruction"]))
        if self.prompt.get("minimum_words"):
            parts.append(f"Minimum words: {self.prompt['minimum_words']}.")
        return "\n".join(parts).strip()

    def table_markdown(self) -> str:
        columns = self.table.get("columns") or []
        rows = self.table.get("rows") or []
        if not columns or not rows:
            return ""
        lines = [
            "| " + " | ".join(str(column) for column in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        return "\n".join(lines)


@dataclass
class ParsedQuestionVisual:
    visual_type: str
    element_id: str
    payload: dict[str, Any]
    display_text: str
    confidence: float


class WritingTaskTableParser:
    """Lightweight parser for OCR text from IELTS Academic Task 1 table prompts."""

    def parse(self, text: str) -> ParsedWritingTable | None:
        normalized = normalize_text(text)
        lowered = normalized.lower()
        if "table" not in lowered or not any(marker in lowered for marker in ["summarise", "summarize"]):
            return None

        rows = self._parse_rows(normalized)
        if len(rows) < 2:
            return None

        columns = self._parse_columns(normalized, row_width=len(rows[0]) - 1)
        prompt = {
            "time_minutes": self._parse_int(r"spend\s+about\s+(\d+)\s+minutes", normalized),
            "minimum_words": self._parse_int(r"at\s+least\s+(\d+)\s+words", normalized),
            "description": self._parse_description(normalized),
            "instruction": self._parse_instruction(normalized),
        }
        table = {
            "type": "table",
            "columns": columns,
            "rows": rows,
            "bbox": [],
            "source": "image_ocr",
            "confidence": 0.0,
        }
        return ParsedWritingTable(
            document_type="ielts_writing_task_1",
            task_type="academic_task_1_table",
            prompt=prompt,
            table=table,
        )

    def _parse_rows(self, text: str) -> list[list[Any]]:
        rows: list[list[Any]] = []
        tokens = re.findall(r"[A-Za-z][A-Za-z.&/-]*|\d+(?:\.\d+)?%?", text)
        index = 0
        while index < len(tokens):
            if self._is_number_token(tokens[index]):
                index += 1
                continue
            label_tokens = []
            cursor = index
            while cursor < len(tokens) and not self._is_number_token(tokens[cursor]) and len(label_tokens) < 5:
                label_tokens.append(tokens[cursor])
                cursor += 1
            values = []
            while cursor < len(tokens) and self._is_number_token(tokens[cursor]):
                values.append(self._parse_number(tokens[cursor]))
                cursor += 1
            if len(values) < 2:
                index += 1
                continue
            label = self._clean_row_label(" ".join(label_tokens))
            if not label:
                index = cursor
                continue
            rows.append([label, *values])
            index = cursor
        if not rows:
            return []
        widths = [len(row) for row in rows]
        width = max(set(widths), key=widths.count)
        return [row for row in rows if len(row) == width]

    def _is_number_token(self, token: str) -> bool:
        return bool(re.fullmatch(r"\d+(?:\.\d+)?%?", token))

    def _clean_row_label(self, label: str) -> str:
        label = re.sub(r"\s+", " ", label).strip(" .:-")
        words = label.split()
        if not words:
            return ""
        if len(words) == 1 and len(words[0]) <= 2:
            letters = re.findall(r"[A-Za-z]", words[0])
            return letters[0].upper() if letters else ""
        return " ".join(words[-4:])

    def _parse_number(self, value: str) -> int | float:
        number = float(value.rstrip("%"))
        return int(number) if number.is_integer() else number

    def _parse_columns(self, text: str, row_width: int) -> list[str]:
        row_match = re.search(r"(?:^|\s)([A-Za-z][A-Za-z .&/-]{0,40}?)\s+(?:\d+(?:\.\d+)?%?\s+){2,8}", text)
        header_text = text[: row_match.start()].strip() if row_match else text
        year_matches = list(re.finditer(r"\b(20\d{2})\b", header_text))
        value_columns: list[str] = []
        cursor = 0
        for match in year_matches:
            label = self._clean_header_label(header_text[cursor : match.start()])
            if label:
                value_columns.append(f"{label} {match.group(1)}")
            cursor = match.end()
        if len(value_columns) > row_width:
            value_columns = value_columns[-row_width:]
        if len(value_columns) == row_width:
            row_header, value_columns = self._split_row_header(value_columns)
            return [row_header, *value_columns]
        return ["Label"] + [f"Value {index}" for index in range(1, row_width + 1)]

    def _clean_header_label(self, text: str) -> str:
        text = re.sub(r"\b(?:the|table|below|shows?|percentages?|percentage|of|people|in|who|had|and)\b", " ", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip(" .:-")

    def _split_row_header(self, columns: list[str]) -> tuple[str, list[str]]:
        if len(columns) < 2:
            return "Label", columns
        first_without_year = re.sub(r"\s+20\d{2}$", "", columns[0])
        later_labels = [re.sub(r"\s+20\d{2}$", "", column) for column in columns[1:]]
        for later_label in later_labels:
            if first_without_year.endswith(later_label):
                prefix = first_without_year[: -len(later_label)].strip()
                prefix = prefix.split(".")[-1].strip()
                prefix_words = prefix.split()
                if len(prefix_words) > 4:
                    prefix = " ".join(prefix_words[-4:])
                return prefix or "Label", [columns[0].replace(first_without_year, later_label, 1), *columns[1:]]
        return "Label", columns

    def _parse_description(self, text: str) -> str | None:
        match = re.search(
            r"(The\s+table\s+below\s+shows.+?)(?=Summari[sz]e\s+the\s+information|Write\s+at\s+least|$)",
            text,
            flags=re.IGNORECASE,
        )
        return self._clean_sentence(match.group(1)) if match else None

    def _parse_instruction(self, text: str) -> str | None:
        match = re.search(
            r"(Summari[sz]e\s+the\s+information.+?)(?=Write\s+at\s+least|$)",
            text,
            flags=re.IGNORECASE,
        )
        return self._clean_sentence(match.group(1)) if match else None

    def _parse_int(self, pattern: str, text: str) -> int | None:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _clean_sentence(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip(" .") + "."


class IELTSQuestionVisualParser:
    """Extracts structured visual placeholders from IELTS question-group text.

    This parser is intentionally conservative. When table/flowchart structure
    cannot be inferred from text order, it records the known range, blanks and
    raw linearized text instead of inventing rows, columns, nodes or edges.
    """

    def parse(
        self,
        text: str,
        question_start: int,
        question_end: int,
        question_type: str | None,
        page_numbers: list[int],
        source_element_ids: list[str],
    ) -> ParsedQuestionVisual | None:
        if question_type == "table_completion":
            return self._parse_table(text, question_start, question_end, page_numbers, source_element_ids)
        if question_type == "flowchart_completion":
            return self._parse_flowchart(text, question_start, question_end, page_numbers, source_element_ids)
        return None

    def _parse_table(
        self,
        text: str,
        question_start: int,
        question_end: int,
        page_numbers: list[int],
        source_element_ids: list[str],
    ) -> ParsedQuestionVisual:
        blank_numbers = list(range(question_start, question_end + 1))
        columns, rows = self._parse_markdown_like_table(text)
        confidence = 0.78 if columns and rows else 0.35
        table = {
            "type": "table",
            "question_range": [question_start, question_end],
            "columns": columns,
            "rows": rows,
            "blank_question_numbers": blank_numbers,
            "bbox": [],
            "source": "ielts_question_group_parser",
            "confidence": confidence,
            "raw_text": text,
            "page_numbers": page_numbers,
            "source_element_ids": source_element_ids,
        }
        display = self._table_markdown(table) if columns and rows else self._visual_fallback_text("table", table)
        return ParsedQuestionVisual(
            visual_type="table",
            element_id=f"visual-table-{question_start}-{question_end}",
            payload=table,
            display_text=display,
            confidence=confidence,
        )

    def _parse_flowchart(
        self,
        text: str,
        question_start: int,
        question_end: int,
        page_numbers: list[int],
        source_element_ids: list[str],
    ) -> ParsedQuestionVisual:
        blank_numbers = list(range(question_start, question_end + 1))
        nodes, edges = self._parse_arrow_flow(text, blank_numbers)
        confidence = 0.7 if nodes and edges else 0.35
        flowchart = {
            "type": "flowchart",
            "question_range": [question_start, question_end],
            "nodes": nodes,
            "edges": edges,
            "blank_question_numbers": blank_numbers,
            "bbox": [],
            "source": "ielts_question_group_parser",
            "confidence": confidence,
            "raw_text": text,
            "page_numbers": page_numbers,
            "source_element_ids": source_element_ids,
        }
        display = self._flowchart_text(flowchart) if nodes else self._visual_fallback_text("flowchart", flowchart)
        return ParsedQuestionVisual(
            visual_type="flowchart",
            element_id=f"visual-flowchart-{question_start}-{question_end}",
            payload=flowchart,
            display_text=display,
            confidence=confidence,
        )

    def _parse_markdown_like_table(self, text: str) -> tuple[list[str], list[list[str]]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        table_lines = [line for line in lines if "|" in line and line.count("|") >= 2]
        if len(table_lines) < 2:
            return [], []

        rows = []
        for line in table_lines:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells and not all(re.fullmatch(r"[-:\s]+", cell) for cell in cells):
                rows.append(cells)
        if len(rows) < 2:
            return [], []

        width = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (width - len(row)) for row in rows]
        return normalized_rows[0], normalized_rows[1:]

    def _parse_arrow_flow(self, text: str, blank_numbers: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        arrow_pattern = r"\s*(?:->|→|↓|=>|➜)\s*"
        if not re.search(arrow_pattern, text):
            return self._blank_nodes(blank_numbers), []

        parts = [part.strip(" .;") for part in re.split(arrow_pattern, text) if part.strip(" .;")]
        if len(parts) < 2:
            return self._blank_nodes(blank_numbers), []

        nodes = []
        for index, part in enumerate(parts, 1):
            number = self._blank_number_in_text(part, blank_numbers)
            nodes.append(
                {
                    "id": f"node-{index}",
                    "text": None if number else part,
                    "question_number": number,
                }
            )
        edges = [
            {"from": nodes[index]["id"], "to": nodes[index + 1]["id"]}
            for index in range(len(nodes) - 1)
        ]
        return nodes, edges

    def _blank_nodes(self, blank_numbers: list[int]) -> list[dict[str, Any]]:
        return [
            {
                "id": f"blank-{number}",
                "text": None,
                "question_number": number,
            }
            for number in blank_numbers
        ]

    def _blank_number_in_text(self, text: str, blank_numbers: list[int]) -> int | None:
        for number in blank_numbers:
            if re.search(rf"(?<!\d){number}(?!\d)", text):
                return number
        return None

    def _table_markdown(self, table: dict[str, Any]) -> str:
        columns = table.get("columns") or []
        rows = table.get("rows") or []
        if not columns or not rows:
            return ""
        lines = [
            "| " + " | ".join(str(column) for column in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        return "\n".join(lines)

    def _flowchart_text(self, flowchart: dict[str, Any]) -> str:
        lines = [f"Flowchart Questions {flowchart['question_range'][0]}-{flowchart['question_range'][1]}"]
        for node in flowchart.get("nodes") or []:
            label = f"Question {node['question_number']} blank" if node.get("question_number") else node.get("text", "")
            lines.append(f"- {node['id']}: {label}")
        for edge in flowchart.get("edges") or []:
            lines.append(f"- edge: {edge['from']} -> {edge['to']}")
        return "\n".join(lines)

    def _visual_fallback_text(self, visual_type: str, payload: dict[str, Any]) -> str:
        start, end = payload["question_range"]
        blanks = ", ".join(str(number) for number in payload["blank_question_numbers"])
        return (
            f"{visual_type.title()} Questions {start}-{end}\n"
            f"Blank question numbers: {blanks}\n"
            "Structured rows/nodes could not be inferred reliably from the extracted text.\n"
            f"Raw visual text: {payload.get('raw_text', '')}"
        )
