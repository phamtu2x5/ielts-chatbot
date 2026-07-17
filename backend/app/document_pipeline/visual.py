import re
from collections import Counter
from dataclasses import dataclass
from statistics import median
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


@dataclass
class _SpatialOCRLine:
    text: str
    confidence: float
    bbox: list[float]

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])


def _flat_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    if all(isinstance(item, (int, float)) for item in value[:4]) and len(value) >= 4:
        x1, y1, x2, y2 = (float(item) for item in value[:4])
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    points = [point for point in value if isinstance(point, (list, tuple)) and len(point) >= 2]
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _spatial_line(item: dict[str, Any]) -> _SpatialOCRLine | None:
    text = normalize_text(str(item.get("text") or ""))
    bbox = _flat_bbox(item.get("bbox"))
    if not text or bbox is None:
        return None
    return _SpatialOCRLine(
        text=text,
        confidence=max(0.0, min(1.0, float(item.get("confidence") or 0.0))),
        bbox=bbox,
    )


def _center_inside(line: _SpatialOCRLine, bbox: list[float]) -> bool:
    return bbox[0] <= line.center_x <= bbox[2] and bbox[1] <= line.center_y <= bbox[3]


def _cluster_spatial_rows(lines: list[_SpatialOCRLine]) -> list[list[_SpatialOCRLine]]:
    if not lines:
        return []
    tolerance = max(4.0, median(line.height for line in lines) * 0.65)
    rows: list[list[_SpatialOCRLine]] = []
    for line in sorted(lines, key=lambda item: (item.center_y, item.center_x)):
        if not rows:
            rows.append([line])
            continue
        row_center = sum(item.center_y for item in rows[-1]) / len(rows[-1])
        if abs(line.center_y - row_center) <= tolerance:
            rows[-1].append(line)
        else:
            rows.append([line])
    return rows


class WritingTaskTableParser:
    """Lightweight parser for OCR text from IELTS Academic Task 1 table prompts."""

    def parse(
        self,
        text: str,
        ocr_lines: list[dict[str, Any]] | None = None,
        layout_regions: list[dict[str, Any]] | None = None,
    ) -> ParsedWritingTable | None:
        normalized = normalize_text(text)
        lowered = normalized.lower()
        if "table" not in lowered or not any(marker in lowered for marker in ["summarise", "summarize"]):
            return None

        spatial_table = self._parse_spatial_table(ocr_lines or [], layout_regions or [])
        rows = spatial_table.get("rows") if spatial_table else self._parse_rows(normalized)
        if len(rows) < 2:
            return None

        columns = (
            spatial_table.get("columns")
            if spatial_table
            else self._parse_columns(normalized, row_width=len(rows[0]) - 1)
        )
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
            "bbox": spatial_table.get("bbox", []) if spatial_table else [],
            "source": spatial_table.get("source", "image_ocr") if spatial_table else "image_ocr",
            "confidence": spatial_table.get("confidence", 0.0) if spatial_table else 0.0,
        }
        return ParsedWritingTable(
            document_type="ielts_writing_task_1",
            task_type="academic_task_1_table",
            prompt=prompt,
            table=table,
        )

    def _parse_spatial_table(
        self,
        raw_lines: list[dict[str, Any]],
        layout_regions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        lines = [line for line in (self._spatial_line(item) for item in raw_lines) if line]
        table_regions = [
            region
            for region in layout_regions
            if "table" in str(region.get("type", "")).lower() and self._flat_bbox(region.get("bbox"))
        ]
        candidates = []
        for region in table_regions:
            bbox = self._flat_bbox(region.get("bbox"))
            assert bbox is not None
            region_lines = [line for line in lines if self._center_inside(line, bbox)]
            parsed = self._reconstruct_table(region_lines)
            if not parsed:
                continue
            parsed.update(
                {
                    "bbox": bbox,
                    "source": "doclayout_yolo+rapidocr_boxes",
                    "confidence": min(
                        float(region.get("confidence") or 0.0),
                        sum(line.confidence for line in region_lines) / max(1, len(region_lines)),
                    ),
                }
            )
            candidates.append(parsed)
        if not candidates:
            return None
        return max(candidates, key=lambda item: len(item["rows"]) * len(item["columns"]))

    def _reconstruct_table(self, lines: list[_SpatialOCRLine]) -> dict[str, Any] | None:
        clustered_rows = self._cluster_rows(lines)
        data_rows: list[tuple[int, str, list[tuple[_SpatialOCRLine, int | float]]]] = []
        for row_index, row in enumerate(clustered_rows):
            ordered = sorted(row, key=lambda item: item.center_x)
            for split_index in range(1, len(ordered) - 1):
                value_lines = ordered[split_index:]
                if not all(self._is_number_token(line.text.strip()) for line in value_lines):
                    continue
                label = normalize_text(" ".join(line.text for line in ordered[:split_index]))
                values = [(line, self._parse_number(line.text)) for line in value_lines]
                if label and len(values) >= 2:
                    data_rows.append((row_index, label, values))
                    break
        if len(data_rows) < 2:
            return None

        width_counts = Counter(len(values) for _, _, values in data_rows)
        value_width = width_counts.most_common(1)[0][0]
        data_rows = [row for row in data_rows if len(row[2]) == value_width]
        if len(data_rows) < 2:
            return None

        value_centers = [
            median(row[2][column][0].center_x for row in data_rows)
            for column in range(value_width)
        ]
        label_centers = []
        for row_index, _, values in data_rows:
            first_value_x = values[0][0].bbox[0]
            label_centers.extend(
                line.center_x for line in clustered_rows[row_index] if line.bbox[2] < first_value_x
            )
        column_centers = [median(label_centers), *value_centers]

        header_parts: list[list[tuple[float, str]]] = [[] for _ in column_centers]
        first_data_row = min(row[0] for row in data_rows)
        for row in clustered_rows[:first_data_row]:
            for line in row:
                column_index = min(
                    range(len(column_centers)),
                    key=lambda index: abs(line.center_x - column_centers[index]),
                )
                header_parts[column_index].append((line.center_y, line.text))
        columns = [self._join_header_parts(parts) for parts in header_parts]
        columns = [column or ("Label" if index == 0 else f"Value {index}") for index, column in enumerate(columns)]
        rows = [[label, *[value for _, value in values]] for _, label, values in data_rows]
        return {"columns": columns, "rows": rows}

    def _cluster_rows(self, lines: list[_SpatialOCRLine]) -> list[list[_SpatialOCRLine]]:
        return _cluster_spatial_rows(lines)

    def _spatial_line(self, item: dict[str, Any]) -> _SpatialOCRLine | None:
        return _spatial_line(item)

    def _flat_bbox(self, value: Any) -> list[float] | None:
        return _flat_bbox(value)

    def _center_inside(self, line: _SpatialOCRLine, bbox: list[float]) -> bool:
        return _center_inside(line, bbox)

    def _join_header_parts(self, parts: list[tuple[float, str]]) -> str:
        values = []
        for _, text in sorted(parts):
            cleaned = normalize_text(text)
            if cleaned and cleaned not in values:
                values.append(cleaned)
        return " ".join(values)

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
            flags=re.IGNORECASE | re.DOTALL,
        )
        return self._clean_sentence(match.group(1)) if match else None

    def _parse_instruction(self, text: str) -> str | None:
        match = re.search(
            r"(Summari[sz]e\s+the\s+information.+?)(?=Write\s+at\s+least|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return self._clean_sentence(match.group(1)) if match else None

    def _parse_int(self, pattern: str, text: str) -> int | None:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _clean_sentence(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip(" .") + "."


class IELTSQuestionVisualParser:
    """Extracts structured visuals from spatial OCR, with a text-only fallback.

    This parser is intentionally conservative. When table/flowchart structure
    cannot be inferred from text order, it records the known range, blanks and
    raw linearized text instead of inventing rows, columns, nodes or edges.
    """

    def __init__(self, direction_min_confidence: float = 0.55) -> None:
        self.direction_min_confidence = direction_min_confidence

    def parse(
        self,
        text: str,
        question_start: int,
        question_end: int,
        question_type: str | None,
        page_numbers: list[int],
        source_element_ids: list[str],
        spatial_pages: list[dict[str, Any]] | None = None,
    ) -> ParsedQuestionVisual | None:
        spatial = self._parse_spatial_visual(
            text,
            question_start,
            question_end,
            question_type,
            source_element_ids,
            spatial_pages or [],
        )
        if spatial:
            return spatial
        if question_type == "table_completion":
            return self._parse_table(text, question_start, question_end, page_numbers, source_element_ids)
        if question_type == "flowchart_completion":
            return self._parse_flowchart(text, question_start, question_end, page_numbers, source_element_ids)
        return None

    def _parse_spatial_visual(
        self,
        text: str,
        question_start: int,
        question_end: int,
        question_type: str | None,
        source_element_ids: list[str],
        spatial_pages: list[dict[str, Any]],
    ) -> ParsedQuestionVisual | None:
        if question_type not in {"flowchart_completion", "diagram_labeling"}:
            table_candidate = self._best_region_candidate(
                spatial_pages,
                "table",
                question_start,
                question_end,
            )
            if table_candidate:
                table = self._reconstruct_spatial_table(
                    table_candidate,
                    text,
                    question_start,
                    question_end,
                    source_element_ids,
                )
                if table:
                    return ParsedQuestionVisual(
                        visual_type="table",
                        element_id=f"visual-table-{question_start}-{question_end}",
                        payload=table,
                        display_text=self._table_markdown(table),
                        confidence=float(table["confidence"]),
                    )

        if question_type == "flowchart_completion":
            figure_candidate = self._best_region_candidate(
                spatial_pages,
                "figure",
                question_start,
                question_end,
            )
            if figure_candidate:
                flowchart = self._reconstruct_spatial_flowchart(
                    figure_candidate,
                    text,
                    question_start,
                    question_end,
                    source_element_ids,
                )
                if flowchart:
                    return ParsedQuestionVisual(
                        visual_type="flowchart",
                        element_id=f"visual-flowchart-{question_start}-{question_end}",
                        payload=flowchart,
                        display_text=self._flowchart_text(flowchart),
                        confidence=float(flowchart["confidence"]),
                    )
        if question_type == "diagram_labeling":
            figure_candidate = self._best_region_candidate(
                spatial_pages,
                "figure",
                question_start,
                question_end,
            )
            if figure_candidate:
                diagram = self._reconstruct_spatial_diagram(
                    figure_candidate,
                    text,
                    question_start,
                    question_end,
                    source_element_ids,
                )
                if diagram:
                    return ParsedQuestionVisual(
                        visual_type="diagram",
                        element_id=f"visual-diagram-{question_start}-{question_end}",
                        payload=diagram,
                        display_text=self._diagram_text(diagram),
                        confidence=float(diagram["confidence"]),
                    )
        return None

    def _best_region_candidate(
        self,
        spatial_pages: list[dict[str, Any]],
        region_type: str,
        question_start: int,
        question_end: int,
    ) -> dict[str, Any] | None:
        expected = set(range(question_start, question_end + 1))
        candidates = []
        for page in spatial_pages:
            lines = [line for line in (_spatial_line(item) for item in page.get("ocr_lines") or []) if line]
            for region in page.get("layout_regions") or []:
                normalized_type = str(region.get("type") or "").strip().lower().replace(" ", "_")
                if normalized_type != region_type:
                    continue
                bbox = _flat_bbox(region.get("bbox"))
                if bbox is None:
                    continue
                region_lines = [line for line in lines if _center_inside(line, bbox)]
                matched = self._question_numbers(" ".join(line.text for line in region_lines), expected)
                if not matched:
                    continue
                candidates.append(
                    {
                        "page": int(page.get("page") or 0),
                        "bbox": bbox,
                        "lines": region_lines,
                        "matched_questions": matched,
                        "layout_confidence": float(region.get("confidence") or 0.0),
                        "connectors": self._region_connectors(page.get("connector_regions") or [], bbox),
                    }
                )
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                len(item["matched_questions"]) / max(1, len(expected)),
                len(item["matched_questions"]),
                item["layout_confidence"],
                len(item["lines"]),
            ),
        )

    def _reconstruct_spatial_table(
        self,
        candidate: dict[str, Any],
        raw_text: str,
        question_start: int,
        question_end: int,
        source_element_ids: list[str],
    ) -> dict[str, Any] | None:
        blank_numbers = list(range(question_start, question_end + 1))
        expected = set(blank_numbers)
        clustered_rows = _cluster_spatial_rows(candidate["lines"])
        usable_rows = [row for row in clustered_rows if 2 <= len(row) <= 8]
        if len(usable_rows) < 2:
            return None

        first_numbered_row = next(
            (
                index
                for index, row in enumerate(clustered_rows)
                if self._question_numbers(" ".join(line.text for line in row), expected)
            ),
            None,
        )
        column_centers = self._table_column_centers(clustered_rows, usable_rows, first_numbered_row)
        if len(column_centers) < 2:
            return None
        column_count = len(column_centers)

        spatial_rows: list[list[str]] = []
        row_numbers: list[set[int]] = []
        for clustered_index, row in enumerate(clustered_rows):
            cells: list[list[_SpatialOCRLine]] = [[] for _ in column_centers]
            for line in sorted(row, key=lambda item: item.center_x):
                column_index = min(
                    range(len(column_centers)),
                    key=lambda index: abs(line.center_x - column_centers[index]),
                )
                cells[column_index].append(line)
            values = [normalize_text(" ".join(line.text for line in cell)) for cell in cells]
            populated = sum(bool(value) for value in values)
            is_header_continuation = first_numbered_row is not None and clustered_index < first_numbered_row
            if populated < 2 and not (is_header_continuation and populated == 1):
                continue
            spatial_rows.append(values)
            row_numbers.append(self._question_numbers(" ".join(values), expected))

        first_data_row = next((index for index, numbers in enumerate(row_numbers) if numbers), None)
        if first_data_row is None or len(spatial_rows) - first_data_row < 2:
            return None

        header_rows = spatial_rows[:first_data_row]
        columns = []
        for column_index in range(column_count):
            parts = [row[column_index] for row in header_rows if row[column_index]]
            columns.append(normalize_text(" ".join(dict.fromkeys(parts))))
        rows = spatial_rows[first_data_row:]
        matched = set().union(*row_numbers[first_data_row:])
        missing_questions = sorted(expected - matched)
        duplicate_questions = sorted(
            number
            for number in expected
            if sum(number in numbers for numbers in row_numbers[first_data_row:]) > 1
        )
        missing_headers = [index + 1 for index, column in enumerate(columns) if not column]
        quality_issues = []
        if missing_questions:
            quality_issues.append("missing_question_numbers")
        if duplicate_questions:
            quality_issues.append("duplicate_question_numbers")
        if missing_headers:
            quality_issues.append("missing_column_headers")
        coverage = len(matched) / max(1, len(expected))
        ocr_confidence = sum(line.confidence for line in candidate["lines"]) / max(1, len(candidate["lines"]))
        confidence = min(candidate["layout_confidence"], ocr_confidence) * (0.75 + 0.25 * coverage)
        return {
            "type": "table",
            "question_range": [question_start, question_end],
            "columns": columns,
            "rows": rows,
            "blank_question_numbers": blank_numbers,
            "bbox": candidate["bbox"],
            "source": "doclayout_yolo+rapidocr_boxes",
            "confidence": round(confidence, 4),
            "raw_text": raw_text,
            "page_numbers": [candidate["page"]],
            "source_element_ids": source_element_ids,
            "quality_status": "degraded" if quality_issues else "passed",
            "quality_issues": quality_issues,
            "missing_question_numbers": missing_questions,
            "duplicate_question_numbers": duplicate_questions,
            "missing_column_headers": missing_headers,
        }

    def _table_column_centers(
        self,
        clustered_rows: list[list[_SpatialOCRLine]],
        usable_rows: list[list[_SpatialOCRLine]],
        first_numbered_row: int | None,
    ) -> list[float]:
        header_rows = clustered_rows[:first_numbered_row] if first_numbered_row is not None else []
        header_candidates = [row for row in header_rows if 2 <= len(row) <= 8]
        if header_candidates:
            anchor = max(header_candidates, key=lambda row: (len(row), self._row_width(row)))
            return [line.center_x for line in sorted(anchor, key=lambda line: line.center_x)]

        width_counts = Counter(len(row) for row in usable_rows)
        column_count = max(width_counts, key=lambda width: (width_counts[width], width))
        anchor_rows = [
            sorted(row, key=lambda line: line.center_x)
            for row in usable_rows
            if len(row) == column_count
        ]
        if not anchor_rows:
            return []
        return [median(row[index].center_x for row in anchor_rows) for index in range(column_count)]

    def _row_width(self, row: list[_SpatialOCRLine]) -> float:
        return max(line.bbox[2] for line in row) - min(line.bbox[0] for line in row)

    def _reconstruct_spatial_flowchart(
        self,
        candidate: dict[str, Any],
        raw_text: str,
        question_start: int,
        question_end: int,
        source_element_ids: list[str],
    ) -> dict[str, Any] | None:
        blank_numbers = list(range(question_start, question_end + 1))
        expected = set(blank_numbers)
        flow_lines = self._without_visual_title(candidate["lines"], candidate["bbox"], expected)
        groups = self._cluster_flow_nodes(flow_lines)
        nodes = []
        matched = set()
        for index, group in enumerate(groups, 1):
            ordered = sorted(group, key=lambda line: (line.center_y, line.center_x))
            text = normalize_text(" ".join(line.text for line in ordered))
            numbers = sorted(self._question_numbers(text, expected))
            matched.update(numbers)
            nodes.append(
                {
                    "id": f"node-{index}",
                    "text": text,
                    "question_number": numbers[0] if len(numbers) == 1 else None,
                    "question_numbers": numbers,
                    "bbox": self._union_bbox([line.bbox for line in ordered]),
                }
            )
        if len(nodes) < 2 or not matched:
            return None
        edges, unresolved = self._connector_graph(nodes, candidate.get("connectors") or [], candidate["bbox"])
        missing_questions = sorted(expected - matched)
        quality_issues = []
        if missing_questions:
            quality_issues.append("missing_question_numbers")
        if unresolved:
            quality_issues.append("unresolved_connectors")
        if not candidate.get("connectors"):
            quality_issues.append("no_connector_geometry")
        ocr_confidence = sum(line.confidence for line in flow_lines) / max(1, len(flow_lines))
        connector_coverage = len(edges) / max(1, len(candidate.get("connectors") or []))
        confidence = min(candidate["layout_confidence"], ocr_confidence) * (0.6 + 0.3 * connector_coverage)
        return {
            "type": "flowchart",
            "question_range": [question_start, question_end],
            "nodes": nodes,
            "edges": edges,
            "connectors": candidate.get("connectors") or [],
            "unresolved_connectors": unresolved,
            "blank_question_numbers": blank_numbers,
            "bbox": candidate["bbox"],
            "source": "doclayout_yolo+rapidocr_boxes+raster_connectors",
            "confidence": round(confidence, 4),
            "edge_detection": (
                "raster_arrowheads"
                if edges
                else "connector_geometry_only"
                if candidate.get("connectors")
                else "not_available"
            ),
            "quality_status": "degraded" if quality_issues else "passed",
            "quality_issues": quality_issues,
            "missing_question_numbers": missing_questions,
            "raw_text": raw_text,
            "page_numbers": [candidate["page"]],
            "source_element_ids": source_element_ids,
        }

    def _reconstruct_spatial_diagram(
        self,
        candidate: dict[str, Any],
        raw_text: str,
        question_start: int,
        question_end: int,
        source_element_ids: list[str],
    ) -> dict[str, Any] | None:
        expected = set(range(question_start, question_end + 1))
        lines = self._without_visual_title(candidate["lines"], candidate["bbox"], expected)
        numbered = [line for line in lines if self._question_numbers(line.text, expected)]
        if not numbered:
            return None

        labels = []
        used_context: set[int] = set()
        context_lines = [line for line in lines if not self._question_numbers(line.text, expected)]
        region_width = max(1.0, candidate["bbox"][2] - candidate["bbox"][0])
        region_height = max(1.0, candidate["bbox"][3] - candidate["bbox"][1])
        context_limit = max(region_width, region_height) * 0.16
        for index, line in enumerate(numbered, 1):
            numbers = sorted(self._question_numbers(line.text, expected))
            nearby = sorted(
                (
                    (self._bbox_distance(line.bbox, candidate_line.bbox), context_index, candidate_line)
                    for context_index, candidate_line in enumerate(context_lines)
                    if context_index not in used_context
                ),
                key=lambda item: item[0],
            )
            context = []
            for distance, context_index, candidate_line in nearby[:2]:
                if distance > context_limit:
                    continue
                used_context.add(context_index)
                context.append(candidate_line)
            grouped = [line, *context]
            labels.append(
                {
                    "id": f"label-{index}",
                    "question_number": numbers[0] if len(numbers) == 1 else None,
                    "question_numbers": numbers,
                    "text": normalize_text(" ".join(item.text for item in sorted(grouped, key=lambda item: (item.center_y, item.center_x)))),
                    "bbox": self._union_bbox([item.bbox for item in grouped]),
                }
            )

        matched = {number for label in labels for number in label["question_numbers"]}
        missing = sorted(expected - matched)
        connectors = candidate.get("connectors") or []
        quality_issues = []
        if missing:
            quality_issues.append("missing_question_numbers")
        if not connectors:
            quality_issues.append("no_connector_geometry")
        elif len(connectors) < len(labels):
            quality_issues.append("connector_coverage_low")
        if connectors and all(
            float(connector.get("direction_confidence") or 0.0) < self.direction_min_confidence
            for connector in connectors
        ):
            quality_issues.append("low_confidence_connectors")
        ocr_confidence = sum(line.confidence for line in lines) / max(1, len(lines))
        confidence = min(candidate["layout_confidence"], ocr_confidence) * (0.7 if connectors else 0.55)
        return {
            "type": "diagram",
            "question_range": [question_start, question_end],
            "labels": labels,
            "connectors": connectors,
            "edges": [],
            "blank_question_numbers": list(range(question_start, question_end + 1)),
            "bbox": candidate["bbox"],
            "source": "doclayout_yolo+rapidocr_boxes+raster_connectors",
            "confidence": round(confidence, 4),
            "quality_status": "degraded" if quality_issues else "passed",
            "quality_issues": quality_issues,
            "missing_question_numbers": missing,
            "raw_text": raw_text,
            "page_numbers": [candidate["page"]],
            "source_element_ids": source_element_ids,
        }

    def _without_visual_title(
        self,
        lines: list[_SpatialOCRLine],
        region_bbox: list[float],
        expected: set[int],
    ) -> list[_SpatialOCRLine]:
        region_width = max(1.0, region_bbox[2] - region_bbox[0])
        region_height = max(1.0, region_bbox[3] - region_bbox[1])
        return [
            line
            for line in lines
            if not (
                not self._question_numbers(line.text, expected)
                and line.center_y <= region_bbox[1] + region_height * 0.14
                and (line.bbox[2] - line.bbox[0]) >= region_width * 0.25
            )
        ]

    def _connector_graph(
        self,
        nodes: list[dict[str, Any]],
        connectors: list[dict[str, Any]],
        region_bbox: list[float],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        edges = []
        unresolved = []
        seen: set[tuple[str, str]] = set()
        max_distance = max(region_bbox[2] - region_bbox[0], region_bbox[3] - region_bbox[1]) * 0.16
        spatial_nodes = [node for node in nodes if node.get("bbox")]
        for connector in connectors:
            bbox = _flat_bbox(connector.get("bbox"))
            head = connector.get("arrowhead_point")
            if bbox is None or not isinstance(head, (list, tuple)) or len(head) < 2:
                unresolved.append({"connector_id": connector.get("id"), "reason": "missing_geometry"})
                continue
            endpoints = connector.get("endpoints") or []
            endpoint_nodes = []
            for endpoint in endpoints[:2]:
                if not isinstance(endpoint, (list, tuple)) or len(endpoint) < 2:
                    continue
                nearest_endpoint = min(
                    ((self._point_bbox_distance(endpoint, node["bbox"]), node) for node in spatial_nodes),
                    key=lambda item: item[0],
                    default=None,
                )
                if nearest_endpoint and all(nearest_endpoint[1]["id"] != item[1]["id"] for item in endpoint_nodes):
                    endpoint_nodes.append(nearest_endpoint)
            nearest = endpoint_nodes if len(endpoint_nodes) == 2 else sorted(
                ((self._bbox_distance(bbox, node["bbox"]), node) for node in spatial_nodes),
                key=lambda item: item[0],
            )[:2]
            if len(nearest) < 2 or nearest[1][0] > max_distance:
                unresolved.append({"connector_id": connector.get("id"), "reason": "nodes_not_resolved"})
                continue
            node_pair = [nearest[0][1], nearest[1][1]]
            head_distances = [self._point_bbox_distance(head, node["bbox"]) for node in node_pair]
            target_index = 0 if head_distances[0] <= head_distances[1] else 1
            target = node_pair[target_index]
            source = node_pair[1 - target_index]
            confidence = float(connector.get("direction_confidence") or 0.0)
            if confidence < self.direction_min_confidence:
                unresolved.append(
                    {
                        "connector_id": connector.get("id"),
                        "node_ids": [source["id"], target["id"]],
                        "reason": "direction_confidence_low",
                        "confidence": round(confidence, 4),
                    }
                )
                continue
            key = (source["id"], target["id"])
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "from": source["id"],
                    "to": target["id"],
                    "connector_id": connector.get("id"),
                    "confidence": round(confidence, 4),
                    "evidence": "raster_arrowhead",
                }
            )
        return edges, unresolved

    def _region_connectors(
        self,
        connector_regions: list[dict[str, Any]],
        region_bbox: list[float],
    ) -> list[dict[str, Any]]:
        selected = []
        for connector_region in connector_regions:
            bbox = _flat_bbox(connector_region.get("bbox"))
            if bbox is None or self._bbox_overlap_ratio(bbox, region_bbox) < 0.5:
                continue
            selected.extend(connector_region.get("connectors") or [])
        return selected

    def _bbox_overlap_ratio(self, first: list[float], second: list[float]) -> float:
        width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
        height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
        intersection = width * height
        first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
        second_area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
        return intersection / min(first_area, second_area)

    def _bbox_distance(self, first: list[float], second: list[float]) -> float:
        dx = max(first[0] - second[2], second[0] - first[2], 0.0)
        dy = max(first[1] - second[3], second[1] - first[3], 0.0)
        return (dx * dx + dy * dy) ** 0.5

    def _point_bbox_distance(self, point: list[float] | tuple[float, ...], bbox: list[float]) -> float:
        x, y = float(point[0]), float(point[1])
        dx = max(bbox[0] - x, x - bbox[2], 0.0)
        dy = max(bbox[1] - y, y - bbox[3], 0.0)
        return (dx * dx + dy * dy) ** 0.5

    def _cluster_flow_nodes(self, lines: list[_SpatialOCRLine]) -> list[list[_SpatialOCRLine]]:
        if not lines:
            return []
        median_height = median(line.height for line in lines)
        max_gap = median_height * 1.15
        groups: list[list[_SpatialOCRLine]] = []
        for line in sorted(lines, key=lambda item: (item.bbox[1], item.bbox[0])):
            target = None
            target_gap = None
            for group in groups:
                bbox = self._union_bbox([item.bbox for item in group])
                horizontal_overlap = max(0.0, min(line.bbox[2], bbox[2]) - max(line.bbox[0], bbox[0]))
                overlap_ratio = horizontal_overlap / max(1.0, min(line.bbox[2] - line.bbox[0], bbox[2] - bbox[0]))
                vertical_gap = max(0.0, line.bbox[1] - bbox[3], bbox[1] - line.bbox[3])
                if overlap_ratio >= 0.25 and vertical_gap <= max_gap and (
                    target_gap is None or vertical_gap < target_gap
                ):
                    target = group
                    target_gap = vertical_gap
            if target is None:
                groups.append([line])
            else:
                target.append(line)
        return sorted(groups, key=lambda group: (min(line.bbox[1] for line in group), min(line.bbox[0] for line in group)))

    def _question_numbers(self, text: str, expected: set[int]) -> set[int]:
        return {
            number
            for number in expected
            if re.search(rf"(?<!\d){number}(?!\d)", text)
        }

    def _union_bbox(self, boxes: list[list[float]]) -> list[float]:
        return [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]

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
        ordered_items: list[str] = []
        edge_detection = "text_arrows" if edges else "not_available"
        if not edges:
            ordered_nodes = self._parse_ordered_flow_items(text, blank_numbers)
            if ordered_nodes:
                nodes = ordered_nodes
                ordered_items = [node["id"] for node in nodes]
                edge_detection = "not_present_in_source"
        matched = {
            number
            for node in nodes
            for number in node.get("question_numbers") or (
                [node["question_number"]] if node.get("question_number") else []
            )
        }
        missing = sorted(set(blank_numbers) - matched)
        quality_issues = []
        if missing:
            quality_issues.append("missing_question_numbers")
        if not edges and not ordered_items:
            quality_issues.append("no_explicit_structure")
        confidence = 0.7 if edges else 0.68 if ordered_items and not missing else 0.35
        flowchart = {
            "type": "flowchart",
            "question_range": [question_start, question_end],
            "nodes": nodes,
            "edges": edges,
            "ordered_items": ordered_items,
            "blank_question_numbers": blank_numbers,
            "bbox": [],
            "source": "ielts_question_group_parser",
            "confidence": confidence,
            "edge_detection": edge_detection,
            "quality_status": "degraded" if quality_issues else "passed",
            "quality_issues": quality_issues,
            "missing_question_numbers": missing,
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

    def _parse_ordered_flow_items(
        self,
        text: str,
        blank_numbers: list[int],
    ) -> list[dict[str, Any]]:
        bullet_pattern = re.compile(r"^\s*[•●▪◦*-]\s+(.+)$")
        items = []
        for line in text.splitlines():
            match = bullet_pattern.match(line)
            if not match:
                continue
            item_text = normalize_text(match.group(1))
            if item_text:
                items.append(item_text)
        if len(items) < 2:
            return []
        nodes = []
        for index, item in enumerate(items, 1):
            numbers = [
                number
                for number in blank_numbers
                if re.search(rf"(?<!\d){number}(?!\d)", item)
            ]
            nodes.append(
                {
                    "id": f"node-{index}",
                    "text": item,
                    "question_number": numbers[0] if len(numbers) == 1 else None,
                    "question_numbers": numbers,
                    "sequence_index": index,
                }
            )
        return nodes

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
            label = node.get("text", "")
            numbers = node.get("question_numbers") or (
                [node["question_number"]] if node.get("question_number") else []
            )
            if numbers:
                label = f"{label} [blanks: {', '.join(str(number) for number in numbers)}]".strip()
            lines.append(f"- {node['id']}: {label}")
        for edge in flowchart.get("edges") or []:
            lines.append(f"- edge: {edge['from']} -> {edge['to']}")
        return "\n".join(lines)

    def _diagram_text(self, diagram: dict[str, Any]) -> str:
        lines = [f"Diagram Questions {diagram['question_range'][0]}-{diagram['question_range'][1]}"]
        for label in diagram.get("labels") or []:
            lines.append(f"- {label['id']}: {label.get('text', '')}")
        lines.append(f"- detected connectors: {len(diagram.get('connectors') or [])}")
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
