from __future__ import annotations

import re
from typing import Any


def row_match_score(message: str, row_label: Any) -> float:
    label = str(row_label).strip()
    if not label:
        return 0.0
    lowered = message.lower()
    label_lower = label.lower()
    if re.search(rf"(?<!\w){re.escape(label_lower)}(?!\w)", lowered):
        return 10.0
    return float(len(_terms(message) & _terms(label)))


def column_match_score(message: str, column_label: Any) -> float:
    label = str(column_label).strip()
    if not label:
        return 0.0
    score = float(len(_terms(message) & _terms(label)))
    query_years = set(re.findall(r"\b\d{4}\b", message))
    column_years = set(re.findall(r"\b\d{4}\b", label))
    if query_years and query_years & column_years:
        score += 4.0
    return score


def table_cell_value(message: str, table: dict[str, Any]) -> tuple[float, Any] | None:
    columns = table.get("columns") or []
    rows = table.get("rows") or []
    if len(columns) < 2 or not rows:
        return None
    column_scores = [
        (index, column_match_score(message, column))
        for index, column in enumerate(columns[1:], 1)
    ]
    column_scores = [(index, score) for index, score in column_scores if score > 0]
    if not column_scores:
        return None
    target_index, column_score = max(column_scores, key=lambda item: item[1])
    best: tuple[float, Any] | None = None
    for row in rows:
        if not row or len(row) <= target_index:
            continue
        row_score = row_match_score(message, row[0])
        if row_score <= 0:
            continue
        match = (row_score + column_score, row[target_index])
        if best is None or match[0] > best[0]:
            best = match
    return best


def table_change_calculations(message: str, table: dict[str, Any]) -> dict[str, Any] | None:
    columns = table.get("columns") or []
    years = list(dict.fromkeys(re.findall(r"\b\d{4}\b", message)))
    if len(years) != 2 or len(columns) < 3:
        return None

    selected_columns: list[int] = []
    for year in years:
        candidates = [
            (index, column_match_score(message, column))
            for index, column in enumerate(columns[1:], 1)
            if year in str(column)
        ]
        if not candidates:
            return None
        selected_columns.append(max(candidates, key=lambda item: item[1])[0])
    if selected_columns[0] == selected_columns[1]:
        return None

    calculations: list[dict[str, Any]] = []
    for row in table.get("rows") or []:
        if not isinstance(row, list) or len(row) <= max(selected_columns):
            continue
        first = numeric_value(row[selected_columns[0]])
        second = numeric_value(row[selected_columns[1]])
        if first is None or second is None:
            continue
        calculations.append(
            {
                "label": str(row[0]),
                "first": first,
                "second": second,
                "change": second - first,
            }
        )
    if not calculations:
        return None
    asks_decrease = "giảm" in message.lower() or "decrease" in message.lower()
    winner = min(calculations, key=lambda item: item["change"]) if asks_decrease else max(
        calculations,
        key=lambda item: item["change"],
    )
    return {
        "years": years,
        "calculations": calculations,
        "winner": winner,
        "direction": "decrease" if asks_decrease else "increase",
    }


def comparison_row(message: str, table: dict[str, Any]) -> list[Any] | None:
    rows = table.get("rows") or []
    matching = [row for row in rows if row and row_match_score(message, row[0]) > 0]
    return max(matching, key=lambda row: row_match_score(message, row[0])) if matching else None


def numeric_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:[.,]\d+)?", str(value).replace(",", "."))
    return float(match.group(0)) if match else None


def format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _terms(text: Any) -> set[str]:
    return {
        term
        for term in re.findall(r"[\w]+", str(text).lower(), flags=re.UNICODE)
        if len(term) > 1 or term.isdigit()
    }
