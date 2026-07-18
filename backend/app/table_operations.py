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


def table_summary_facts(table: dict[str, Any]) -> list[str]:
    columns = table.get("columns") or []
    rows = [row for row in table.get("rows") or [] if isinstance(row, list) and row]
    if len(columns) < 2 or not rows:
        return []

    facts: list[str] = []
    metric_columns: dict[str, list[tuple[int, int, str]]] = {}
    for index, column in enumerate(columns[1:], 1):
        label = str(column)
        year_match = re.search(r"\b(\d{4})\b", label)
        if not year_match:
            continue
        metric = _metric_key(label)
        if metric:
            metric_columns.setdefault(metric, []).append((int(year_match.group(1)), index, label))

    for entries in metric_columns.values():
        if len(entries) < 2:
            continue
        first_year, first_index, first_label = min(entries)
        last_year, last_index, last_label = max(entries)
        if first_year == last_year:
            continue
        changes = []
        paired_values = []
        initial_values = []
        final_values = []
        for row in rows:
            if len(row) <= max(first_index, last_index):
                continue
            first = numeric_value(row[first_index])
            last = numeric_value(row[last_index])
            if first is None or last is None:
                continue
            label = str(row[0])
            changes.append((label, last - first))
            paired_values.append((label, first, last, last - first))
            initial_values.append((label, first))
            final_values.append((label, last))
        if not changes:
            continue
        value_text = "; ".join(
            f"{label} {format_number(first)} -> {format_number(last)} ({format_signed(change)})"
            for label, first, last, change in paired_values
        )
        largest_label, largest_change = max(changes, key=lambda item: item[1])
        highest_label, highest_value = max(final_values, key=lambda item: item[1])
        metric_label = re.sub(r"\s+\d{4}\b.*$", "", first_label).strip() or first_label
        initial_ranking = " > ".join(
            f"{label} {format_number(value)}"
            for label, value in sorted(initial_values, key=lambda item: item[1], reverse=True)
        )
        final_ranking = " > ".join(
            f"{label} {format_number(value)}"
            for label, value in sorted(final_values, key=lambda item: item[1], reverse=True)
        )
        facts.append(
            f"{metric_label}, values from {first_year} to {last_year}: {value_text}. "
            f"Largest increase: {largest_label} ({format_signed(largest_change)})."
        )
        facts.append(
            f"{metric_label} ranking in {first_year}: {initial_ranking}. "
            f"Ranking in {last_year}: {final_ranking}. "
            f"Highest final value in {last_label}: {highest_label} ({format_number(highest_value)})."
        )
    return facts


def numeric_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:[.,]\d+)?", str(value).replace(",", "."))
    return float(match.group(0)) if match else None


def format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def format_signed(value: float) -> str:
    return f"+{format_number(value)}" if value >= 0 else format_number(value)


def _metric_key(label: str) -> str:
    without_year = re.sub(r"\b\d{4}\b", " ", label.lower())
    return " ".join(re.findall(r"[^\W_]+|\d+", without_year, flags=re.UNICODE))


def _terms(text: Any) -> set[str]:
    return {
        term
        for term in re.findall(r"[\w]+", str(text).lower(), flags=re.UNICODE)
        if len(term) > 1 or term.isdigit()
    }
