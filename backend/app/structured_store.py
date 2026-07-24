import re
from typing import Callable, Dict, List

from .config import settings


def _append_unique(values: list[str], value: object) -> None:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    if normalized and normalized.casefold() not in {item.casefold() for item in values}:
        values.append(normalized)


def _target_descriptor(text: object, sentence_count: int = 1) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    sentence_ends = list(re.finditer(r"(?<=[.!?])(?:\s|$)", normalized))
    sentence_end = sentence_ends[min(sentence_count, len(sentence_ends)) - 1] if sentence_ends else None
    descriptor = normalized[: sentence_end.start()].strip() if sentence_end else normalized
    limit = settings.target_descriptor_chars
    if len(descriptor) > limit:
        descriptor = descriptor[: limit - 3].rstrip() + "..."
    return descriptor


class StructuredDocumentStore:
    """Schema-first lookup over indexed document chunks.

    This store reads structured chunk metadata only. It does not embed text and
    does not depend on vector similarity, so exact question/table/passage
    lookups stay separate from semantic retrieval.
    """

    def __init__(self, docs_provider: Callable[[], List[Dict]]) -> None:
        self._docs_provider = docs_provider

    @property
    def docs(self) -> List[Dict]:
        return self._docs_provider()

    def overview(self, top_k: int = 8, document_ids: List[str] | None = None) -> List[Dict]:
        top_k = max(1, min(top_k, 50))
        docs = self._docs_in_scope(document_ids)
        outline_docs = [
            doc
            for doc in docs
            if doc.get("metadata", {}).get("unit_type")
            in {
                "document_outline",
                "passage_summary",
                "writing_prompt",
                "writing_table",
                "writing_task",
                "sample_answer",
            }
        ]
        if outline_docs:
            results = []
            for doc in sorted(outline_docs, key=lambda item: item.get("chunk_index", 0))[:top_k]:
                item = self._mark_retrieval(doc, "overview", 1.0)
                item["overview_score"] = 1.0
                results.append(item)
            passage_docs = [
                doc
                for doc in sorted(docs, key=lambda item: item.get("chunk_index", 0))
                if doc.get("metadata", {}).get("unit_type") == "passage"
            ]
            seen_passages = set()
            for doc in passage_docs:
                if len(results) >= top_k:
                    break
                passage_number = doc.get("metadata", {}).get("passage_number")
                passage_key = (doc.get("document_id"), passage_number)
                if passage_key in seen_passages:
                    continue
                seen_passages.add(passage_key)
                item = self._mark_retrieval(doc, "overview_passage", 0.8)
                item["overview_score"] = 0.8
                results.append(item)
            return results

        docs = sorted(docs, key=lambda doc: (min(doc.get("pages") or [999]), doc.get("chunk_index", 0)))
        content_docs = [doc for doc in docs if doc.get("token_count", 0) >= 80]
        selected = content_docs[:top_k] if content_docs else docs[:top_k]
        results = []
        for doc in selected:
            item = self._mark_retrieval(doc, "overview_fallback", 1.0)
            item["overview_score"] = 1.0
            results.append(item)
        return results

    def structured_lookup(
        self,
        query: str,
        intent: str,
        top_k: int = 8,
        document_ids: List[str] | None = None,
    ) -> List[Dict]:
        top_k = max(1, min(top_k, 50))
        if intent == "document_overview":
            return self.overview(top_k=top_k, document_ids=document_ids)

        ranges = self._question_ranges(query)
        if intent == "table_cell":
            table_cell_hits = self._table_cell_lookup(query, top_k=top_k, document_ids=document_ids)
            if table_cell_hits:
                return table_cell_hits
        if intent in {"show_table", "extract_table", "table_calculation", "table_comparison"}:
            return self._visual_lookup(
                ranges=ranges,
                visual_unit_type="table",
                question_type="table_completion",
                top_k=top_k,
                prefer_writing=self._looks_like_writing_reference(query),
                document_ids=document_ids,
            )
        if intent == "show_flowchart":
            return self._visual_lookup(
                ranges=ranges,
                visual_unit_type="flowchart",
                question_type="flowchart_completion",
                top_k=top_k,
                prefer_writing=False,
                document_ids=document_ids,
            )
        if intent == "show_diagram":
            return self._visual_lookup(
                ranges=ranges,
                visual_unit_type="diagram",
                question_type="diagram_completion",
                top_k=top_k,
                prefer_writing=False,
                document_ids=document_ids,
            )
        if intent == "show_writing_prompt":
            return self._unit_lookup({"writing_prompt", "writing_task"}, top_k, document_ids)
        if intent == "writing_generation":
            return self._unit_lookup({"writing_prompt", "writing_table"}, top_k, document_ids)
        if ranges and intent in {"show_questions", "translate_questions", "explain_questions", "solve_questions"}:
            return self._question_lookup(ranges, top_k=top_k, document_ids=document_ids)
        return []

    def document_catalog(self, document_ids: List[str] | None = None) -> List[Dict]:
        catalog: dict[str, Dict] = {}
        for doc in self._docs_in_scope(document_ids):
            source_file = doc.get("source_file", "unknown")
            document_id = str(doc.get("document_id") or source_file)
            entry = catalog.setdefault(
                document_id,
                {
                    "source_file": source_file,
                    "chunks": 0,
                    "pages": set(),
                    "document_ids": set(),
                    "mime_types": set(),
                    "unit_types": set(),
                    "passage_numbers": set(),
                    "document_types": set(),
                    "task_types": set(),
                    "section_titles": set(),
                    "visual_types": set(),
                    "table_columns": [],
                    "target_descriptors": [],
                    "untitled_writing_parents": set(),
                    "sample_descriptors": {},
                },
            )
            entry["chunks"] += 1
            entry["pages"].update(doc.get("pages") or [])
            if doc.get("document_id"):
                entry["document_ids"].add(doc["document_id"])
            metadata = doc.get("metadata", {})
            for metadata_key, catalog_key in [
                ("mime_type", "mime_types"),
                ("unit_type", "unit_types"),
                ("passage_number", "passage_numbers"),
                ("document_type", "document_types"),
                ("task_type", "task_types"),
            ]:
                value = metadata.get(metadata_key)
                if value:
                    entry[catalog_key].add(value)
            for title_key in ("passage_title", "task_title"):
                title = metadata.get(title_key)
                if title:
                    entry["section_titles"].add(str(title))
            visual_type = metadata.get("visual_type")
            if visual_type:
                entry["visual_types"].add(str(visual_type))
            table = metadata.get("table")
            if isinstance(table, dict):
                table_type = table.get("type")
                if table_type:
                    entry["visual_types"].add(str(table_type))
                for column in table.get("columns") or []:
                    _append_unique(entry["table_columns"], column)

            unit_type = metadata.get("unit_type")
            parent_id = str(metadata.get("parent_id") or "")
            descriptor = _target_descriptor(
                doc.get("display_text") or doc.get("text"),
                sentence_count=2 if unit_type == "writing_prompt" else 1,
            )
            if unit_type == "writing_prompt" and descriptor:
                _append_unique(entry["target_descriptors"], descriptor)
            elif unit_type == "writing_task" and not metadata.get("task_title"):
                if descriptor:
                    _append_unique(entry["target_descriptors"], descriptor)
                if parent_id:
                    entry["untitled_writing_parents"].add(parent_id)
            elif unit_type == "sample_answer" and not metadata.get("task_title") and parent_id:
                if descriptor:
                    entry["sample_descriptors"].setdefault(parent_id, descriptor)

        results = []
        for item in catalog.values():
            for parent_id in item["untitled_writing_parents"]:
                descriptor = item["sample_descriptors"].get(parent_id)
                if descriptor:
                    _append_unique(item["target_descriptors"], descriptor)
            results.append({
                "source_file": item["source_file"],
                "chunks": item["chunks"],
                "pages": sorted(item["pages"]),
                "document_ids": sorted(item["document_ids"]),
                "mime_types": sorted(item["mime_types"]),
                "unit_types": sorted(item["unit_types"]),
                "passage_numbers": sorted(item["passage_numbers"]),
                "document_types": sorted(item["document_types"]),
                "task_types": sorted(item["task_types"]),
                "section_titles": sorted(item["section_titles"]),
                "visual_types": sorted(item["visual_types"]),
                "table_columns": item["table_columns"],
                "target_descriptors": item["target_descriptors"],
            })
        return results

    def question_context_for_sources(
        self,
        sources: List[Dict],
        top_k: int = 8,
        document_ids: List[str] | None = None,
    ) -> List[Dict]:
        wanted_parent_ids = {
            (source.get("document_id"), source.get("metadata", {}).get("parent_id"))
            for source in sources
            if source.get("metadata", {}).get("unit_type") == "question"
        }
        wanted_ranges = {
            (source.get("document_id"), tuple(source.get("metadata", {}).get("question_range") or []))
            for source in sources
            if source.get("metadata", {}).get("unit_type") == "question_group"
        }
        wanted_ranges = {item for item in wanted_ranges if len(item[1]) == 2}
        if not wanted_parent_ids and not wanted_ranges:
            return []

        results = []
        for doc in sorted(self._docs_in_scope(document_ids), key=lambda item: item.get("chunk_index", 0)):
            metadata = doc.get("metadata", {})
            unit_type = metadata.get("unit_type")
            question_range = metadata.get("question_range")
            document_id = doc.get("document_id")
            if unit_type == "question_group":
                chunk_id = doc.get("chunk_id")
                if (document_id, chunk_id) in wanted_parent_ids or (
                    document_id,
                    tuple(question_range or []),
                ) in wanted_ranges:
                    results.append(self._mark_retrieval(doc, "parent_question_group", 1.0))
            elif unit_type == "question" and (document_id, tuple(question_range or [])) in wanted_ranges:
                results.append(self._mark_retrieval(doc, "child_question", 0.8))
            if len(results) >= top_k:
                break
        return results

    def passage_context_for_sources(
        self,
        sources: List[Dict],
        max_chunks_per_passage: int = 3,
        document_ids: List[str] | None = None,
    ) -> List[Dict]:
        passage_keys = {
            (source.get("document_id"), source.get("metadata", {}).get("passage_number"))
            for source in sources
            if source.get("metadata", {}).get("passage_number")
        }
        if not passage_keys:
            return []

        results: list[Dict] = []
        counts: dict[tuple[str | None, int], int] = {}
        for doc in sorted(self._docs_in_scope(document_ids), key=lambda item: item.get("chunk_index", 0)):
            metadata = doc.get("metadata", {})
            passage_number = metadata.get("passage_number")
            passage_key = (doc.get("document_id"), passage_number)
            if metadata.get("unit_type") != "passage" or passage_key not in passage_keys:
                continue
            count = counts.get(passage_key, 0)
            if count >= max_chunks_per_passage:
                continue
            results.append(self._mark_retrieval(doc, "parent_passage", 1.0))
            counts[passage_key] = count + 1
        return results

    def writing_context_for_sources(
        self,
        sources: List[Dict],
        top_k: int = 4,
        document_ids: List[str] | None = None,
    ) -> List[Dict]:
        target = next(
            (
                source
                for source in sources
                if source.get("metadata", {}).get("unit_type") in {"writing_task", "sample_answer"}
                and source.get("metadata", {}).get("parent_id")
            ),
            None,
        )
        if target is None:
            return []

        document_id = target.get("document_id")
        parent_id = target.get("metadata", {}).get("parent_id")
        results = []
        for doc in sorted(self._docs_in_scope(document_ids), key=lambda item: item.get("chunk_index", 0)):
            metadata = doc.get("metadata", {})
            if doc.get("document_id") != document_id or metadata.get("parent_id") != parent_id:
                continue
            if metadata.get("unit_type") not in {"writing_task", "sample_answer"}:
                continue
            results.append(self._mark_retrieval(doc, "writing_parent", 1.0))
            if len(results) >= top_k:
                break
        return results

    def _question_lookup(
        self,
        ranges: List[tuple[int, int]],
        top_k: int,
        document_ids: List[str] | None,
    ) -> List[Dict]:
        scored = []
        for doc in self._docs_in_scope(document_ids):
            metadata = doc.get("metadata", {})
            unit_type = metadata.get("unit_type")
            if unit_type not in {"question_group", "question"}:
                continue
            question_range = metadata.get("question_range")
            if not isinstance(question_range, list) or len(question_range) != 2:
                continue
            chunk_start, chunk_end = int(question_range[0]), int(question_range[1])
            score = 0.0
            for start, end in ranges:
                if not self._ranges_overlap(start, end, chunk_start, chunk_end):
                    continue
                exact = start == chunk_start and end == chunk_end
                contains = chunk_start <= start and end <= chunk_end
                overlap = min(end, chunk_end) - max(start, chunk_start) + 1
                score += overlap
                score += 100.0 if exact else 70.0 if contains else 30.0
                score += 15.0 if unit_type == "question_group" else 8.0
            if score > 0:
                scored.append(self._mark_retrieval(doc, "structured_question", score))
        return sorted(scored, key=lambda item: item["structured_score"], reverse=True)[:top_k]

    def _visual_lookup(
        self,
        ranges: List[tuple[int, int]],
        visual_unit_type: str,
        question_type: str,
        top_k: int,
        prefer_writing: bool,
        document_ids: List[str] | None,
    ) -> List[Dict]:
        scored = []
        for doc in self._docs_in_scope(document_ids):
            metadata = doc.get("metadata", {})
            unit_type = metadata.get("unit_type")
            is_visual = unit_type == visual_unit_type or metadata.get("question_type") == question_type
            is_writing_table = visual_unit_type == "table" and unit_type in {"writing_table", "table_row"}
            if not is_visual and not is_writing_table:
                continue
            if prefer_writing and not is_writing_table and metadata.get("document_type") != "ielts_writing_task_1":
                continue

            score = 40.0 if is_visual else 20.0
            question_range = metadata.get("question_range")
            if ranges:
                if not isinstance(question_range, list) or len(question_range) != 2:
                    continue
                chunk_start, chunk_end = int(question_range[0]), int(question_range[1])
                if not any(self._ranges_overlap(start, end, chunk_start, chunk_end) for start, end in ranges):
                    continue
                score += 60.0
            if metadata.get(visual_unit_type):
                score += 20.0
            if is_writing_table:
                score += 20.0
            scored.append(self._mark_retrieval(doc, f"structured_{visual_unit_type}", score))
        return sorted(scored, key=lambda item: item["structured_score"], reverse=True)[:top_k]

    def _table_cell_lookup(
        self,
        query: str,
        top_k: int,
        document_ids: List[str] | None,
    ) -> List[Dict]:
        scored = []
        for doc in self._docs_in_scope(document_ids):
            metadata = doc.get("metadata", {})
            table = metadata.get("table")
            columns = table.get("columns") if isinstance(table, dict) else metadata.get("table_columns")
            rows = table.get("rows") if isinstance(table, dict) else [metadata.get("table_row")]
            if not columns or not rows or len(columns) < 2:
                continue
            best = 0.0
            for row in rows:
                if not isinstance(row, list) or not row:
                    continue
                row_score = self._row_match_score(query, row[0])
                if row_score <= 0:
                    continue
                column_score = max((self._column_match_score(query, column) for column in columns[1:]), default=0.0)
                if column_score <= 0:
                    continue
                best = max(best, row_score + column_score)
            if best > 0:
                scored.append(self._mark_retrieval(doc, "structured_table_cell", best))
        return sorted(scored, key=lambda item: item["structured_score"], reverse=True)[:top_k]

    def _unit_lookup(
        self,
        unit_types: set[str],
        top_k: int,
        document_ids: List[str] | None,
    ) -> List[Dict]:
        results = [
            self._mark_retrieval(doc, "structured_unit", 1.0)
            for doc in sorted(self._docs_in_scope(document_ids), key=lambda item: item.get("chunk_index", 0))
            if doc.get("metadata", {}).get("unit_type") in unit_types
        ]
        return results[:top_k]

    def _docs_in_scope(self, document_ids: List[str] | None) -> List[Dict]:
        if document_ids is None:
            return self.docs
        if not document_ids:
            return []
        allowed = set(document_ids)
        return [doc for doc in self.docs if doc.get("document_id") in allowed]

    def _question_ranges(self, query: str) -> List[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        normalized = query.lower().replace("đến", "to").replace("tới", "to")
        for match in re.finditer(
            r"(?:questions?|question|câu hỏi|câu)\s*(?:từ\s+)?(\d{1,2})"
            r"(?:\s*(?:-|–|to)\s*(?:questions?|question|câu hỏi|câu)?\s*(\d{1,2}))?",
            normalized,
        ):
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if start > end:
                start, end = end, start
            ranges.append((start, end))
        return ranges

    def _lookup_terms(self, text: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[\w]+", str(text).lower(), flags=re.UNICODE)
            if len(term) > 1 or term.isdigit()
        }

    def _row_match_score(self, query: str, row_label: object) -> float:
        label = str(row_label).strip()
        if not label:
            return 0.0
        if re.search(rf"(?<!\w){re.escape(label.lower())}(?!\w)", query.lower()):
            return 10.0
        return float(len(self._lookup_terms(query) & self._lookup_terms(label)))

    def _column_match_score(self, query: str, column_label: object) -> float:
        score = float(len(self._lookup_terms(query) & self._lookup_terms(str(column_label))))
        query_years = set(re.findall(r"\b\d{4}\b", query))
        column_years = set(re.findall(r"\b\d{4}\b", str(column_label)))
        if query_years and query_years & column_years:
            score += 4.0
        return score

    def _looks_like_writing_reference(self, query: str) -> bool:
        lowered = query.lower()
        return any(marker in lowered for marker in ["writing", "ảnh", "hình", "image", "task 1"])

    def _ranges_overlap(self, a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return max(a_start, b_start) <= min(a_end, b_end)

    def _mark_retrieval(self, doc: Dict, method: str, score: float) -> Dict:
        item = dict(doc)
        item["score"] = float(item.get("score", 0.0))
        item["retrieval_method"] = method
        item["structured_score"] = float(score)
        return item
