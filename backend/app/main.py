import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import settings
from .document_scope import DocumentScope, apply_document_affinity, resolve_document_scope
from .document_pipeline import DocumentProcessor
from .intent import (
    dedupe_sources,
    filter_sources_for_intent,
    has_explicit_no_solution_constraint,
    semantic_intent_decision,
)
from .llm import (
    OLLAMA_MODEL,
    OLLAMA_NUM_PREDICT,
    OllamaRequestError,
    RouteGatewayDecision,
    classify_rag_intent,
    direct_answer_prompt,
    query_ollama,
    rag_prompt,
    response_output_contract,
    response_output_issues,
    response_output_penalty,
    response_retry_prompt,
    resolve_rag_target,
    classify_chat_route,
    select_best_writing_output,
    stream_ollama,
    writing_output_contract,
    writing_output_issues,
    writing_output_penalty,
    writing_retry_prompt,
)
from .rag import get_store
from .schemas import (
    ChatAffinity,
    ChatConversationState,
    ChatRequest,
    ChatResponse,
    ChatUserProfile,
    SearchRequest,
    SearchResponse,
    StatsResponse,
    UploadResponse,
)
from .table_operations import (
    comparison_row_facts,
    comparison_row,
    format_number,
    table_cell_value,
    table_change_calculations,
    table_summary_facts,
)


logger = logging.getLogger(__name__)

UPLOAD_DIR = settings.upload_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENT_PROCESSOR = DocumentProcessor()
LAST_WARMUP_STATUS: dict[str, Any] | None = None

app = FastAPI(title="Standalone IELTS Chatbot", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ollama_failure_detail(exc: Exception) -> dict[str, Any]:
    diagnostic = (
        exc.debug_detail()
        if isinstance(exc, OllamaRequestError)
        else {"kind": type(exc).__name__, "message": str(exc)[:500]}
    )
    return {
        "message": "Không thể kết nối hoặc nhận câu trả lời từ Ollama.",
        "ollama": diagnostic,
    }


def stream_event(event_type: str, **payload: Any) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


def format_context(sources: list[dict], max_chars_per_source: int | None = None) -> str:
    parts = []
    for index, source in enumerate(sources, 1):
        source_file = source.get("source_file", "unknown")
        pages = source.get("pages") or []
        text = source.get("display_text") or source.get("text", "")
        if max_chars_per_source and len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rsplit(" ", 1)[0] + " ..."
        unit_type = source.get("metadata", {}).get("unit_type")
        role = {
            "question_group": "QUESTION INSTRUCTIONS",
            "question": "QUESTION",
            "passage": "PASSAGE EVIDENCE",
            "document_outline": "DOCUMENT OUTLINE",
            "writing_prompt": "WRITING PROMPT",
            "writing_task": "WRITING TASK",
            "sample_answer": "SAMPLE ANSWER",
            "writing_table": "STRUCTURED TABLE",
            "table": "STRUCTURED TABLE",
            "flowchart": "STRUCTURED FLOWCHART",
            "diagram": "STRUCTURED DIAGRAM",
        }.get(unit_type, "DOCUMENT CONTEXT")
        parts.append(
            f"--- {role} {index} ---\n"
            f"File: {source_file}\n"
            f"Pages: {', '.join(str(page) for page in pages) if pages else 'unknown'}\n"
            f"{text}"
        )
    return "\n\n".join(parts)


@dataclass(frozen=True)
class DocumentCatalogContext:
    text: str
    document_refs: dict[str, str]


def format_document_catalog_context(catalog: list[dict]) -> DocumentCatalogContext:
    lines: list[str] = []
    document_refs: dict[str, str] = {}
    if catalog:
        lines.append("Available uploaded documents:")
        for index, item in enumerate(catalog, 1):
            document_id = next(iter(item.get("document_ids") or []), "")
            if not document_id:
                continue
            reference = f"D{index}"
            document_refs[reference] = document_id
            lines.append(
                f"- {reference}: {item.get('source_file', 'unknown')} | "
                f"document_types={item.get('document_types') or []} | "
                f"task_types={item.get('task_types') or []} | "
                f"unit_types={item.get('unit_types') or []} | "
                f"section_titles={item.get('section_titles') or []}"
            )
    else:
        lines.append("Available uploaded documents: none")

    return DocumentCatalogContext(
        text="\n".join(lines),
        document_refs=document_refs,
    )


def evidence_query_for_sources(sources: list[dict[str, Any]], fallback: str) -> str:
    question_sources = [
        source
        for source in sources
        if source.get("metadata", {}).get("unit_type") == "question"
    ]
    candidates = question_sources or [
        source
        for source in sources
        if source.get("metadata", {}).get("unit_type") == "question_group"
    ]
    queries: list[str] = []
    for source in candidates:
        text = (source.get("display_text") or source.get("text") or "").strip()
        if not text:
            continue
        text = re.sub(r"^\s*\d{1,2}\s*[.)]\s*", "", text)
        option_matches = list(re.finditer(r"(?<![A-Za-z0-9])([A-H])(?=\s+\S)", text))
        if len(option_matches) >= 2:
            text = text[: option_matches[0].start()].strip()
        if text and text not in queries:
            queries.append(text)
    return " ".join(queries).strip() or fallback


def compact_probe_debug(probe: dict) -> dict:
    return {
        "has_hits": probe.get("has_hits", False),
        "has_strong_hits": probe.get("has_strong_hits", False),
        "top_score": probe.get("top_score", 0.0),
        "top_fused_score": probe.get("top_fused_score", 0.0),
        "top_keyword_score": probe.get("top_keyword_score", 0.0),
        "top_question_score": probe.get("top_question_score", 0.0),
        "top_overview_score": probe.get("top_overview_score", 0.0),
        "has_document_intent": probe.get("has_document_intent", False),
        "is_overview": probe.get("is_overview", False),
        "results": [
            {
                "source_file": item.get("source_file"),
                "pages": item.get("pages"),
                "score": item.get("score", 0.0),
                "dense": item.get("probe_dense_score", 0.0),
                "keyword": item.get("probe_keyword_score", 0.0),
                "question": item.get("probe_question_score", 0.0),
                "overview": item.get("probe_overview_score", 0.0),
                "fused": item.get("rrf_score", 0.0),
                "methods": item.get("retrieval_methods", []),
                "chunk_id": item.get("chunk_id"),
                "unit_type": item.get("metadata", {}).get("unit_type"),
                "chunk_reason": item.get("metadata", {}).get("chunk_reason"),
                "passage_number": item.get("metadata", {}).get("passage_number"),
                "question_range": item.get("metadata", {}).get("question_range"),
                "parent_id": item.get("metadata", {}).get("parent_id"),
                "text_preview": " ".join(
                    (item.get("display_text") or item.get("text") or "").split()
                )[:220],
            }
            for item in (probe.get("results") or [])[:3]
        ],
    }


NO_RAG_MATCH_RESPONSE = (
    "Mình chưa tìm thấy nội dung phù hợp trong tài liệu đã upload để trả lời câu hỏi này. "
    "Bạn có thể hỏi rõ hơn theo tên bài, số trang, hoặc upload lại tài liệu nếu phần đó nằm trong bảng/ảnh chưa được trích xuất tốt."
)

AMBIGUOUS_DOCUMENT_RESPONSE = (
    "Mình chưa xác định được bạn đang hỏi tài liệu nào vì có nhiều file phù hợp. "
    "Vui lòng nêu tên file hoặc đính kèm lại đúng tài liệu cần hỏi."
)

INCOMPLETE_QUESTION_RESPONSE = (
    "Mình đã tìm thấy câu hỏi nhưng phần lựa chọn hoặc dữ liệu cần thiết để giải chưa được "
    "trích xuất đầy đủ. Vì vậy mình chưa thể xác định đáp án đáng tin cậy từ tài liệu hiện có."
)

ROUTE_UNDETERMINED_RESPONSE = (
    "Mình chưa xác định chắc chắn câu hỏi này có cần dùng tài liệu đã tải lên hay không. "
    "Vui lòng nói rõ bạn muốn hỏi kiến thức chung hay nội dung trong một tài liệu cụ thể."
)

INTENT_UNDETERMINED_RESPONSE = (
    "Mình chưa xác định được thao tác bạn muốn thực hiện với tài liệu. "
    "Vui lòng nói rõ bạn muốn xem, dịch, giải thích, trả lời câu hỏi hay phân tích nội dung."
)


def document_extraction_failure_detail(document: Any) -> str:
    metadata = document.metadata or {}
    ocr_engine = metadata.get("ocr_engine")
    ocr_metadata = metadata.get("ocr_metadata") or {}
    attempts = ocr_metadata.get("cascade_attempts") or []
    if not attempts and isinstance(ocr_metadata.get("attempt"), dict):
        attempts = [ocr_metadata["attempt"]]
    if not attempts and ocr_metadata.get("error"):
        attempts = [ocr_metadata]
    errors = [
        str(attempt.get("error"))
        for attempt in attempts
        if isinstance(attempt, dict) and attempt.get("error")
    ]
    reasons = [
        str(attempt.get("engine") or attempt.get("reason"))
        for attempt in attempts
        if isinstance(attempt, dict) and (attempt.get("engine") or attempt.get("reason"))
    ]

    if ocr_engine == "rapidocr_failed":
        if any("rapidocr_unavailable" in reason for reason in reasons):
            return (
                "RapidOCR chưa khả dụng trong môi trường backend hiện tại, nên ảnh chưa được OCR. "
                "Hãy cài đúng rapidocr/torch CUDA rồi restart backend."
            )
        if errors:
            return f"RapidOCR không trích xuất được ảnh. Lỗi OCR đầu tiên: {errors[0][:300]}"

    return "Không trích xuất được văn bản từ tài liệu. File có thể quá mờ, không có chữ, hoặc OCR chưa phù hợp."


def generation_fallback(prepared: "ChatPreparation") -> str:
    if prepared.route_used.startswith("vector_rag"):
        return NO_RAG_MATCH_RESPONSE
    return "Mình chưa nhận được nội dung trả lời từ model. Vui lòng thử lại."


def generation_temperature(prepared: "ChatPreparation") -> float:
    if is_writing_response(prepared):
        return 0.1
    return 0.2 if prepared.route_used.startswith("vector_rag") else 0.7


async def generate_answer(prepared: "ChatPreparation", message: str) -> str:
    prompt = prepared.prompt or ""
    answer = await query_ollama(prompt, temperature=generation_temperature(prepared))

    if is_writing_response(prepared):
        contract = writing_output_contract(message)
        issues = writing_output_issues(answer, contract)
        generation_debug = prepared.debug.setdefault("generation", {})
        generation_debug["writing_contract"] = {
            "language": contract.language,
            "min_words": contract.min_words,
            "max_words": contract.max_words,
            "target_words": list(contract.target_words) if contract.target_words else None,
            "single_paragraph": contract.single_paragraph,
            "overview_only": contract.overview_only,
            "first_draft_issues": issues,
        }
        if issues:
            retry = await query_ollama(
                writing_retry_prompt(prompt, contract),
                temperature=0.1,
            )
            selected = select_best_writing_output(answer, retry, contract)
            generation_debug["retry_used"] = True
            generation_debug["candidate_penalties"] = {
                "first": list(writing_output_penalty(answer, contract)),
                "retry": list(writing_output_penalty(retry, contract)),
            }
            generation_debug["selected_candidate"] = "first" if selected == answer else "retry"
            answer = selected
        else:
            generation_debug["retry_used"] = False
        generation_debug["final_issues"] = writing_output_issues(answer, contract)
    else:
        allow_solution = bool(prepared.debug.get("intent_decision", {}).get("allow_solution", False))
        contract = response_output_contract(
            message,
            prepared.query_intent,
            allow_solution=allow_solution,
            writing_context=is_writing_response(prepared),
        )
        issues = response_output_issues(answer, contract)
        generation_debug = prepared.debug.setdefault("generation", {})
        generation_debug["response_contract"] = {
            "language": contract.language,
            "forbid_solution": contract.forbid_solution,
            "required_question_numbers": list(contract.required_question_numbers),
            "first_draft_issues": issues,
        }
        should_retry = bool(issues) and (
            prepared.query_intent == "translate_questions"
            or any("malformed Markdown table" in issue for issue in issues)
            or any("conversation role prefix" in issue for issue in issues)
            or (
                has_explicit_no_solution_constraint(message)
                and contract.forbid_solution
            )
        )
        if should_retry:
            retry = await query_ollama(
                response_retry_prompt(prompt, contract),
                temperature=0.1,
            )
            selected = min(
                (answer, retry),
                key=lambda text: response_output_penalty(text, contract),
            )
            generation_debug["retry_used"] = True
            generation_debug["candidate_penalties"] = {
                "first": list(response_output_penalty(answer, contract)),
                "retry": list(response_output_penalty(retry, contract)),
            }
            generation_debug["selected_candidate"] = "first" if selected == answer else "retry"
            answer = selected
        else:
            generation_debug["retry_used"] = False
        final_issues = response_output_issues(answer, contract)
        if contract.forbid_solution and any("reveals or narrows" in issue for issue in final_issues):
            answer = (
                "Hãy đối chiếu từng câu với đúng đoạn liên quan, xác định từ khóa và điều kiện trong "
                "hướng dẫn, nhưng chưa chọn hoặc loại trừ bất kỳ đáp án nào."
            )
            generation_debug["safe_fallback_used"] = True
            final_issues = response_output_issues(answer, contract)
        generation_debug["final_issues"] = final_issues
    return answer


def requires_reviewed_generation(prepared: "ChatPreparation", message: str) -> bool:
    return (
        prepared.route_used == "base_model"
        or is_writing_response(prepared)
        or prepared.query_intent == "translate_questions"
        or (
            has_explicit_no_solution_constraint(message)
            and not prepared.debug.get("intent_decision", {}).get("allow_solution", False)
        )
    )


def is_writing_response(prepared: "ChatPreparation") -> bool:
    return prepared.query_intent == "writing_generation" or bool(
        prepared.debug.get("retrieval", {}).get("writing_parent_id")
    )


def response_chunks(text: str, size: int = 180) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)] or [""]


def conversation_state_for_result(
    req: ChatRequest,
    prepared: "ChatPreparation",
) -> ChatConversationState:
    previous_affinity = effective_affinity(req) or ChatAffinity()
    if prepared.route_used == "base_model":
        route = "direct"
        affinity = previous_affinity
    elif prepared.route_used == "vector_rag_ambiguous_document":
        route = "clarify"
        affinity = previous_affinity
    elif prepared.route_used in {"vector_rag_no_match", "route_undetermined", "intent_undetermined"}:
        route = "no_match"
        affinity = previous_affinity
    else:
        route = "rag"
        document_ids = list(
            dict.fromkeys(
                str(source.get("document_id"))
                for source in prepared.sources
                if source.get("document_id")
            )
        )
        passage_numbers = sorted(
            {
                int(source.get("metadata", {}).get("passage_number"))
                for source in prepared.sources
                if source.get("metadata", {}).get("passage_number") is not None
            }
        )
        question_ranges = []
        for source in prepared.sources:
            values = source.get("metadata", {}).get("question_range")
            if isinstance(values, list) and len(values) == 2 and values not in question_ranges:
                question_ranges.append([int(values[0]), int(values[1])])
        affinity = ChatAffinity(
            document_ids=document_ids or previous_affinity.document_ids,
            passage_numbers=passage_numbers,
            question_ranges=question_ranges,
        )
    state = ChatConversationState(
        last_route=route,
        last_intent=prepared.query_intent,
        rag_affinity=affinity,
        user_profile=user_profile_for_request(req),
    )
    prepared.debug["conversation_state"] = {
        "input": req.conversation_state.model_dump() if req.conversation_state else None,
        "output": state.model_dump(),
    }
    return state


def _markdown_table(table: dict[str, Any]) -> str:
    columns = table.get("columns") or []
    rows = table.get("rows") or []
    if not columns or not rows:
        return ""
    header = "| " + " | ".join(str(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = list(row) + [""] * max(0, len(columns) - len(row))
        body.append("| " + " | ".join(str(cell) for cell in cells[: len(columns)]) + " |")
    return "\n".join([header, separator, *body])


def _visual_incomplete_text(visual: dict[str, Any], source: dict[str, Any]) -> str:
    visual_type = visual.get("type", "visual")
    question_range = visual.get("question_range") or []
    range_label = f" Questions {question_range[0]}-{question_range[1]}" if len(question_range) == 2 else ""
    blanks = ", ".join(str(number) for number in visual.get("blank_question_numbers") or [])
    raw_text = visual.get("raw_text") or ""
    return (
        f"Mình đã nhận diện được {visual_type}{range_label}, nhưng chưa trích xuất đủ cấu trúc hàng/cột hoặc node/edge đáng tin cậy.\n\n"
        + (f"Các ô/câu trống nhận diện được: {blanks}.\n\n" if blanks else "")
        + (f"Nội dung OCR/native liên quan:\n{raw_text}\n\n" if raw_text else "")
        + f"Nguồn: {_source_label(source)}."
    )


def _source_label(source: dict[str, Any]) -> str:
    source_file = source.get("source_file", "unknown")
    pages = source.get("pages") or []
    if not pages:
        return source_file
    return f"{source_file}, trang {', '.join(str(page) for page in pages)}"


def _table_from_source(source: dict[str, Any]) -> dict[str, Any] | None:
    metadata = source.get("metadata", {})
    table = metadata.get("table")
    if isinstance(table, dict):
        return table
    return None


def _render_show_questions(sources: list[dict[str, Any]]) -> str | None:
    question_groups = [
        source
        for source in sources
        if source.get("metadata", {}).get("unit_type") == "question_group"
    ]
    if question_groups:
        lines = []
        for source in question_groups:
            text = (source.get("display_text") or source.get("text") or "").strip()
            if not text:
                continue
            lines.append(text)
            lines.append(f"Nguồn: {_source_label(source)}.")
        return "\n\n".join(lines).strip() or None

    questions = [
        source
        for source in sources
        if source.get("metadata", {}).get("unit_type") == "question"
    ]
    if not questions:
        return None
    questions = sorted(questions, key=lambda source: source.get("metadata", {}).get("question_start") or 999)
    lines = ["Nội dung câu hỏi:"]
    for source in questions:
        text = (source.get("display_text") or source.get("text") or "").strip()
        if text:
            lines.append(f"- {text}")
    lines.append(f"\nNguồn: {_source_label(questions[0])}.")
    return "\n".join(lines)


def _lookup_table_cell(message: str, sources: list[dict[str, Any]]) -> str | None:
    best_match: tuple[float, Any, dict[str, Any]] | None = None
    for source in sources:
        table = _table_from_source(source)
        metadata = source.get("metadata", {})
        if table:
            columns = table.get("columns") or []
            rows = table.get("rows") or []
        else:
            columns = metadata.get("table_columns") or []
            row = metadata.get("table_row")
            rows = [row] if isinstance(row, list) else []
        match = table_cell_value(message, {"columns": columns, "rows": rows})
        if match and (best_match is None or match[0] > best_match[0]):
            best_match = (match[0], match[1], source)
    if best_match is None:
        return None
    _, value, source = best_match
    return f"{value}\n\nNguồn: {_source_label(source)}."


def _full_table_source(sources: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    candidates = []
    for source in sources:
        table = _table_from_source(source)
        if not table or not table.get("columns") or not table.get("rows"):
            continue
        candidates.append((len(table.get("rows") or []), table, source))
    if not candidates:
        return None
    _, table, source = max(candidates, key=lambda item: item[0])
    return table, source


def _render_table_calculation(message: str, sources: list[dict[str, Any]]) -> str | None:
    selected = _full_table_source(sources)
    if not selected:
        return None
    table, source = selected
    result = table_change_calculations(message, table)
    if not result:
        return None
    lines = [
        f"- {item['label']}: {format_number(item['second'])} - {format_number(item['first'])} = {format_number(item['change'])}"
        for item in result["calculations"]
    ]
    direction = "giảm" if result["direction"] == "decrease" else "tăng"
    winner = result["winner"]
    lines.append(
        f"\n{winner['label']} có mức {direction} lớn nhất: {format_number(winner['change'])}."
    )
    lines.append(f"\nNguồn: {_source_label(source)}.")
    return "\n".join(lines)


def _render_table_comparison(message: str, sources: list[dict[str, Any]]) -> str | None:
    selected = _full_table_source(sources)
    if not selected:
        return None
    table, source = selected
    row = comparison_row(message, table)
    if not row:
        return None
    markdown = _markdown_table({"columns": table.get("columns") or [], "rows": [row]})
    if not markdown:
        return None
    facts = comparison_row_facts(table, row)
    comparison = "\n".join(f"- {fact}" for fact in facts)
    if comparison:
        return f"{markdown}\n\n{comparison}\n\nNguồn: {_source_label(source)}."
    return f"{markdown}\n\nNguồn: {_source_label(source)}."


def _render_writing_prompt(sources: list[dict[str, Any]]) -> str | None:
    for source in sources:
        if source.get("metadata", {}).get("unit_type") not in {"writing_prompt", "writing_task"}:
            continue
        text = (source.get("display_text") or source.get("text") or "").strip()
        if text:
            return f"{text}\n\nNguồn: {_source_label(source)}."
    return None


def _render_writing_inventory(sources: list[dict[str, Any]]) -> str | None:
    tasks = [source for source in sources if source.get("metadata", {}).get("unit_type") == "writing_task"]
    if not tasks:
        return None
    answer_keys = {
        str(source.get("metadata", {}).get("section_id", "")).removesuffix("-answer")
        for source in sources
        if source.get("metadata", {}).get("unit_type") == "sample_answer"
    }
    lines = ["Các đề và bài mẫu trong tài liệu:"]
    for source in sorted(tasks, key=lambda item: (min(item.get("pages") or [999]), item.get("chunk_index", 0))):
        text = (source.get("display_text") or source.get("text") or "").strip()
        title = next((line.strip() for line in text.splitlines() if line.strip()), "Writing task")
        section_key = str(source.get("metadata", {}).get("section_id", "")).removesuffix("-task")
        sample_label = "có bài mẫu" if section_key in answer_keys else "chưa thấy bài mẫu"
        lines.append(f"- Trang {', '.join(str(page) for page in source.get('pages') or [])}: {title} ({sample_label})")
    lines.append(f"\nNguồn: {_source_label(tasks[0])}.")
    return "\n".join(lines)


def solve_context_issue(sources: list[dict[str, Any]]) -> str | None:
    question_text = "\n".join(
        (source.get("display_text") or source.get("text") or "").strip()
        for source in sources
        if source.get("metadata", {}).get("unit_type") in {"question", "question_group"}
    )
    if not question_text:
        return "missing_question"
    requires_options = bool(
        re.search(
            r"(?:from\s+the\s+list\s+below|choose\s+(?:the\s+)?(?:correct\s+)?letter|"
            r"which\s+of\s+the\s+following)",
            question_text,
            flags=re.IGNORECASE,
        )
    )
    option_labels = set(
        re.findall(r"(?:^|\s)([A-H])(?:[.)]|\s+(?=\S))", question_text)
    )
    if requires_options and len(option_labels) < 2:
        return "missing_answer_options"
    return None


def writing_table_facts(sources: list[dict[str, Any]]) -> list[str]:
    selected = _full_table_source(sources)
    return table_summary_facts(selected[0]) if selected else []


def static_response_for_sources(message: str, query_intent: str, sources: list[dict[str, Any]]) -> str | None:
    if query_intent == "table_cell":
        return _lookup_table_cell(message, sources)
    if query_intent == "table_calculation":
        return _render_table_calculation(message, sources)
    if query_intent == "table_comparison":
        return _render_table_comparison(message, sources)
    if query_intent == "show_writing_prompt":
        return _render_writing_prompt(sources)
    if query_intent == "document_overview":
        inventory = _render_writing_inventory(sources)
        has_non_writing_sections = any(
            source.get("metadata", {}).get("unit_type")
            in {"document_outline", "passage", "passage_summary", "question_group"}
            for source in sources
        )
        if inventory and not has_non_writing_sections:
            return inventory

    if query_intent == "show_questions":
        questions = _render_show_questions(sources)
        if questions:
            return questions

    if query_intent in {"show_table", "extract_table"}:
        for source in sources:
            table = _table_from_source(source)
            table_markdown = _markdown_table(table) if table else ""
            if table_markdown:
                return f"Dưới đây là bảng mình trích xuất được từ tài liệu:\n\n{table_markdown}\n\nNguồn: {_source_label(source)}."
            if table:
                return _visual_incomplete_text(table, source)
        return (
            "Mình chưa có dữ liệu bảng đã được trích xuất theo cấu trúc cho phần này. "
            "Để tránh tự dựng sai hàng/cột hoặc ô trống, mình chưa hiển thị bảng."
        )

    if query_intent == "show_flowchart":
        for source in sources:
            metadata = source.get("metadata", {})
            flowchart = metadata.get("flowchart")
            if isinstance(flowchart, dict):
                nodes = flowchart.get("nodes") or []
                edges = flowchart.get("edges") or []
                if not nodes or not edges:
                    return _visual_incomplete_text(flowchart, source)
                lines = ["Mình tìm thấy cấu trúc flowchart:"]
                for node in nodes:
                    label = f"Question {node['question_number']} blank" if node.get("question_number") else node.get("text", "")
                    lines.append(f"- {node['id']}: {label}")
                for edge in edges:
                    lines.append(f"- edge: {edge['from']} -> {edge['to']}")
                lines.append(f"\nNguồn: {_source_label(source)}.")
                return "\n".join(lines)
        return (
            "Mình chưa có dữ liệu flowchart đã được trích xuất theo node/edge cho phần này. "
            "Để tránh tự tưởng tượng cấu trúc, mình chưa mô tả flowchart."
        )

    if query_intent == "show_diagram":
        for source in sources:
            diagram = source.get("metadata", {}).get("diagram")
            if not isinstance(diagram, dict):
                continue
            nodes = diagram.get("nodes") or []
            edges = diagram.get("edges") or []
            if not nodes or not edges:
                return _visual_incomplete_text(diagram, source)
            lines = ["Mình tìm thấy cấu trúc diagram:"]
            for node in nodes:
                label = f"Question {node['question_number']} blank" if node.get("question_number") else node.get("text", "")
                lines.append(f"- {node['id']}: {label}")
            for edge in edges:
                lines.append(f"- edge: {edge['from']} -> {edge['to']}")
            lines.append(f"\nNguồn: {_source_label(source)}.")
            return "\n".join(lines)
        return (
            "Mình chưa có dữ liệu diagram đã được trích xuất theo cấu trúc cho phần này. "
            "Để tránh tự tưởng tượng nhãn hoặc quan hệ, mình chưa mô tả diagram."
        )

    return None


def is_presence_check_query(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in ["có nhắc đến", "có nói về", "có đề cập", "mentions", "mention"])


def has_lexical_source_hit(sources: list[dict[str, Any]]) -> bool:
    for source in sources:
        if source.get("probe_keyword_score", 0.0) > 0 or source.get("keyword_score", 0.0) > 0:
            return True
        if source.get("probe_question_score", 0.0) > 0 or source.get("question_score", 0.0) > 0:
            return True
        if source.get("probe_overview_score", 0.0) > 0 or source.get("overview_score", 0.0) > 0:
            return True
    return False


def compact_final_context_debug(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": source.get("chunk_id"),
            "document_id": source.get("document_id"),
            "source_file": source.get("source_file"),
            "method": source.get("retrieval_method"),
            "unit_type": source.get("metadata", {}).get("unit_type"),
            "passage_number": source.get("metadata", {}).get("passage_number"),
            "question_range": source.get("metadata", {}).get("question_range"),
            "parent_id": source.get("metadata", {}).get("parent_id"),
            "pages": source.get("pages"),
        }
        for source in sources
    ]


@dataclass
class ChatPreparation:
    prompt: str | None
    static_response: str | None
    route_used: str
    sources: list[dict[str, Any]]
    debug: dict[str, Any]
    query_intent: str = "direct"


def affinity_retrieval_query(req: ChatRequest, scope: DocumentScope) -> str:
    message = req.message.strip()
    if scope.method != "conversation_affinity" or not req.conversation_history:
        return message

    previous_user_message = next(
        (
            item.content.strip()
            for item in reversed(req.conversation_history)
            if item.role == "user" and item.content.strip()
        ),
        "",
    )
    if not previous_user_message:
        return message

    affinity_hints: list[str] = []
    affinity = effective_affinity(req)
    if affinity:
        if affinity.passage_numbers:
            affinity_hints.append(
                "Passages: " + ", ".join(str(value) for value in affinity.passage_numbers)
            )
        if affinity.question_ranges:
            affinity_hints.append(
                "Question ranges: "
                + ", ".join(
                    f"{values[0]}-{values[1]}"
                    for values in affinity.question_ranges
                    if len(values) == 2
                )
            )
    suffix = f"\n{' | '.join(affinity_hints)}" if affinity_hints else ""
    return f"{previous_user_message}\nFollow-up: {message}{suffix}"


def effective_affinity(req: ChatRequest) -> ChatAffinity | None:
    if req.conversation_state and req.conversation_state.rag_affinity.document_ids:
        return req.conversation_state.rag_affinity
    return req.affinity


CURRENT_BAND_PATTERNS = (
    re.compile(
        r"\b(?:hiện\s*tại|currently)\b\s*(?:tôi|mình|i\s+am)?\s*(?:đang|ở|at)?\s*"
        r"(?:band\s*)?([0-9](?:[.,][0-9])?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:tôi|mình)\s+(?:đang\s+|ở\s+)?band\s*([0-9](?:[.,][0-9])?)\b", re.IGNORECASE),
    re.compile(r"\b(?:i\s+am|i'm|my\s+(?:current\s+)?band\s+is)\s+(?:at\s+)?(?:band\s*)?([0-9](?:[.,][0-9])?)\b", re.IGNORECASE),
)
TARGET_BAND_PATTERNS = (
    re.compile(r"\b(?:mục\s*tiêu|đạt|lên)\b.{0,24}?\b(?:band\s*)?([0-9](?:[.,][0-9])?)\b", re.IGNORECASE),
    re.compile(r"\b(?:target|reach|aim(?:ing)?\s+for)\b.{0,24}?\b(?:band\s*)?([0-9](?:[.,][0-9])?)\b", re.IGNORECASE),
)
STUDY_DURATION_PATTERN = re.compile(
    r"\b(?:trong(?:\s+vòng)?|within|over|for)\s+([0-9]{1,3})\s*(?:tháng|months?)\b",
    re.IGNORECASE,
)


def _first_band_value(message: str, patterns: tuple[re.Pattern[str], ...]) -> float | None:
    for pattern in patterns:
        match = pattern.search(message)
        if match:
            value = float(match.group(1).replace(",", "."))
            if 0 <= value <= 9:
                return value
    return None


def user_profile_for_request(req: ChatRequest) -> ChatUserProfile:
    previous = req.conversation_state.user_profile if req.conversation_state else ChatUserProfile()
    current_band = _first_band_value(req.message, CURRENT_BAND_PATTERNS)
    target_band = _first_band_value(req.message, TARGET_BAND_PATTERNS)
    duration_match = STUDY_DURATION_PATTERN.search(req.message)
    duration = int(duration_match.group(1)) if duration_match else None
    return ChatUserProfile(
        current_band=current_band if current_band is not None else previous.current_band,
        target_band=target_band if target_band is not None else previous.target_band,
        study_duration_months=(
            duration if duration is not None and 1 <= duration <= 120 else previous.study_duration_months
        ),
    )


def user_profile_prompt_context(profile: ChatUserProfile) -> str:
    facts = []
    if profile.current_band is not None:
        facts.append(f"- Current IELTS band: {profile.current_band:g}")
    if profile.target_band is not None:
        facts.append(f"- Target IELTS band: {profile.target_band:g}")
    if profile.study_duration_months is not None:
        facts.append(f"- Study duration: {profile.study_duration_months} months")
    return "\n".join(facts)


def gateway_state_context(req: ChatRequest) -> str:
    if not req.conversation_state:
        return ""
    return json.dumps(
        {
            "last_route": req.conversation_state.last_route,
            "last_intent": req.conversation_state.last_intent,
            "has_rag_affinity": bool(req.conversation_state.rag_affinity.document_ids),
        },
        ensure_ascii=False,
    )


def resolve_target_refs(
    references: tuple[str, ...],
    routing_context: DocumentCatalogContext,
    allowed_document_ids: list[str],
) -> tuple[list[str], list[str]]:
    allowed = set(allowed_document_ids)
    resolved: list[str] = []
    invalid: list[str] = []
    for reference in references:
        document_id = routing_context.document_refs.get(reference)
        if not document_id or document_id not in allowed:
            invalid.append(reference)
        elif document_id not in resolved:
            resolved.append(document_id)
    return resolved, invalid


def gateway_clarification_response(catalog: list[dict[str, Any]]) -> str:
    files = [item.get("source_file", "unknown") for item in catalog]
    if not files:
        return NO_RAG_MATCH_RESPONSE
    choices = "\n".join(f"- {name}" for name in files[:10])
    return f"{AMBIGUOUS_DOCUMENT_RESPONSE}\n\nCác file hiện có:\n{choices}"


async def prepare_chat(req: ChatRequest) -> ChatPreparation:
    message = req.message.strip()
    store = get_store()
    route = "direct"
    sources: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    probe: dict[str, Any] = {"results": []}
    query_intent = "direct"
    intent_debug: dict[str, Any] = {}
    writing_parent_id: str | None = None
    evidence_query: str | None = None
    gateway_debug: dict[str, Any]
    document_resolution_debug: dict[str, Any] = {}

    stats = await run_in_threadpool(store.stats)
    full_catalog = await run_in_threadpool(store.document_catalog)
    scope = resolve_document_scope(
        message,
        full_catalog,
        req.document_ids,
        req.document_scope,
    )
    affinity = effective_affinity(req)
    scope = apply_document_affinity(
        scope,
        full_catalog,
        affinity.document_ids if affinity else None,
    )
    allowed_scope_ids = scope.allowed_document_ids
    catalog = [
        item
        for item in full_catalog
        if any(document_id in allowed_scope_ids for document_id in item.get("document_ids", []))
    ]
    if scope.request_mode == "explicit" and scope.resolved_document_ids:
        route = "rag"
        gateway_debug = {
            "used": False,
            "route": "rag",
            "attempts": 0,
            "duration_seconds": 0.0,
            "raw_output_preview": "",
            "fallback_reason": None,
            "reason": "explicit_current_turn_document_scope",
        }
    else:
        gateway_decision = await classify_chat_route(
            message,
            req.conversation_history,
            gateway_state_context(req),
        )
        route = gateway_decision.route
        gateway_debug = {"used": True, **gateway_decision.to_debug()}

    if route == "direct":
        user_profile = user_profile_for_request(req)
        return ChatPreparation(
            prompt=direct_answer_prompt(
                message,
                req.conversation_history,
                user_profile_prompt_context(user_profile),
            ),
            static_response=None,
            route_used="base_model",
            sources=[],
            debug={
                "route_decision": "direct",
                "query_intent": "direct",
                "route_gateway": gateway_debug,
                "document_resolution": {"skipped": True, "reason": "direct_route"},
                "intent_classifier": {"skipped": True, "reason": "direct_route"},
                "direct_generation": {
                    "used": True,
                    "response_contract": "adaptive_direct_answer",
                },
                "retrieval": {"skipped": True, "final_context": []},
                "conversation_state": req.conversation_state.model_dump() if req.conversation_state else None,
                "user_profile": user_profile.model_dump(exclude_none=True),
                "source_count": 0,
            },
            query_intent="direct",
        )

    has_safe_rag_fallback = bool(
        (scope.request_mode == "explicit" and scope.resolved_document_ids)
        or (scope.document_grounded and scope.resolved_document_ids)
        or (affinity and affinity.document_ids)
    )
    if route == "undetermined" and not has_safe_rag_fallback:
        return ChatPreparation(
            prompt=None,
            static_response=ROUTE_UNDETERMINED_RESPONSE,
            route_used="route_undetermined",
            sources=[],
            debug={
                "route_decision": "undetermined",
                "query_intent": "route_undetermined",
                "route_gateway": gateway_debug,
                "document_resolution": {"skipped": True, "reason": "route_undetermined"},
                "intent_classifier": {"skipped": True, "reason": "route_undetermined"},
                "retrieval": {"skipped": True, "final_context": []},
                "conversation_state": req.conversation_state.model_dump() if req.conversation_state else None,
                "source_count": 0,
            },
            query_intent="route_undetermined",
        )
    if route == "undetermined":
        route = "rag"
        gateway_debug["fallback_reason"] = "valid_document_scope_or_rag_affinity"

    gateway_context = format_document_catalog_context(catalog)
    scope_ids = list(scope.resolved_document_ids)
    needs_target_model = not scope_ids and len(allowed_scope_ids) > 1
    target_decision = (
        await resolve_rag_target(message, gateway_context.text)
        if needs_target_model
        else None
    )

    if not scope_ids and len(allowed_scope_ids) == 1:
        scope_ids = list(allowed_scope_ids)
        document_resolution_debug = {"method": "single_allowed_document"}
    elif scope_ids:
        document_resolution_debug = {"method": scope.method}
    elif target_decision and target_decision.action == "all":
        scope_ids = list(allowed_scope_ids)
        document_resolution_debug = {"method": "semantic_target_all", **target_decision.to_debug()}
    elif target_decision and target_decision.action == "selected":
        scope_ids, invalid_refs = resolve_target_refs(
            target_decision.document_refs,
            gateway_context,
            allowed_scope_ids,
        )
        document_resolution_debug = {
            "method": "semantic_target",
            **target_decision.to_debug(),
            "invalid_refs": invalid_refs,
        }
    else:
        document_resolution_debug = {
            "method": "semantic_target_clarify",
            **(target_decision.to_debug() if target_decision else {}),
        }

    if not scope_ids:
        debug = {
            "route_decision": "rag",
            "query_intent": "ambiguous_document" if len(allowed_scope_ids) > 1 else "document_no_match",
            "route_gateway": gateway_debug,
            "document_resolution": document_resolution_debug,
            "intent_classifier": {"skipped": True, "reason": "document_scope_unresolved"},
            "target_resolution": scope.to_debug(),
            "catalog": full_catalog,
            "probe": compact_probe_debug(probe),
            "retrieval": {
                "method": None,
                "structured_hits": 0,
                "before_filter_count": 0,
                "after_filter_count": 0,
                "final_context": [],
            },
            "source_count": 0,
        }
        return ChatPreparation(
            prompt=None,
            static_response=(
                gateway_clarification_response(catalog)
                if len(allowed_scope_ids) > 1
                else NO_RAG_MATCH_RESPONSE
            ),
            route_used=(
                "vector_rag_ambiguous_document"
                if len(allowed_scope_ids) > 1
                else "vector_rag_no_match"
            ),
            sources=[],
            debug=debug,
            query_intent=("ambiguous_document" if len(allowed_scope_ids) > 1 else "document_no_match"),
        )

    intent_classifier = await classify_rag_intent(message, req.conversation_history)
    if intent_classifier.intent == "undetermined":
        debug = {
            "route_decision": "rag",
            "query_intent": "intent_undetermined",
            "route_gateway": gateway_debug,
            "document_resolution": {
                **document_resolution_debug,
                "resolved_document_ids": scope_ids,
                "requested_scope": scope.to_debug(),
            },
            "intent_classifier": intent_classifier.to_debug(),
            "target_resolution": scope.to_debug(),
            "catalog": catalog,
            "probe": compact_probe_debug(probe),
            "retrieval": {"skipped": True, "final_context": []},
            "source_count": 0,
        }
        return ChatPreparation(
            prompt=None,
            static_response=INTENT_UNDETERMINED_RESPONSE,
            route_used="intent_undetermined",
            sources=[],
            debug=debug,
            query_intent="intent_undetermined",
        )

    intent_decision = semantic_intent_decision(
        message,
        intent_classifier.intent,
        1.0,
        "Semantic intent enum classifier.",
    )
    query_intent = intent_decision.intent
    intent_debug = intent_decision.to_debug()
    intent_debug["classifier"] = intent_classifier.to_debug()

    if scope_ids:
        catalog = [
            item
            for item in full_catalog
            if any(document_id in scope_ids for document_id in item.get("document_ids", []))
        ]

    retrieval_query = affinity_retrieval_query(req, scope)
    probe_top_k = max(settings.rag_probe_top_k, settings.rag_top_k)

    if stats["chunks"] > 0 and route == "rag" and query_intent == "semantic_qa":
        semantic_results = await run_in_threadpool(
            store.hybrid_search,
            retrieval_query,
            probe_top_k,
            scope_ids,
            None,
            None,
        )
        probe = {
            "results": semantic_results,
            "has_hits": bool(semantic_results),
            "has_strong_hits": bool(semantic_results),
            "has_document_intent": True,
            "is_overview": False,
            "top_score": semantic_results[0].get("score", 0.0) if semantic_results else 0.0,
            "top_fused_score": semantic_results[0].get("rrf_score", 0.0) if semantic_results else 0.0,
            "top_keyword_score": semantic_results[0].get("probe_keyword_score", 0.0) if semantic_results else 0.0,
            "top_question_score": 0.0,
            "top_overview_score": 0.0,
        }

    if route == "rag":
        evidence_candidate_count = 0
        evidence_context_count = 0
        structured_top_k = 50 if query_intent == "document_overview" else max(
            settings.rag_top_k,
            settings.rag_overview_top_k,
        )
        structured_sources = await run_in_threadpool(
            store.structured_lookup,
            retrieval_query,
            query_intent,
            structured_top_k,
            scope_ids,
        )
        retrieval_method = "structured" if structured_sources else None
        if structured_sources:
            source_limit = 50 if query_intent == "document_overview" else settings.rag_top_k
            sources = structured_sources[:source_limit]
        elif query_intent == "document_overview":
            sources = await run_in_threadpool(
                store.overview,
                settings.rag_overview_top_k,
                scope_ids,
            )
            for source in sources:
                source["probe_overview_score"] = 1.0
            retrieval_method = "overview"
        elif probe.get("has_strong_hits"):
            sources = (probe.get("results") or [])[: settings.rag_top_k]
            retrieval_method = "probe"
        elif route == "rag":
            sources = []
            retrieval_method = "no_strong_document_match"
        else:
            sources = await run_in_threadpool(
                store.search,
                retrieval_query,
                settings.rag_top_k,
                scope_ids,
            )
            retrieval_method = "dense"
        before_filter_count = len(sources)
        sources = filter_sources_for_intent(sources, message, query_intent)
        if query_intent in {"semantic_qa", "writing_generation"} and any(
            source.get("metadata", {}).get("unit_type") in {"writing_task", "sample_answer"}
            for source in sources
        ):
            writing_context = await run_in_threadpool(
                store.writing_context_for_sources,
                sources,
                4,
                scope_ids,
            )
            if writing_context:
                sources = writing_context
                writing_parent_id = sources[0].get("metadata", {}).get("parent_id")
                retrieval_method = "writing_parent"
        if query_intent == "solve_questions" and sources:
            question_context = await run_in_threadpool(
                store.question_context_for_sources,
                sources,
                8,
                scope_ids,
            )
            target_passages = {
                (source.get("document_id"), source.get("metadata", {}).get("passage_number"))
                for source in sources + question_context
                if source.get("document_id")
                and source.get("metadata", {}).get("passage_number")
            }
            evidence_query = evidence_query_for_sources(sources + question_context, message)
            evidence_candidates = []
            for document_id, passage_number in sorted(target_passages):
                pair_candidates = await run_in_threadpool(
                    store.hybrid_search,
                    evidence_query,
                    max(settings.rag_top_k * 3, 12),
                    [document_id],
                    ["passage"],
                    [passage_number],
                )
                evidence_candidates.extend(pair_candidates)
            evidence_candidates = dedupe_sources(evidence_candidates)
            evidence_candidate_count = len(evidence_candidates)
            evidence_context = evidence_candidates[:3]
            if not evidence_context:
                evidence_context = await run_in_threadpool(
                    store.passage_context_for_sources,
                    sources,
                    3,
                    scope_ids,
                )
            evidence_context_count = len(evidence_context)
            sources = dedupe_sources(sources + question_context + evidence_context)
        elif query_intent == "semantic_qa" and scope.method == "conversation_affinity" and sources:
            passage_context = await run_in_threadpool(
                store.passage_context_for_sources,
                sources,
                3,
                scope_ids,
            )
            if passage_context:
                sources = dedupe_sources(sources + passage_context)
                retrieval_method = f"{retrieval_method or 'semantic'}_with_parent"
        sources = dedupe_sources(sources)
    else:
        structured_sources = []
        retrieval_method = None
        before_filter_count = 0
        evidence_candidate_count = 0
        evidence_context_count = 0

    debug = {
        "route_decision": route,
        "query_intent": query_intent,
        "intent_decision": intent_debug,
        "route_gateway": gateway_debug,
        "document_resolution": {
            **document_resolution_debug,
            "resolved_document_ids": scope_ids,
            "requested_scope": scope.to_debug(),
        },
        "intent_classifier": intent_classifier.to_debug(),
        "target_resolution": scope.to_debug(),
        "catalog": catalog,
        "probe": compact_probe_debug(probe),
        "retrieval": {
            "method": retrieval_method,
            "structured_hits": len(structured_sources),
            "before_filter_count": before_filter_count,
            "after_filter_count": len(sources),
            "evidence_candidate_count": evidence_candidate_count,
            "evidence_context_count": evidence_context_count,
            "evidence_query": evidence_query,
            "retrieval_query": retrieval_query,
            "writing_parent_id": writing_parent_id,
            "final_context": compact_final_context_debug(sources),
        },
        "source_count": len(sources),
        "conversation_state": req.conversation_state.model_dump() if req.conversation_state else None,
    }

    if sources:
        if query_intent == "solve_questions":
            context_issue = solve_context_issue(sources)
            if context_issue:
                debug["no_match_guard"] = context_issue
                return ChatPreparation(
                    prompt=None,
                    static_response=INCOMPLETE_QUESTION_RESPONSE,
                    route_used="vector_rag_no_match",
                    sources=sources,
                    debug=debug,
                    query_intent=query_intent,
                )
        if is_presence_check_query(message) and not has_lexical_source_hit(sources):
            debug["no_match_guard"] = "presence_check_without_lexical_hit"
            return ChatPreparation(
                prompt=None,
                static_response=NO_RAG_MATCH_RESPONSE,
                route_used="vector_rag_no_match",
                sources=sources,
                debug=debug,
                query_intent=query_intent,
            )
        static_response = static_response_for_sources(message, query_intent, sources)
        if static_response:
            debug["static_response"] = True
            return ChatPreparation(
                prompt=None,
                static_response=static_response,
                route_used="vector_rag_static",
                sources=sources,
                debug=debug,
                query_intent=query_intent,
            )
        deterministic_intents = {
            "show_questions",
            "show_table",
            "extract_table",
            "table_cell",
            "table_calculation",
            "table_comparison",
            "show_flowchart",
            "show_diagram",
            "show_writing_prompt",
        }
        if query_intent in deterministic_intents:
            debug["no_match_guard"] = "deterministic_intent_without_structured_response"
            return ChatPreparation(
                prompt=None,
                static_response=NO_RAG_MATCH_RESPONSE,
                route_used="vector_rag_no_match",
                sources=sources,
                debug=debug,
                query_intent=query_intent,
            )
        context = (
            format_context(sources, max_chars_per_source=settings.rag_overview_source_chars)
            if probe.get("is_overview")
            else format_context(sources)
        )
        if query_intent == "writing_generation":
            facts = writing_table_facts(sources)
            if facts:
                debug["retrieval"]["deterministic_table_facts"] = facts
                context += "\n\n[Deterministic table facts]\n" + "\n".join(
                    f"- {fact}" for fact in facts
                )
        return ChatPreparation(
            prompt=rag_prompt(
                message,
                context,
                req.conversation_history,
                query_intent=query_intent,
                allow_solution=bool(intent_debug.get("allow_solution")),
                writing_context=query_intent == "writing_generation" or bool(writing_parent_id),
            ),
            static_response=None,
            route_used="vector_rag",
            sources=sources,
            debug=debug,
            query_intent=query_intent,
        )

    if route == "rag":
        return ChatPreparation(
            prompt=None,
            static_response=NO_RAG_MATCH_RESPONSE,
            route_used="vector_rag_no_match",
            sources=[],
            debug=debug,
            query_intent=query_intent,
        )

    return ChatPreparation(
        prompt=direct_answer_prompt(
            message,
            req.conversation_history,
            user_profile_prompt_context(user_profile_for_request(req)),
        ),
        static_response=None,
        route_used="base_model",
        sources=[],
        debug=debug,
        query_intent=query_intent,
    )


@app.get("/health")
async def health() -> dict:
    stats = await run_in_threadpool(get_store().stats)
    return {
        "status": "ok",
        "runtime_status": (LAST_WARMUP_STATUS or {}).get("status", "not_warmed"),
        "model_readiness": (LAST_WARMUP_STATUS or {}).get("components", {}),
        "document_rag_documents": stats["documents"],
        "document_rag_chunks": stats["chunks"],
        "pdf_rag_documents": stats["documents"],
        "pdf_rag_chunks": stats["chunks"],
        "ollama_api_url": settings.ollama_api_url,
        "ollama_model": OLLAMA_MODEL,
        "ollama_num_predict": OLLAMA_NUM_PREDICT,
    }


@app.post("/warmup")
async def warmup() -> dict:
    global LAST_WARMUP_STATUS
    started = time.perf_counter()
    results = {}

    if settings.warmup_llm:
        llm_started = time.perf_counter()
        try:
            response = await query_ollama(
                direct_answer_prompt("Give me one concise IELTS Speaking tip."),
                temperature=0.2,
                num_predict=192,
            )
            direct_check = await classify_chat_route("Give me one concise IELTS Speaking tip.")
            rag_check = await classify_chat_route("Summarize the content of the uploaded document.")
            intent_check = await classify_rag_intent("List Questions 1-4 without solving them.")
            gateway_ok = (
                bool(response.strip())
                and direct_check.route == "direct"
                and rag_check.route == "rag"
                and intent_check.intent in {
                    "show_questions",
                    "translate_questions",
                    "explain_questions",
                    "solve_questions",
                    "semantic_qa",
                    "document_overview",
                    "show_table",
                    "extract_table",
                    "table_cell",
                    "table_calculation",
                    "table_comparison",
                    "show_flowchart",
                    "show_diagram",
                    "show_writing_prompt",
                    "writing_generation",
                }
            )
            results["llm"] = {
                "ok": gateway_ok,
                "model": OLLAMA_MODEL,
                "duration_seconds": round(time.perf_counter() - llm_started, 2),
                "sample": response[:120],
                "gateway": {
                    "direct": direct_check.to_debug(),
                    "rag": rag_check.to_debug(),
                    "intent": intent_check.to_debug(),
                },
            }
        except Exception as exc:
            results["llm"] = {
                "ok": False,
                "error": str(exc),
                "diagnostic": ollama_failure_detail(exc).get("ollama"),
            }
    else:
        results["llm"] = {"skipped": True}

    if settings.warmup_embedding:
        embedding_started = time.perf_counter()
        try:
            embedding_result = await run_in_threadpool(get_store().warmup)
            results["embedding"] = {
                "ok": True,
                "duration_seconds": round(time.perf_counter() - embedding_started, 2),
                **embedding_result,
            }
        except Exception as exc:
            results["embedding"] = {"ok": False, "error": str(exc)}
    else:
        results["embedding"] = {"skipped": True}

    layout_started = time.perf_counter()
    try:
        layout_result = await run_in_threadpool(DOCUMENT_PROCESSOR.warmup_layout)
        results["layout"] = {
            "ok": bool(layout_result.get("skipped") or layout_result.get("ok", False)),
            "duration_seconds": round(time.perf_counter() - layout_started, 2),
            **layout_result,
        }
    except Exception as exc:
        results["layout"] = {"ok": False, "error": str(exc)}

    ocr_started = time.perf_counter()
    try:
        ocr_result = await run_in_threadpool(DOCUMENT_PROCESSOR.warmup_ocr)
        results["ocr"] = {
            "ok": bool(ocr_result.get("skipped") or ocr_result.get("models_ready", False)),
            "duration_seconds": round(time.perf_counter() - ocr_started, 2),
            **ocr_result,
        }
    except Exception as exc:
        results["ocr"] = {"ok": False, "error": str(exc)}

    ok = all(component.get("ok", True) for component in results.values())
    status = "ok" if ok else "partial"
    LAST_WARMUP_STATUS = {
        "status": status,
        "components": {
            name: bool(result.get("ok", result.get("skipped", False)))
            for name, result in results.items()
        },
    }
    return {
        "status": status,
        "duration_seconds": round(time.perf_counter() - started, 2),
        "results": results,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung câu hỏi")

    try:
        prepared = await prepare_chat(req)
        answer = prepared.static_response
        if answer is None and prepared.prompt is not None:
            answer = await generate_answer(prepared, req.message)
            if not answer.strip():
                answer = generation_fallback(prepared)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=502,
            detail=ollama_failure_detail(exc),
        ) from exc
    return ChatResponse(
        response=answer or "",
        route_used=prepared.route_used,
        sources=prepared.sources,
        debug=prepared.debug,
        conversation_state=conversation_state_for_result(req, prepared),
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung câu hỏi")

    async def generate():
        try:
            yield stream_event("status", message="Đang phân tích câu hỏi...")
            prepared = await prepare_chat(req)
            yield stream_event(
                "metadata",
                route_used=prepared.route_used,
                sources=prepared.sources,
                debug=prepared.debug,
                conversation_state=conversation_state_for_result(req, prepared).model_dump(),
            )
            if prepared.static_response is not None:
                yield stream_event("token", token=prepared.static_response)
                yield stream_event("done")
                return

            yield stream_event("status", message="Đang soạn câu trả lời...")
            if requires_reviewed_generation(prepared, req.message):
                answer = await generate_answer(prepared, req.message)
                if not answer.strip():
                    answer = generation_fallback(prepared)
                for token in response_chunks(answer):
                    yield stream_event("token", token=token)
                yield stream_event("done")
                return

            has_token = False
            temperature = generation_temperature(prepared)
            async for token in stream_ollama(
                prepared.prompt or "",
                temperature=temperature,
            ):
                has_token = True
                yield stream_event("token", token=token)
            if not has_token:
                logger.warning("Ollama stream completed without visible tokens; retrying with non-stream request")
                fallback_answer = await query_ollama(
                    prepared.prompt or "",
                    temperature=temperature,
                )
                if not fallback_answer.strip():
                    fallback_answer = generation_fallback(prepared)
                yield stream_event("token", token=fallback_answer)
            yield stream_event("done")
        except Exception as exc:
            logger.exception("Streaming chat failed")
            yield stream_event(
                "error",
                message="Không thể tạo câu trả lời lúc này. Vui lòng thử lại.",
                detail=ollama_failure_detail(exc),
            )

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/documents/upload", response_model=UploadResponse)
@app.post("/rag/upload-pdf", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    upload_started = time.perf_counter()
    upload_timing: dict[str, Any] = {}
    timing_debug: dict[str, Any] = {"upload": {}}

    if not file.filename:
        raise HTTPException(status_code=400, detail="Tên tệp không hợp lệ")

    safe_name = Path(file.filename).name
    request_id = uuid4().hex
    file_path = UPLOAD_DIR / f"{request_id}-{safe_name}"
    max_bytes = DOCUMENT_PROCESSOR.config.max_upload_mb * 1024 * 1024

    try:
        save_started = time.perf_counter()
        total_bytes = 0
        async with aiofiles.open(file_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Tệp quá lớn. Giới hạn hiện tại là {DOCUMENT_PROCESSOR.config.max_upload_mb}MB.",
                    )
                await out.write(chunk)
        upload_timing["save_file_seconds"] = round(time.perf_counter() - save_started, 3)

        process_started = time.perf_counter()
        document, chunks = await run_in_threadpool(
            DOCUMENT_PROCESSOR.process_file,
            file_path,
            safe_name,
            file.content_type,
        )
        upload_timing["process_file_seconds"] = round(time.perf_counter() - process_started, 3)
        document_timing = document.metadata.get("timing", {})
        timing_debug = {
            "upload": dict(upload_timing),
            "extraction": document_timing.get("extraction", {}),
            "process_file": document_timing.get("process_file", {}),
            "chunking": document_timing.get("chunking", {}),
            "embedding": {},
        }
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=document_extraction_failure_detail(document),
            )

        store = get_store()
        upsert_started = time.perf_counter()
        inserted = await run_in_threadpool(
            store.upsert,
            [chunk.to_dict() for chunk in chunks],
            safe_name,
        )
        upload_timing["upsert_seconds"] = round(time.perf_counter() - upsert_started, 3)
        upload_timing["total_seconds"] = round(time.perf_counter() - upload_started, 3)
        timing_debug["upload"] = dict(upload_timing)
        timing_debug["embedding"] = dict(store.last_upsert_timing)
        logger.info(
            "Document indexed",
            extra={
                "source_file": safe_name,
                "document_id": document.document_id,
                "chunks": inserted,
                "bytes": total_bytes,
            },
        )
        return UploadResponse(
            message=f"Processed {inserted} chunks",
            file_name=document.filename,
            document_id=document.document_id,
            document_type=document.metadata.get("document_type") or document.mime_type,
            chunks_processed=inserted,
            collection_stats=await run_in_threadpool(get_store().stats),
            debug={
                "timing": timing_debug,
                "extraction": document.metadata.get("extraction_report", {}),
                "structure": document.metadata.get("ielts_structure", {}).get("diagnostics", {}),
                "outline": document.metadata.get("ielts_structure", {}).get("outline", {}),
            },
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Document processing failed for %s", safe_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected document processing failure for %s", safe_name)
        raise HTTPException(status_code=500, detail="Không thể xử lý tài liệu này.") from exc
    finally:
        file_path.unlink(missing_ok=True)


@app.post("/rag/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung tìm kiếm")
    results = await run_in_threadpool(
        get_store().search,
        query,
        req.top_k,
        req.document_ids,
    )
    return SearchResponse(query=query, results=results)


@app.get("/rag/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    return StatsResponse(**(await run_in_threadpool(get_store().stats)))
