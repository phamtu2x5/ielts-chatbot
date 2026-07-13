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
from .document_pipeline import DocumentProcessor
from .intent import dedupe_sources, detect_query_intent, filter_sources_for_intent
from .llm import OLLAMA_MODEL, OLLAMA_NUM_PREDICT, classify_route, direct_prompt, query_ollama, rag_prompt, stream_ollama
from .rag import get_store
from .schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse, StatsResponse, UploadResponse


logger = logging.getLogger(__name__)

UPLOAD_DIR = settings.upload_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENT_PROCESSOR = DocumentProcessor()

app = FastAPI(title="Standalone IELTS Chatbot", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def stream_event(event_type: str, **payload: Any) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


def format_context(sources: list[dict], max_chars_per_source: int | None = None) -> str:
    parts = []
    for index, source in enumerate(sources, 1):
        source_file = source.get("source_file", "unknown")
        pages = source.get("pages") or []
        page_label = f", pages {', '.join(str(page) for page in pages)}" if pages else ""
        text = source.get("display_text") or source.get("text", "")
        if max_chars_per_source and len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rsplit(" ", 1)[0] + " ..."
        parts.append(f"[Source {index}: {source_file}{page_label}]\n{text}")
    return "\n\n".join(parts)


def format_router_document_context(catalog: list[dict], probe: dict) -> str:
    lines = []
    if catalog:
        lines.append("Uploaded documents:")
        for item in catalog:
            pages = item.get("pages") or []
            page_label = f"pages {pages[0]}-{pages[-1]}" if pages else "pages unknown"
            mime_types = ", ".join(item.get("mime_types") or [])
            unit_types = ", ".join(item.get("unit_types") or [])
            document_types = ", ".join(item.get("document_types") or [])
            task_types = ", ".join(item.get("task_types") or [])
            passage_numbers = item.get("passage_numbers") or []
            lines.append(
                f"- {item.get('source_file', 'unknown')} | chunks={item.get('chunks', 0)} | {page_label}"
                + (f" | type={mime_types}" if mime_types else "")
                + (f" | units={unit_types}" if unit_types else "")
                + (f" | doc_type={document_types}" if document_types else "")
                + (f" | task_type={task_types}" if task_types else "")
                + (f" | passages={passage_numbers}" if passage_numbers else "")
            )

    results = probe.get("results") or []
    lines.append(
        "Retrieval probe strength: "
        + ("strong" if probe.get("has_strong_hits") else "weak_or_none")
    )
    if results:
        lines.append("Retrieval probe top hits:")
        for index, result in enumerate(results[:3], 1):
            source_file = result.get("source_file", "unknown")
            pages = result.get("pages") or []
            text = " ".join((result.get("display_text") or result.get("text") or "").split())[:260]
            dense = result.get("probe_dense_score", result.get("score", 0.0))
            keyword = result.get("probe_keyword_score", 0.0)
            question = result.get("probe_question_score", 0.0)
            overview = result.get("probe_overview_score", 0.0)
            lines.append(
                f"{index}. {source_file} pages={pages} dense={dense:.3f} keyword={keyword:.1f} "
                f"question={question:.1f} overview={overview:.1f}: {text}"
            )
    else:
        lines.append("Retrieval probe top hits: none")

    return "\n".join(lines)


def compact_probe_debug(probe: dict) -> dict:
    return {
        "has_hits": probe.get("has_hits", False),
        "has_strong_hits": probe.get("has_strong_hits", False),
        "top_score": probe.get("top_score", 0.0),
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


def document_extraction_failure_detail(document: Any) -> str:
    metadata = document.metadata or {}
    ocr_engine = metadata.get("ocr_engine")
    ocr_metadata = metadata.get("ocr_metadata") or {}
    attempts = ocr_metadata.get("cascade_attempts") or []
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

    if ocr_engine == "paddleocr_failed":
        if any("ConvertPirAttribute2RuntimeAttribute" in error or "onednn_instruction" in error for error in errors):
            return (
                "PaddleOCR đã được gọi nhưng lỗi ở runtime Paddle oneDNN/PIR, nên ảnh chưa được OCR. "
                "Hãy restart backend sau khi tắt FLAGS_use_mkldnn, FLAGS_use_onednn, "
                "FLAGS_enable_pir_api và FLAGS_enable_pir_in_executor, rồi upload lại ảnh."
            )
        if any("paddleocr_unavailable" in reason for reason in reasons):
            return (
                "PaddleOCR chưa khả dụng trong môi trường backend hiện tại, nên ảnh chưa được OCR. "
                "Hãy cài đúng paddlepaddle/paddleocr rồi restart backend."
            )
        if errors:
            return f"PaddleOCR không trích xuất được ảnh. Lỗi OCR đầu tiên: {errors[0][:300]}"

    return "Không trích xuất được văn bản từ tài liệu. File có thể quá mờ, không có chữ, hoặc OCR chưa phù hợp."


def generation_fallback(prepared: "ChatPreparation") -> str:
    if prepared.route_used.startswith("vector_rag"):
        return NO_RAG_MATCH_RESPONSE
    return "Mình chưa nhận được nội dung trả lời từ model. Vui lòng thử lại."


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


def _lookup_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)
        if len(term) > 1 or term.isdigit()
    }


def _row_match_score(message: str, row_label: Any) -> float:
    label = str(row_label).strip()
    if not label:
        return 0.0
    lowered = message.lower()
    label_lower = label.lower()
    if re.search(rf"(?<!\w){re.escape(label_lower)}(?!\w)", lowered):
        return 10.0
    overlap = _lookup_terms(message) & _lookup_terms(label)
    return float(len(overlap))


def _column_match_score(message: str, column_label: Any) -> float:
    label = str(column_label).strip()
    if not label:
        return 0.0
    query_terms = _lookup_terms(message)
    column_terms = _lookup_terms(label)
    score = float(len(query_terms & column_terms))
    query_years = set(re.findall(r"\b\d{4}\b", message))
    column_years = set(re.findall(r"\b\d{4}\b", label))
    if query_years and query_years & column_years:
        score += 4.0
    return score


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
        if len(columns) < 2 or not rows:
            continue
        column_scores = [
            (index, _column_match_score(message, column))
            for index, column in enumerate(columns[1:], 1)
        ]
        column_scores = [(index, score) for index, score in column_scores if score > 0]
        if not column_scores:
            continue
        target_index, column_score = max(column_scores, key=lambda item: item[1])
        for row in rows:
            if not row or len(row) <= target_index:
                continue
            row_score = _row_match_score(message, row[0])
            if row_score <= 0:
                continue
            score = row_score + column_score
            if best_match is None or score > best_match[0]:
                best_match = (score, row[target_index], source)
    if best_match is None:
        return None
    _, value, source = best_match
    return f"{value}\n\nNguồn: {_source_label(source)}."


def static_response_for_sources(message: str, query_intent: str, sources: list[dict[str, Any]]) -> str | None:
    cell_answer = _lookup_table_cell(message, sources)
    if cell_answer:
        return cell_answer

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


async def prepare_chat(req: ChatRequest) -> ChatPreparation:
    message = req.message.strip()
    store = get_store()
    route = "direct"
    sources: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    probe: dict[str, Any] = {"results": []}
    query_intent = "direct"

    stats = await run_in_threadpool(store.stats)
    if stats["chunks"] > 0:
        probe_top_k = max(settings.rag_probe_top_k, settings.rag_top_k)
        probe, catalog = await run_in_threadpool(store.probe_with_catalog, message, probe_top_k)
        query_intent = detect_query_intent(message, probe)

        if probe.get("has_document_intent"):
            route = "rag"
        else:
            document_context = format_router_document_context(catalog, probe)
            route = await classify_route(message, req.conversation_history, document_context)

        if route == "direct" and (
            probe.get("is_overview") or probe.get("top_question_score", 0.0) >= 1
        ):
            route = "rag"
    else:
        query_intent = "direct"

    if route == "rag":
        structured_sources = await run_in_threadpool(
            store.structured_lookup,
            message,
            query_intent,
            max(settings.rag_top_k, settings.rag_overview_top_k),
        )
        retrieval_method = "structured" if structured_sources else None
        if structured_sources:
            sources = structured_sources[: settings.rag_top_k]
        elif probe.get("is_overview"):
            sources = await run_in_threadpool(store.overview, settings.rag_overview_top_k)
            for source in sources:
                source["probe_overview_score"] = 1.0
            retrieval_method = "overview"
        elif probe.get("has_strong_hits"):
            sources = (probe.get("results") or [])[: settings.rag_top_k]
            retrieval_method = "probe"
        else:
            sources = await run_in_threadpool(store.search, message, settings.rag_top_k)
            retrieval_method = "dense"
        before_filter_count = len(sources)
        sources = filter_sources_for_intent(sources, message, query_intent)
        if query_intent == "solve_questions" and sources:
            question_context = await run_in_threadpool(store.question_context_for_sources, sources, 8)
            expansion = await run_in_threadpool(store.passage_context_for_sources, sources, 3)
            sources = dedupe_sources(sources + question_context + expansion)
        sources = dedupe_sources(sources)
    else:
        structured_sources = []
        retrieval_method = None
        before_filter_count = 0

    debug = {
        "route_decision": route,
        "query_intent": query_intent,
        "catalog": catalog,
        "probe": compact_probe_debug(probe),
        "retrieval": {
            "method": retrieval_method,
            "structured_hits": len(structured_sources),
            "before_filter_count": before_filter_count,
            "after_filter_count": len(sources),
            "final_context": compact_final_context_debug(sources),
        },
        "source_count": len(sources),
    }

    if sources:
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
        context = (
            format_context(sources, max_chars_per_source=settings.rag_overview_source_chars)
            if probe.get("is_overview")
            else format_context(sources)
        )
        return ChatPreparation(
            prompt=rag_prompt(message, context, req.conversation_history, query_intent=query_intent),
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
        prompt=direct_prompt(message, req.conversation_history),
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
    started = time.perf_counter()
    results = {}

    if settings.warmup_llm:
        llm_started = time.perf_counter()
        try:
            response = await query_ollama(
                "Warm up the IELTS assistant. Reply with exactly: ready",
                temperature=0.0,
                num_predict=8,
            )
            results["llm"] = {
                "ok": True,
                "model": OLLAMA_MODEL,
                "duration_seconds": round(time.perf_counter() - llm_started, 2),
                "sample": response[:120],
            }
        except Exception as exc:
            results["llm"] = {"ok": False, "error": str(exc)}
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
    return {
        "status": "ok" if ok else "partial",
        "duration_seconds": round(time.perf_counter() - started, 2),
        "results": results,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung câu hỏi")

    prepared = await prepare_chat(req)
    answer = prepared.static_response
    if answer is None and prepared.prompt is not None:
        try:
            answer = await query_ollama(prepared.prompt)
            if not answer.strip():
                answer = generation_fallback(prepared)
        except Exception as exc:
            logger.exception("Chat generation failed")
            raise HTTPException(
                status_code=502,
                detail="Không thể kết nối hoặc nhận câu trả lời từ Ollama.",
            ) from exc
    return ChatResponse(
        response=answer or "",
        route_used=prepared.route_used,
        sources=prepared.sources,
        debug=prepared.debug,
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
            )
            if prepared.static_response is not None:
                yield stream_event("token", token=prepared.static_response)
                yield stream_event("done")
                return

            yield stream_event("status", message="Đang soạn câu trả lời...")
            has_token = False
            async for token in stream_ollama(prepared.prompt or ""):
                has_token = True
                yield stream_event("token", token=token)
            if not has_token:
                logger.warning("Ollama stream completed without visible tokens; retrying with non-stream request")
                fallback_answer = await query_ollama(prepared.prompt or "")
                if not fallback_answer.strip():
                    fallback_answer = generation_fallback(prepared)
                yield stream_event("token", token=fallback_answer)
            yield stream_event("done")
        except Exception:
            logger.exception("Streaming chat failed")
            yield stream_event("error", message="Không thể tạo câu trả lời lúc này. Vui lòng thử lại.")

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/documents/upload", response_model=UploadResponse)
@app.post("/rag/upload-pdf", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Tên tệp không hợp lệ")

    safe_name = Path(file.filename).name
    file_path = UPLOAD_DIR / f"{uuid4().hex}-{safe_name}"
    max_bytes = DOCUMENT_PROCESSOR.config.max_upload_mb * 1024 * 1024
    try:
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

        document, chunks = await run_in_threadpool(
            DOCUMENT_PROCESSOR.process_file,
            file_path,
            safe_name,
            file.content_type,
        )
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=document_extraction_failure_detail(document),
            )

        inserted = await run_in_threadpool(
            get_store().upsert,
            [chunk.to_dict() for chunk in chunks],
            safe_name,
        )
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
            document_type=document.mime_type,
            chunks_processed=inserted,
            collection_stats=await run_in_threadpool(get_store().stats),
            debug={
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
    results = await run_in_threadpool(get_store().search, query, req.top_k)
    return SearchResponse(query=query, results=results)


@app.get("/rag/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    return StatsResponse(**(await run_in_threadpool(get_store().stats)))
