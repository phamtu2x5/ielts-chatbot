import json
import logging
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
            passage_numbers = item.get("passage_numbers") or []
            lines.append(
                f"- {item.get('source_file', 'unknown')} | chunks={item.get('chunks', 0)} | {page_label}"
                + (f" | type={mime_types}" if mime_types else "")
                + (f" | units={unit_types}" if unit_types else "")
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


@dataclass
class ChatPreparation:
    prompt: str | None
    static_response: str | None
    route_used: str
    sources: list[dict[str, Any]]
    debug: dict[str, Any]


async def prepare_chat(req: ChatRequest) -> ChatPreparation:
    message = req.message.strip()
    store = get_store()
    route = "direct"
    sources: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    probe: dict[str, Any] = {"results": []}

    stats = await run_in_threadpool(store.stats)
    if stats["chunks"] > 0:
        probe_top_k = max(settings.rag_probe_top_k, settings.rag_top_k)
        probe, catalog = await run_in_threadpool(store.probe_with_catalog, message, probe_top_k)

        if probe.get("has_document_intent"):
            route = "rag"
        else:
            document_context = format_router_document_context(catalog, probe)
            route = await classify_route(message, req.conversation_history, document_context)

        if route == "direct" and (
            probe.get("is_overview") or probe.get("top_question_score", 0.0) >= 1
        ):
            route = "rag"

    if route == "rag":
        if probe.get("is_overview"):
            sources = await run_in_threadpool(store.overview, settings.rag_overview_top_k)
            for source in sources:
                source["probe_overview_score"] = 1.0
        elif probe.get("has_strong_hits"):
            sources = (probe.get("results") or [])[: settings.rag_top_k]
        else:
            sources = await run_in_threadpool(store.search, message, settings.rag_top_k)

    debug = {
        "route_decision": route,
        "catalog": catalog,
        "probe": compact_probe_debug(probe),
        "source_count": len(sources),
    }

    if sources:
        context = (
            format_context(sources, max_chars_per_source=settings.rag_overview_source_chars)
            if probe.get("is_overview")
            else format_context(sources)
        )
        return ChatPreparation(
            prompt=rag_prompt(message, context, req.conversation_history),
            static_response=None,
            route_used="vector_rag",
            sources=sources,
            debug=debug,
        )

    if route == "rag":
        return ChatPreparation(
            prompt=None,
            static_response=NO_RAG_MATCH_RESPONSE,
            route_used="vector_rag_no_match",
            sources=[],
            debug=debug,
        )

    return ChatPreparation(
        prompt=direct_prompt(message, req.conversation_history),
        static_response=None,
        route_used="base_model",
        sources=[],
        debug=debug,
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
                    yield stream_event("error", message="Không nhận được nội dung trả lời từ model. Vui lòng thử lại.")
                    return
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
                detail="Không trích xuất được văn bản từ tài liệu. File có thể quá mờ, không có chữ, hoặc OCR chưa phù hợp.",
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
