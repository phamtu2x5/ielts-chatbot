import os
import time
from pathlib import Path

import aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .document_pipeline import DocumentProcessor
from .llm import OLLAMA_MODEL, OLLAMA_NUM_PREDICT, classify_route, direct_prompt, query_ollama, rag_prompt
from .rag import get_store
from .schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse, StatsResponse, UploadResponse


load_dotenv()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENT_PROCESSOR = DocumentProcessor()

app = FastAPI(title="Standalone IELTS Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_context(sources: list[dict]) -> str:
    parts = []
    for index, source in enumerate(sources, 1):
        source_file = source.get("source_file", "unknown")
        pages = source.get("pages") or []
        page_label = f", pages {', '.join(str(page) for page in pages)}" if pages else ""
        text = source.get("text", "")
        parts.append(f"[Source {index}: {source_file}{page_label}]\n{text}")
    return "\n\n".join(parts)


@app.get("/health")
async def health() -> dict:
    stats = get_store().stats()
    return {
        "status": "ok",
        "document_rag_documents": stats["documents"],
        "document_rag_chunks": stats["chunks"],
        "pdf_rag_documents": stats["documents"],
        "pdf_rag_chunks": stats["chunks"],
        "ollama_api_url": os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate"),
        "ollama_model": OLLAMA_MODEL,
        "ollama_num_predict": OLLAMA_NUM_PREDICT,
    }


@app.post("/warmup")
async def warmup() -> dict:
    started = time.perf_counter()
    results = {}

    if os.getenv("WARMUP_LLM", "true").lower() == "true":
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

    if os.getenv("WARMUP_EMBEDDING", "true").lower() == "true":
        embedding_started = time.perf_counter()
        try:
            results["embedding"] = {
                "ok": True,
                "duration_seconds": round(time.perf_counter() - embedding_started, 2),
                **get_store().warmup(),
            }
        except Exception as exc:
            results["embedding"] = {"ok": False, "error": str(exc)}
    else:
        results["embedding"] = {"skipped": True}

    ocr_started = time.perf_counter()
    try:
        ocr_result = DOCUMENT_PROCESSOR.warmup_ocr()
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
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung câu hỏi")

    store = get_store()
    route = "direct"
    sources = []
    if store.stats()["chunks"] > 0:
        route = await classify_route(message, req.conversation_history)

    if route == "rag":
        top_k = int(os.getenv("RAG_TOP_K", "5"))
        sources = store.search(message, top_k=top_k)

    if sources:
        prompt = rag_prompt(message, format_context(sources), req.conversation_history)
        answer = await query_ollama(prompt)
        return ChatResponse(response=answer, route_used="vector_rag", sources=sources)

    prompt = direct_prompt(message, req.conversation_history)
    answer = await query_ollama(prompt)
    route_used = "base_model_no_rag_match" if route == "rag" else "base_model"
    return ChatResponse(response=answer, route_used=route_used, sources=None)


@app.post("/documents/upload", response_model=UploadResponse)
@app.post("/rag/upload-pdf", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Tên tệp không hợp lệ")
    file_path = UPLOAD_DIR / Path(file.filename).name
    async with aiofiles.open(file_path, "wb") as out:
        await out.write(await file.read())

    try:
        document, chunks = DOCUMENT_PROCESSOR.process_file(file_path, file_path.name, file.content_type)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Không trích xuất được văn bản từ tài liệu. File có thể quá mờ, không có chữ, hoặc OCR chưa phù hợp.",
            )

        inserted = get_store().upsert([chunk.to_dict() for chunk in chunks], source_file=file_path.name)
        return UploadResponse(
            message=f"Processed {inserted} chunks",
            file_name=document.filename,
            document_id=document.document_id,
            document_type=document.mime_type,
            chunks_processed=inserted,
            collection_stats=get_store().stats(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        file_path.unlink(missing_ok=True)


@app.post("/rag/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    return SearchResponse(query=req.query, results=get_store().search(req.query, top_k=req.top_k))


@app.get("/rag/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    return StatsResponse(**get_store().stats())
