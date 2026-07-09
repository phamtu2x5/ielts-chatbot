import os
from pathlib import Path

import aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .llm import OLLAMA_MODEL, OLLAMA_NUM_PREDICT, classify_route, direct_prompt, query_ollama, rag_prompt
from .pdf_utils import chunk_text, extract_pdf_text
from .rag import get_store
from .schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse, StatsResponse, UploadResponse


load_dotenv()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
        text = source.get("text", "")
        parts.append(f"[Source {index}: {source_file}]\n{text}")
    return "\n\n".join(parts)


@app.get("/health")
async def health() -> dict:
    stats = get_store().stats()
    return {
        "status": "ok",
        "pdf_rag_documents": stats["documents"],
        "pdf_rag_chunks": stats["chunks"],
        "ollama_api_url": os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate"),
        "ollama_model": OLLAMA_MODEL,
        "ollama_num_predict": OLLAMA_NUM_PREDICT,
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


@app.post("/rag/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hiện chỉ hỗ trợ tệp PDF")

    file_path = UPLOAD_DIR / Path(file.filename).name
    async with aiofiles.open(file_path, "wb") as out:
        await out.write(await file.read())

    try:
        text = extract_pdf_text(file_path)
        chunks = chunk_text(
            text,
            source_file=file_path.name,
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
        )
        if not chunks:
            raise HTTPException(status_code=400, detail="Không trích xuất được văn bản từ PDF")

        inserted = get_store().upsert(chunks, source_file=file_path.name)
        return UploadResponse(
            message=f"Processed {inserted} chunks",
            file_name=file_path.name,
            chunks_processed=inserted,
            collection_stats=get_store().stats(),
        )
    finally:
        file_path.unlink(missing_ok=True)


@app.post("/rag/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    return SearchResponse(query=req.query, results=get_store().search(req.query, top_k=req.top_k))


@app.get("/rag/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    return StatsResponse(**get_store().stats())
