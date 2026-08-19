# IELTS Chatbot Backend

Standalone chatbot repo extracted from the IELTS learning system.

This repository includes:

- FastAPI backend
- Ollama LLM integration
- Document RAG for text, PDF, DOCX, and images using an embedded local vector store and an LLM router

The React/Vite application lives in the separate
[`ielts-chatbot-fe`](https://github.com/phamtu2x5/ielts-chatbot-fe) repository.

## Architecture

```text
External web frontend
-> authenticated FastAPI backend
-> Ollama
-> Zkare IELTS chatbot model
```

For document RAG:

```text
Upload text/PDF/DOCX/image
-> route by file type
-> extract native text first
-> OCR only pages/images that need it
-> normalize into structured document elements
-> reconcile duplicate native/OCR content
-> parse IELTS Passage/Question Group/Question structure when present
-> structure-aware chunks, with semantic chunk fallback for general documents
-> sentence-transformers embedding
-> local vector store
-> Patch 0: semantic gateway returns only {"route":"direct|rag"}
-> resolve the target document from same-turn attachments, exact catalog metadata,
   semantic catalog selection, or weak conversation affinity
-> Patch 1: classify the RAG action with a separate enum-only model call
-> run structured lookup or metadata-filtered retrieval inside the resolved scope
-> deterministic renderer or Ollama answer with grounded context
```

Conversation history, user facts, and document affinity are owned by the backend
and stored under the same session directory as that session's RAG index. The
client-carried `conversation_history` and `conversation_state` fields remain in
the request schema for compatibility but are not trusted as session memory.
Successful document affinity remains weak follow-up context and never forces a
later question into that file. Deleting or expiring a session removes its memory,
documents, embeddings, and cache together.
Only a bounded LRU set of inactive session indexes stays loaded in CPU RAM;
eviction never deletes on-disk session data. The shared BGE-M3 model and the
LLM/OCR/layout GPU runtimes remain resident.
External clients and the evaluation runner use this single chat endpoint.

### Current chat patch boundaries

The current baseline intentionally separates routing responsibilities:

1. **Patch 0 - direct/RAG gateway** receives the user message, filtered successful
   history, and compact route state. It returns only a JSON `direct` or `rag`
   classification. It does not answer, choose a file, or choose an RAG action.
   A `direct` decision is followed by the normal direct-generation prompt.
2. **Document resolution** runs only after a `rag` decision. A document attached in the
   current turn is an explicit allowed scope. Without a current attachment, all
   indexed documents are candidates; exact catalog references and the semantic
   target resolver choose the target. Previous RAG affinity is only a weak hint.
3. **Patch 1 - RAG intent classifier** runs only after document resolution and
   returns one allowed final intent enum such as `document_overview`,
   `show_questions`, `translate_questions`, `solve_questions`, or `semantic_qa`.
4. Structured lookup/retrieval and generation then operate only inside the
   resolved document scope.

## Run Locally

Start Ollama:

```bash
ollama serve
ollama pull hf.co/Zkare/Chatbot_Ielts_Assistant_v2:Q4_K_M
```

Start backend:

```bash
cd backend
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8765 --workers 1
```

For a local UI, run the separate `ielts-chatbot-fe` repository and point its
`VITE_CHATBOT_API_URL` at this backend or at a trusted server-side proxy.

Warm up large models before opening the UI:

```bash
curl -s -X POST http://127.0.0.1:8765/warmup
```

This loads the Ollama LLM, embedding model, and RapidOCR model up front so the first real user request is smoother.

The Colab runtime is configured for OCR with RapidOCR and PyTorch CUDA. DocLayout-YOLO can detect table/figure/layout regions before OCR/table parsing. PaddleOCR, PaddlePaddle, Tesseract, PP-StructureV3, and ONNX Runtime are not part of the streamlined runtime.

## Run On Colab

Use the companion notebook tracked at the repository root:

```text
IELTS_Chatbot_BE.ipynb
```

Set:

```python
REPO_URL = "https://github.com/phamtu2x5/ielts-chatbot"
```

Add two private Colab Secrets before running the notebook:

- `CLOUDFLARE_TUNNEL_TOKEN`: token of the named tunnel.
- `IELTS_API_TOKEN`: a fixed random value containing at least 32 characters.

Then run all cells. The current notebook starts Ollama, the FastAPI backend and
the named tunnel at `https://api.mywsite.online`. Document upload is disabled,
so BGE-M3, RapidOCR and DocLayout-YOLO are not warmed into RAM/VRAM. The
notebook does not install or start the frontend and does not run regression.

## Important Environment Variables

```env
OLLAMA_API_URL=http://127.0.0.1:11434/api/generate
OLLAMA_CHAT_API_URL=http://127.0.0.1:11434/api/chat
OLLAMA_CHAT_FALLBACK=true
OLLAMA_MODEL=hf.co/Zkare/Chatbot_Ielts_Assistant_v2:Q4_K_M
OLLAMA_NUM_PREDICT=2800
OLLAMA_NUM_CTX=4096
OLLAMA_TIMEOUT_SECONDS=180
OLLAMA_THINK=false
EMBEDDING_MODEL_NAME=BAAI/bge-m3
UPLOAD_DIR=uploads
RAG_DATA_DIR=data/rag
CORS_ALLOW_ORIGINS=http://localhost:8000,http://127.0.0.1:8000,https://mywsite.online,https://www.mywsite.online
API_AUTH_REQUIRED=false
API_AUTH_TOKEN=
DEBUG_PAYLOADS=false
DOCUMENT_UPLOAD_ENABLED=true
RAG_TOP_K=5
RAG_MIN_SCORE=0.45
RAG_PROBE_TOP_K=3
RAG_PROBE_MIN_DENSE_SCORE=0.35
RAG_RRF_K=60
RAG_OVERVIEW_TOP_K=8
RAG_OVERVIEW_SOURCE_CHARS=900
RAG_SESSION_GRACE_TTL_SECONDS=300
RAG_SESSION_HARD_TTL_SECONDS=1800
RAG_SESSION_MAX_DOCUMENTS=30
RAG_SESSION_MAX_CHUNKS=15000
RAG_SESSION_CACHE_MAX_STORES=4
CHAT_RATE_LIMIT=30
CHAT_RATE_WINDOW_SECONDS=60
UPLOAD_RATE_LIMIT=30
UPLOAD_RATE_WINDOW_SECONDS=600
CHAT_MAX_CONCURRENCY=2
UPLOAD_MAX_CONCURRENCY=1
```

Run the local JSON/NumPy RAG backend with exactly one Uvicorn worker. Its
in-process locks do not coordinate writes across multiple worker processes.
For the public Colab tunnel, the notebook overrides `API_AUTH_REQUIRED=true`
and `DEBUG_PAYLOADS=false`. Protected requests use
`Authorization: Bearer <IELTS_API_TOKEN>`. Do not embed this shared token in a
public JavaScript bundle; the production web should inject it through a trusted
server-side proxy or replace it with per-user authentication.

`GET /health` is intentionally lightweight and public for Cloudflare health
checks. Detailed disk/session/storage counters are available from the protected
`GET /admin/stats` endpoint.

Document ingestion settings:

```env
DOCUMENT_MAX_UPLOAD_MB=25
DOCUMENT_MAX_PDF_PAGES=80
DOCUMENT_CHUNK_TARGET_TOKENS=600
DOCUMENT_CHUNK_MAX_TOKENS=800
DOCUMENT_CHUNK_OVERLAP_TOKENS=80
DOCUMENT_ENABLE_IELTS_STRUCTURE=true
DOCUMENT_OCR_DUPLICATE_SIMILARITY=0.88
DOCUMENT_OCR_DUPLICATE_TOKEN_OVERLAP=0.92
DOCUMENT_OCR_MIN_NEW_TOKEN_RATIO=0.08
DOCUMENT_OCR_DPI=180
DOCUMENT_CONNECTOR_ENABLE=true
DOCUMENT_CONNECTOR_MIN_COMPONENT_AREA_RATIO=0.0015
DOCUMENT_CONNECTOR_MAX_COMPONENT_AREA_RATIO=0.08
DOCUMENT_CONNECTOR_MIN_SPAN_RATIO=0.07
DOCUMENT_CONNECTOR_DIRECTION_MIN_CONFIDENCE=0.55
DOCUMENT_VISUAL_SPATIAL_ASSOCIATION_DISTANCE_RATIO=0.16
DOCUMENT_VISUAL_DIRECTION_FORWARD_WEIGHT=0.15
OCR_ENGINE=rapidocr
OCR_RUNTIME=torch
OCR_DEVICE=cuda:0
OCR_LANG=en
OCR_DET_LANG=ch
OCR_VERSION=PP-OCRv6
OCR_MODEL_SIZE=medium
OCR_MIN_CONFIDENCE=0.72
LAYOUT_ENABLE=true
LAYOUT_ENGINE=doclayout_yolo
LAYOUT_DEVICE=cuda:0
LAYOUT_MODEL_REPO=juliozhao/DocLayout-YOLO-DocStructBench
LAYOUT_MODEL_FILENAME=doclayout_yolo_docstructbench_imgsz1024.pt
LAYOUT_MODEL_PATH=
LAYOUT_CONFIDENCE=0.25
LAYOUT_IMAGE_SIZE=1024
WARMUP_LLM=true
WARMUP_EMBEDDING=true
WARMUP_OCR=true
WARMUP_LAYOUT=true
```

Runtime paths are resolved relative to `backend/` unless an absolute path is configured. Uploaded source files are temporary; persistent chunks and embeddings are stored under `backend/data/rag/` by default.

The backend expects RapidOCR, CUDA-enabled PyTorch, and DocLayout-YOLO to be importable. `/warmup` must report `ocr.ok=true` before uploading images or scanned PDFs. OCR and DocLayout-YOLO use `cuda:0` by default. Layout warmup is enabled by default so the first document upload does not pay the DocLayout model load cost.

The document pipeline uses one OCR path by default: RapidOCR with PyTorch CUDA using PP-OCRv6 medium. DocLayout-YOLO is used only for visual region detection; it does not OCR text or parse table cells by itself. PP-StructureV3, PaddleOCR, Tesseract, and ONNX Runtime are not loaded in the streamlined Colab pipeline.

The extraction baseline is frozen at parser version `1.10.0`. The corpus regression reached zero
failed documents; isolated OCR ambiguity remains preserved as raw text with
degraded visual-quality metadata instead of being repaired with document-specific
rules. Reopen extraction work only for a reproducible issue across multiple
documents or a production-blocking failure.

## Tests

```bash
python -m unittest discover -s backend/tests -v
```

## Notes

This repo intentionally does not include:

- PostgreSQL database RAG from the full IELTS platform
- Milvus/etcd/minio stack
- Auth/admin/teacher/student modules
- Writing/Speaking grading modules
- DOC legacy, Excel, PowerPoint, audio, and video ingestion

Those can be reconnected later if needed.
