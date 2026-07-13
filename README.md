# IELTS Chatbot Standalone

Standalone chatbot repo extracted from the IELTS learning system.

It includes:

- FastAPI backend
- React/Vite frontend
- Ollama LLM integration
- Document RAG for text, PDF, DOCX, and images using an embedded local vector store and an LLM router

## Architecture

```text
Browser
-> React frontend
-> FastAPI backend
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
-> deterministic document-intent guard, then LLM router for ambiguous queries
-> retrieve context
-> Ollama answer with context
```

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
uvicorn app.main:app --host 0.0.0.0 --port 2222
```

Start frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:8000
```

Warm up large models before opening the UI:

```bash
curl -s -X POST http://127.0.0.1:2222/warmup
```

This loads the Ollama LLM, embedding model, and PaddleOCR model up front so the first real user request is smoother.

For Colab CPU, keep `paddlepaddle` below `3.3` for now. `paddlepaddle==3.3.1` has been observed to crash PaddleOCR PP-OCRv6 inference with `ConvertPirAttribute2RuntimeAttribute ... onednn_instruction.cc` even when oneDNN/PIR flags are disabled.

## Run On Colab

Use the companion notebook kept outside this repo in the project folder:

```text
../IELTS_Chatbot_Standalone_Colab.ipynb
```

Set:

```python
REPO_URL = "https://github.com/phamtu2x5/ielts-chatbot"
```

Then run all cells. The last cell prints a `trycloudflare.com` URL for the frontend.

## Important Environment Variables

```env
OLLAMA_API_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=hf.co/Zkare/Chatbot_Ielts_Assistant_v2:Q4_K_M
OLLAMA_NUM_PREDICT=1200
OLLAMA_NUM_CTX=4096
OLLAMA_TIMEOUT_SECONDS=180
EMBEDDING_MODEL_NAME=BAAI/bge-m3
UPLOAD_DIR=uploads
RAG_DATA_DIR=data/rag
CORS_ALLOW_ORIGINS=*
RAG_TOP_K=5
RAG_MIN_SCORE=0.45
RAG_PROBE_TOP_K=3
RAG_PROBE_MIN_DENSE_SCORE=0.35
RAG_OVERVIEW_TOP_K=8
RAG_OVERVIEW_SOURCE_CHARS=900
```

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
OCR_ENGINE=paddle
PADDLEOCR_DEVICE=cpu
PADDLEOCR_LANG=latin
PADDLEOCR_DET_MODEL=PP-OCRv6_medium_det
PADDLEOCR_REC_MODEL=PP-OCRv6_medium_rec
PADDLEOCR_DISABLE_ONEDNN=1
FLAGS_use_mkldnn=0
FLAGS_use_onednn=0
FLAGS_enable_pir_api=0
FLAGS_enable_pir_in_executor=0
WARMUP_LLM=true
WARMUP_EMBEDDING=true
WARMUP_OCR=true
```

Runtime paths are resolved relative to `backend/` unless an absolute path is configured. Uploaded source files are temporary; persistent chunks and embeddings are stored under `backend/data/rag/` by default.

On Colab CPU, PaddleOCR may fail inside Paddle's oneDNN/PIR runtime. Keep the `FLAGS_*` variables above in the backend process environment before importing Paddle/PaddleOCR. `/warmup` must report `ocr.ok=true` before uploading images or scanned PDFs.

The document pipeline uses one OCR model by default: PP-OCRv6 medium. PP-OCRv6 small and PP-StructureV3 are not loaded in the streamlined Colab pipeline.

## Tests

```bash
python -m unittest discover -s backend/tests -v
cd frontend && npm run build
```

## Notes

This repo intentionally does not include:

- PostgreSQL database RAG from the full IELTS platform
- Milvus/etcd/minio stack
- Auth/admin/teacher/student modules
- Writing/Speaking grading modules
- DOC legacy, Excel, PowerPoint, audio, and video ingestion

Those can be reconnected later if needed.
