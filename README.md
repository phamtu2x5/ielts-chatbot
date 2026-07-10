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
-> semantic chunk
-> sentence-transformers embedding
-> local vector store
-> LLM router decides direct answer vs document retrieval
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

This loads the Ollama LLM, embedding model, and PaddleOCR models up front so the first real user request is smoother.

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
EMBEDDING_MODEL_NAME=BAAI/bge-m3
RAG_PROBE_TOP_K=3
RAG_PROBE_MIN_DENSE_SCORE=0.35
RAG_OVERVIEW_TOP_K=6
```

Document ingestion settings:

```env
DOCUMENT_MAX_UPLOAD_MB=25
DOCUMENT_MAX_PDF_PAGES=80
DOCUMENT_CHUNK_TARGET_TOKENS=600
DOCUMENT_CHUNK_MAX_TOKENS=800
DOCUMENT_CHUNK_OVERLAP_TOKENS=80
DOCUMENT_OCR_LANG=vie+eng
DOCUMENT_OCR_DPI=180
OCR_ENGINE=paddle
OCR_FALLBACK_ENGINE=tesseract
PADDLEOCR_DEFAULT_DET_MODEL=PP-OCRv6_small_det
PADDLEOCR_DEFAULT_REC_MODEL=PP-OCRv6_small_rec
PADDLEOCR_FALLBACK_DET_MODEL=PP-OCRv6_medium_det
PADDLEOCR_FALLBACK_REC_MODEL=PP-OCRv6_medium_rec
WARMUP_LLM=true
WARMUP_EMBEDDING=true
WARMUP_OCR=true
WARMUP_OCR_MEDIUM=true
```

For a lighter Colab RAG embedding model, use:

```env
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-base
```

## Notes

This repo intentionally does not include:

- PostgreSQL database RAG from the full IELTS platform
- Milvus/etcd/minio stack
- Auth/admin/teacher/student modules
- Writing/Speaking grading modules
- DOC legacy, Excel, PowerPoint, audio, and video ingestion

Those can be reconnected later if needed.
