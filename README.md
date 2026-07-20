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
-> retrieve compact document/section routing candidates
-> one structured LLM gateway decides direct/RAG/clarify, intent, and target documents
-> validate the decision and retrieve only inside the selected document scope
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

This loads the Ollama LLM, embedding model, and RapidOCR model up front so the first real user request is smoother.

The Colab runtime is configured for OCR with RapidOCR and PyTorch CUDA. DocLayout-YOLO can detect table/figure/layout regions before OCR/table parsing. PaddleOCR, PaddlePaddle, Tesseract, PP-StructureV3, and ONNX Runtime are not part of the streamlined runtime.

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
OLLAMA_THINK=false
EMBEDDING_MODEL_NAME=BAAI/bge-m3
UPLOAD_DIR=uploads
RAG_DATA_DIR=data/rag
CORS_ALLOW_ORIGINS=*
RAG_TOP_K=5
RAG_MIN_SCORE=0.45
RAG_PROBE_TOP_K=3
RAG_PROBE_MIN_DENSE_SCORE=0.35
RAG_RRF_K=60
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
cd frontend && npm run build
```

To collect end-to-end answers and RAG diagnostics for manual review, start the
backend with all models warmed up, then run:

```bash
python backend/tools/chat_evaluation.py --base-url http://127.0.0.1:2222
```

The runner verifies and uploads all seven files in `docs/`, sends the 66 questions
from `backend/evaluation/chat_corpus_v2.json`, and writes the raw answers, routes,
sources and debug metadata under `backend/data/chat_evaluation/`. It does not
score answer quality; the report is reviewed manually. Use `--skip-upload` when
the same corpus is already indexed, or repeat `--case CASE_ID` to collect selected
cases. The direct-router cases deliberately run while all uploaded document IDs
are active, so they expose false RAG routing without tying the question to a
specific document. The previous 19-question set is retired because it does not
represent the expanded corpus.

## Notes

This repo intentionally does not include:

- PostgreSQL database RAG from the full IELTS platform
- Milvus/etcd/minio stack
- Auth/admin/teacher/student modules
- Writing/Speaking grading modules
- DOC legacy, Excel, PowerPoint, audio, and video ingestion

Those can be reconnected later if needed.
