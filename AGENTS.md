# AGENTS.md

Project-level instructions for AI agents working on this repository.

These instructions apply to the whole repo unless a more specific `AGENTS.md`
is added in a subdirectory.

## 1. Project Overview

This repo is a standalone IELTS chatbot extracted from a larger IELTS learning
system. It provides:

- A FastAPI backend.
- A React/Vite frontend.
- Ollama-based LLM answering.
- Document AI + RAG for TXT/Markdown, PDF, DOCX, and images.
- IELTS-aware document structure parsing for Reading passages, question groups,
  individual questions, and Writing Task 1 tables when available.

The current product goal is not to overfit one sample PDF. The goal is:

```text
Resolve the target document and structured unit first.
Use semantic retrieval only for evidence and explanation.
Let the LLM solve or infer only when the user explicitly asks for it.
```

The repo is actively evolving. Prefer the current code and `README.md` over
older handoff notes if they disagree.

## 2. Tech Stack

Backend:

- Python 3.12 in the Colab/runtime path.
- FastAPI + Uvicorn.
- PyMuPDF for PDF native extraction and rendering.
- `python-docx` for DOCX extraction.
- RapidOCR 3.9.1+ with PyTorch CUDA for OCR.
- PP-OCRv6 medium through RapidOCR as the single OCR model path.
- DocLayout-YOLO for visual region detection.
- Sentence Transformers with `BAAI/bge-m3` for embeddings.
- Local JSON/NumPy vector store under `backend/data/rag/`.
- Ollama API for LLM generation.

Frontend:

- React.
- Vite.
- CSS in `frontend/src/styles.css`.

Runtime model/config defaults:

```env
OLLAMA_MODEL=hf.co/Zkare/Chatbot_Ielts_Assistant_v2:Q4_K_M
EMBEDDING_MODEL_NAME=BAAI/bge-m3
OCR_ENGINE=rapidocr
OCR_RUNTIME=torch
OCR_DEVICE=cuda:0
OCR_LANG=en
OCR_DET_LANG=ch
OCR_VERSION=PP-OCRv6
OCR_MODEL_SIZE=medium
LAYOUT_ENABLE=true
LAYOUT_ENGINE=doclayout_yolo
LAYOUT_DEVICE=cuda:0
LAYOUT_MODEL_REPO=juliozhao/DocLayout-YOLO-DocStructBench
LAYOUT_MODEL_FILENAME=doclayout_yolo_docstructbench_imgsz1024.pt
WARMUP_LLM=true
WARMUP_EMBEDDING=true
WARMUP_OCR=true
WARMUP_LAYOUT=true
```

Important current runtime decision:

- Do not reintroduce PaddleOCR/PaddlePaddle, Tesseract, PP-StructureV3, or
  multi-model OCR fallback unless the user explicitly asks and there is a clear
  measured reason.
- The streamlined OCR path is RapidOCR + PyTorch CUDA + PP-OCRv6 medium.
- DocLayout-YOLO detects visual regions. It does not OCR text and does not parse
  table cells by itself.

## 3. Project Structure

Important paths:

```text
backend/app/main.py
backend/app/rag.py
backend/app/structured_store.py
backend/app/intent.py
backend/app/llm.py
backend/app/schemas.py

backend/app/document_pipeline/
  config.py
  processor.py
  routing.py
  quality.py
  models.py
  normalization.py
  reconciliation.py
  ielts.py
  visual.py
  chunking.py
  ocr.py
  layout.py
  extractors/
    text.py
    pdf.py
    docx.py
    image.py

backend/tests/
  test_document_pipeline.py
  test_rag_store.py

backend/evaluation/
  chat_corpus_v2.json

backend/tools/
  chat_evaluation.py

frontend/src/App.jsx
frontend/src/styles.css

README.md
RAG_PIPELINE_REVIEW.md
PROJECT_HANDOFF.md
```

`PROJECT_HANDOFF.md` is local/project context and may be stale. Do not treat it
as the source of truth when it conflicts with current code or `README.md`.

The Colab notebook is outside this repo:

```text
/Users/phamvantu/Desktop/Elsa-Speaker/IELTS_Chatbot_Standalone_Colab.ipynb
```

Editing that notebook requires filesystem access outside the repo.

## 4. Coding Conventions

General:

- Make surgical, minimal changes.
- Do not refactor unrelated code.
- Do not add speculative abstractions.
- Match the existing style.
- Use type hints for new functions/classes.
- Keep comments short and only where they explain non-obvious logic.
- Use structured metadata/debug fields rather than ad hoc print debugging.
- Keep thresholds and model choices in config/environment variables.

Python:

- Prefer dataclasses and existing schema classes in `document_pipeline/models.py`.
- Avoid hard-coded file names, sample document titles, page numbers, question
  ranges, or topic-specific rules.
- If adding parsing patterns, make them generic and test them with minimal
  examples.
- Do not use LLM calls to compensate for missing document structure.

Frontend:

- Keep UI changes scoped to the requested debug or rendering behavior.
- Do not redesign the app unless requested.
- Preserve streaming behavior and debug download support.

File editing:

- Use patches for manual edits.
- Do not delete user changes.
- Check `git diff` before and after meaningful changes.

## 5. Build, Run, and Test Commands

Backend local setup:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 2222
```

Frontend local setup:

```bash
cd frontend
npm install
npm run dev
```

Warmup:

```bash
curl -s -X POST http://127.0.0.1:2222/warmup
```

Required checks after backend changes:

```bash
python3 -m compileall -q backend/app
python3 -m unittest discover -s backend/tests -v
```

Frontend check when touching frontend:

```bash
cd frontend && npm run build
```

Full Colab verification:

1. Pull latest repo into Colab.
2. Install `backend/requirements.txt`.
3. Start Ollama/backend/frontend.
4. Call `/warmup`.
5. Upload sample PDF/image.
6. Inspect upload timing/debug metadata.
7. Use `backend/tools/chat_evaluation.py` to collect answers and RAG debug data
   for the questions in `backend/evaluation/chat_corpus_v2.json`.

The legacy 19-question set is retired. Do not use it as the acceptance baseline;
it predates the current seven-document corpus.

## 6. Architecture Notes

Document ingestion pipeline:

```text
Upload file
-> POST /documents/upload
-> save temporary upload
-> DocumentProcessor.process_file(...)
-> FileRouter
-> extractor for text/pdf/docx/image
-> native text first
-> OCR only when needed
-> optional layout region detection
-> NativeOCRReconciler
-> IELTSStructureParser
-> StructuredChunker
-> LocalVectorStore.upsert(...)
-> documents.json + embeddings.npy
```

Chat/RAG pipeline:

```text
User query
-> allowed document catalog + compact semantic routing candidates
-> one structured LLM gateway decides route, intent, and target documents/sections
-> validate gateway references and explicit no-solve/no-writing constraints
-> structured lookup when possible
-> metadata-filtered retrieval
-> parent/context expansion when needed
-> RAG prompt
-> Ollama response or stream
-> frontend debug panel/download
```

Core principle:

```text
Structured lookup finds the document unit.
Semantic retrieval finds evidence.
Generation policy decides whether the LLM may solve.
```

Intent policy:

- `document_overview`: use outline/passages, not arbitrary top-k.
- `show_questions`: show/translate/explain question content without solving.
- `show_table` / `show_flowchart`: do not fill blanks.
- `solve_questions`: only when the user asks to answer/solve.
- `semantic_qa`: use context from the correct target document/section.
- Document-grounded questions must not silently fall back to general LLM answers
  when no valid source is found.

Storage:

- Persistent RAG data defaults to `backend/data/rag/documents.json` and
  `backend/data/rag/embeddings.npy`.
- Uploading the same source replaces that source only after embeddings succeed.
- Session/user isolation is not complete yet; treat the local store as shared by
  one backend process.

## 7. Rules for AI Agents

Before changing code:

1. Read the relevant files first.
2. Check the current git diff.
3. State assumptions when the task is ambiguous.
4. Prefer the smallest correct patch.

Do:

- Keep the pipeline general across IELTS-like documents.
- Preserve raw/native/OCR text and provenance where possible.
- Add tests for parser, retrieval, intent, and schema changes.
- Use timing/debug metadata to locate bottlenecks.
- Keep model configuration in env/config.
- Keep OCR/layout warmup explicit and observable.

Do not:

- Do not hard-code sample file names, sample passage titles, fixed page numbers,
  or fixed question ranges.
- Do not patch hallucinations only at the prompt layer when the root issue is
  extraction, structure, target resolution, or retrieval.
- Do not increase top-k/context size to hide bad chunking or bad routing.
- Do not add back Tesseract, PaddleOCR, PaddlePaddle, PP-StructureV3, or a
  fallback OCR cascade without explicit approval.
- Do not use DocLayout as a table-cell parser; it is region detection only.
- Do not call the LLM for deterministic rendering when structured data is enough.
- Do not remove raw extraction/debug metadata just to make outputs look clean.
- Do not clean the whole repo when asked to fix one pipeline issue.

Colab-specific:

- If notebook errors mention stale code, ensure Colab has pulled the latest repo.
- If DocLayout warmup fails, inspect `layout` detail from `/warmup`; do not guess.
- If upload is slow, inspect upload timing metadata before changing models.
- If Cloudflare URL fails, first verify backend health and the tunnel cell state.

Current known issue area:

- Recent Colab output showed `ocr.ok=true` and `layout.ok=false` because
  DocLayout loading tried to find `yolov10n.pt`. The repo has been patched to
  load the explicit Hugging Face checkpoint
  `doclayout_yolo_docstructbench_imgsz1024.pt`. After pushing/pulling that patch,
  rerun install, backend start, and `/warmup`.
- Upload failures seen through Cloudflare may be caused by backend warmup/layout
  failure, stale Colab code, or request timeout. Confirm with backend logs and
  upload timing debug before changing extraction logic.

## 8. Testing Strategy

Unit tests should cover:

- File routing.
- OCR result normalization and failure diagnostics.
- DocLayout output normalization.
- Native/OCR reconciliation.
- IELTS structure parser boundaries.
- Visual/Writing table parser.
- Structured chunk emission.
- Intent classification, especially negative constraints.
- Structured lookup for question ranges and table cells.
- Retrieval filtering and overview behavior.

Use these commands for normal verification:

```bash
python3 -m compileall -q backend/app
python3 -m unittest discover -s backend/tests -v
```

When touching frontend:

```bash
cd frontend && npm run build
```

When touching ingestion/RAG behavior, also verify with:

- Upload a native-text PDF.
- Upload an image with a Writing Task 1 table.
- Ask a document overview query.
- Ask a show-only question query.
- Ask a solve question query with evidence.
- Ask a negative document query that should return no-match instead of a general
  answer.

## 9. PR / Commit Guidelines

- Keep commits focused.
- Mention tests run.
- Mention any intentionally deferred issues.
- Do not include local runtime artifacts:
  - `backend/data/rag/`
  - `backend/uploads/`
  - notebook outputs
  - model checkpoints
  - cache directories
- `PROJECT_HANDOFF.md` is local-only context and should not be pushed unless the
  user explicitly changes that policy.

## 10. Current Direction

Near-term direction:

1. Keep the current extraction baseline frozen unless a failure is reproduced
   across multiple documents or blocks production.
2. Rebuild the structured and vector indexes from the current schema.
3. Evaluate the semantic route/intent gateway and target resolution across the multi-document corpus.
4. Run `backend/tools/chat_evaluation.py` against the rebuilt seven-document corpus
   and review its raw answer/debug report manually. Do not auto-score answers.
5. Fix retrieval, grounding and generation-policy failures before changing prompts.

Longer-term direction:

- Strengthen target document resolution for multiple uploaded files.
- Add better session isolation.
- Keep deterministic renderers for show/extract flows.
- Keep semantic retrieval for evidence and explanation, not for exact document
  addressing.
