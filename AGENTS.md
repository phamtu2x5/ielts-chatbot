# AGENTS.md

This branch is the production direct-chat backend for the IELTS assistant.

## Scope

Keep only:

- FastAPI direct chat and NDJSON streaming.
- Ollama generation and direct-output validation.
- API authentication, CORS, rate limiting and concurrency limiting.
- Backend-owned, per-session conversation memory and cleanup.
- Ubuntu `systemd` and Cloudflare Tunnel deployment assets.

Do not add document upload, OCR, layout models, embeddings, vector storage, retrieval, or RAG to
this branch. Those features remain on `main` and require an explicit product decision to restore.

## Commands

```bash
python3 -m compileall -q backend/app
PYTHONPATH=backend python3 -m unittest discover -s backend/tests -v
```

Run Uvicorn with exactly one worker:

```bash
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8765 --workers 1
```

## Coding rules

- Prefer the smallest correct change.
- Do not weaken session isolation or deletion safety.
- Do not accept conversation history from the client as authoritative; backend session memory wins.
- Keep secrets in `backend/.env`, never in Git.
- Preserve the NDJSON event contract expected by the external frontend.
- Add focused tests for API, memory, cancellation and session cleanup changes.
