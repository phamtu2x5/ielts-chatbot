# IELTS Direct Chatbot Backend

Production backend for the direct-chat IELTS assistant. This branch deliberately contains no
document upload, OCR, layout detection, embeddings, vector retrieval, or RAG endpoints.

The React frontend is maintained separately at
<https://github.com/phamtu2x5/ielts-chatbot-fe>.

## Runtime architecture

```text
IELTS website
  -> trusted server-side proxy adds Authorization: Bearer <token>
  -> https://api.mywsite.online
  -> Cloudflare Tunnel
  -> FastAPI on 127.0.0.1:8765
  -> Ollama on 127.0.0.1:11434
  -> IELTS Q4 model on the RTX 3060
```

FastAPI owns conversation history and user facts per `session_id`. Session data is isolated in
`backend/data/sessions/<session-id>/memory.json`, expires after the configured TTL, and is deleted
by the session cleanup endpoints or background cleanup loop.

Exactly one FastAPI worker is required because session storage and request locks are local to one
process.

## API

- `GET /health`
- `GET /admin/stats` (Bearer token)
- `POST /warmup` (Bearer token)
- `POST /chat/stream` (Bearer token, NDJSON streaming)
- `POST /sessions/{session_id}/expire` (Bearer token)
- `DELETE /sessions/{session_id}` (Bearer token)

The chat request body is:

```json
{
  "session_id": "a-valid-uuid",
  "message": "Lập cho tôi lộ trình IELTS 6 tuần"
}
```

The stream emits `status`, `metadata`, `token`, `done`, or `error` NDJSON events.

## Local development

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8765 --workers 1
```

Ollama must already be running with the model configured in `.env`.

Verify:

```bash
curl -s http://127.0.0.1:8765/health
python3 -m compileall -q backend/app
PYTHONPATH=backend python3 -m unittest discover -s backend/tests -v
```

## Ubuntu production

Follow [SERVER_DEPLOYMENT.md](SERVER_DEPLOYMENT.md). The server installs Ollama, the model, the
repository and Python packages once. Normal restarts do not download them again.

Never commit `backend/.env`, the API token, the Cloudflare tunnel token, session data, model files,
virtual environments, caches, or logs.
