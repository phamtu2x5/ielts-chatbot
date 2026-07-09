# IELTS Chatbot Standalone

Standalone chatbot repo extracted from the IELTS learning system.

It includes:

- FastAPI backend
- React/Vite frontend
- Ollama LLM integration
- Optional PDF RAG using an embedded local vector store

## Architecture

```text
Browser
-> React frontend
-> FastAPI backend
-> Ollama
-> Zkare IELTS chatbot model
```

For PDF RAG:

```text
Upload PDF
-> extract text
-> chunk
-> sentence-transformers embedding
-> local vector store
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
ENABLE_VECTOR_RAG=true
EMBEDDING_MODEL_NAME=BAAI/bge-m3
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

Those can be reconnected later if needed.
