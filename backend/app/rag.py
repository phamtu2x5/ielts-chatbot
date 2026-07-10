import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "data/rag"))
INDEX_PATH = DATA_DIR / "embeddings.npy"
DOCS_PATH = DATA_DIR / "documents.json"


class LocalVectorStore:
    def __init__(self) -> None:
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        self.min_score = float(os.getenv("RAG_MIN_SCORE", "0.45"))
        self._model = None
        self._docs: List[Dict] = []
        self._embeddings = np.empty((0, 0), dtype=np.float32)
        self._load()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _load(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if DOCS_PATH.exists():
            self._docs = json.loads(DOCS_PATH.read_text())
        if INDEX_PATH.exists():
            self._embeddings = np.load(INDEX_PATH)

    def _save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DOCS_PATH.write_text(json.dumps(self._docs, ensure_ascii=False, indent=2))
        np.save(INDEX_PATH, self._embeddings)

    def _embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)

    def warmup(self) -> Dict:
        embedding = self._embed(["IELTS document retrieval warmup"])[0]
        return {"embedding_model": self.model_name, "embedding_dimensions": int(embedding.shape[0])}

    def upsert(self, chunks: List[Dict], source_file: str) -> int:
        self.delete_source(source_file)
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        new_embeddings = self._embed(texts)

        if self._embeddings.size == 0:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        self._docs.extend(chunks)
        self._save()
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self._docs or self._embeddings.size == 0:
            return []

        query_embedding = self._embed([query])[0]
        scores = self._embeddings @ query_embedding
        order = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in order:
            score = float(scores[idx])
            if score < self.min_score:
                continue
            doc = dict(self._docs[int(idx)])
            doc["score"] = score
            results.append(doc)
        return results

    def delete_source(self, source_file: str) -> int:
        if not self._docs:
            return 0

        keep_indices = [idx for idx, doc in enumerate(self._docs) if doc.get("source_file") != source_file]
        removed = len(self._docs) - len(keep_indices)
        if removed == 0:
            return 0

        self._docs = [self._docs[idx] for idx in keep_indices]
        if self._embeddings.size:
            self._embeddings = self._embeddings[keep_indices]
        self._save()
        return removed

    def stats(self) -> Dict:
        return {
            "documents": len({doc.get("source_file") for doc in self._docs}),
            "chunks": len(self._docs),
            "embedding_model": self.model_name,
        }


_store = None


def get_store() -> LocalVectorStore:
    global _store
    if _store is None:
        _store = LocalVectorStore()
    return _store
