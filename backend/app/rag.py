import json
import os
import re
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

        return self._dense_search(query, top_k=top_k, min_score=self.min_score)

    def probe(self, query: str, top_k: int = 3) -> Dict:
        if not self._docs:
            return {"results": [], "has_hits": False, "has_strong_hits": False}

        probe_min_dense = float(os.getenv("RAG_PROBE_MIN_DENSE_SCORE", "0.35"))
        dense_results = self._dense_search(query, top_k=top_k, min_score=probe_min_dense) if self._embeddings.size else []
        keyword_results = self._keyword_search(query, top_k=top_k)
        question_results = self._question_range_search(query, top_k=top_k)

        merged: dict[str, Dict] = {}
        for result in dense_results:
            key = result.get("chunk_id") or f"dense-{result.get('chunk_index')}"
            merged[key] = dict(result)
            merged[key]["probe_dense_score"] = float(result.get("score", 0.0))
            merged[key]["probe_keyword_score"] = 0.0
            merged[key]["probe_question_score"] = 0.0

        for result in keyword_results:
            key = result.get("chunk_id") or f"keyword-{result.get('chunk_index')}"
            if key not in merged:
                merged[key] = dict(result)
                merged[key]["score"] = 0.0
                merged[key]["probe_dense_score"] = 0.0
                merged[key]["probe_question_score"] = 0.0
            merged[key]["probe_keyword_score"] = float(result.get("keyword_score", 0.0))

        for result in question_results:
            key = result.get("chunk_id") or f"question-{result.get('chunk_index')}"
            if key not in merged:
                merged[key] = dict(result)
                merged[key]["score"] = 0.0
                merged[key]["probe_dense_score"] = 0.0
                merged[key]["probe_keyword_score"] = 0.0
            merged[key]["probe_question_score"] = float(result.get("question_score", 0.0))

        results = sorted(
            merged.values(),
            key=lambda item: (
                item.get("probe_question_score", 0.0),
                item.get("probe_keyword_score", 0.0),
                item.get("probe_dense_score", 0.0),
            ),
            reverse=True,
        )[:top_k]

        return {
            "results": results,
            "has_hits": bool(results),
            "has_strong_hits": bool(
                results
                and (
                    results[0].get("probe_keyword_score", 0.0) >= 2
                    or results[0].get("probe_question_score", 0.0) >= 1
                    or results[0].get("probe_dense_score", 0.0) >= self.min_score
                )
            ),
            "top_score": results[0].get("score", 0.0) if results else 0.0,
            "top_keyword_score": results[0].get("probe_keyword_score", 0.0) if results else 0.0,
            "top_question_score": results[0].get("probe_question_score", 0.0) if results else 0.0,
        }

    def document_catalog(self) -> List[Dict]:
        catalog: dict[str, Dict] = {}
        for doc in self._docs:
            source_file = doc.get("source_file", "unknown")
            entry = catalog.setdefault(
                source_file,
                {
                    "source_file": source_file,
                    "chunks": 0,
                    "pages": set(),
                    "document_ids": set(),
                    "mime_types": set(),
                },
            )
            entry["chunks"] += 1
            entry["pages"].update(doc.get("pages") or [])
            if doc.get("document_id"):
                entry["document_ids"].add(doc["document_id"])
            mime_type = doc.get("metadata", {}).get("mime_type")
            if mime_type:
                entry["mime_types"].add(mime_type)

        return [
            {
                "source_file": item["source_file"],
                "chunks": item["chunks"],
                "pages": sorted(item["pages"]),
                "document_ids": sorted(item["document_ids"]),
                "mime_types": sorted(item["mime_types"]),
            }
            for item in catalog.values()
        ]

    def _dense_search(self, query: str, top_k: int, min_score: float) -> List[Dict]:
        query_embedding = self._embed([query])[0]
        scores = self._embeddings @ query_embedding
        order = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in order:
            score = float(scores[idx])
            if score < min_score:
                continue
            doc = dict(self._docs[int(idx)])
            doc["score"] = score
            results.append(doc)
        return results

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        query_terms = self._terms(query)
        if not query_terms:
            return []

        results = []
        for doc in self._docs:
            text = doc.get("text", "")
            text_terms = set(self._terms(text))
            overlap = sum(1 for term in query_terms if term in text_terms)
            phrase_bonus = 2 if query.lower() in text.lower() else 0
            score = overlap + phrase_bonus
            if score <= 0:
                continue
            item = dict(doc)
            item["keyword_score"] = float(score)
            results.append(item)

        return sorted(results, key=lambda item: item["keyword_score"], reverse=True)[:top_k]

    def _question_range_search(self, query: str, top_k: int) -> List[Dict]:
        ranges = self._question_ranges(query)
        if not ranges:
            return []

        results = []
        for doc in self._docs:
            text = doc.get("text", "")
            score = 0.0
            for start, end in ranges:
                score += self._question_match_score(text, start, end)
            if score <= 0:
                continue
            item = dict(doc)
            item["question_score"] = score
            results.append(item)

        return sorted(results, key=lambda item: item["question_score"], reverse=True)[:top_k]

    def _question_ranges(self, query: str) -> List[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        normalized = query.lower().replace("đến", "to").replace("tới", "to")
        for match in re.finditer(r"(?:question|questions|câu)\s*(\d{1,2})(?:\s*(?:-|–|to)\s*(\d{1,2}))?", normalized):
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if start > end:
                start, end = end, start
            ranges.append((start, end))
        return ranges

    def _question_match_score(self, text: str, start: int, end: int) -> float:
        score = 0.0
        lowered = text.lower()
        for header_start, header_end in self._question_headers(lowered):
            if self._ranges_overlap(start, end, header_start, header_end):
                score += 8.0

        for number in range(start, end + 1):
            if re.search(rf"(?<!\d){number}\s*[\.)]", lowered) or re.search(rf"\({number}\)", lowered):
                score += 1.0
        return score

    def _question_headers(self, text: str) -> List[tuple[int, int]]:
        headers = []
        for match in re.finditer(r"questions?\s+(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})", text):
            headers.append((int(match.group(1)), int(match.group(2))))
        return headers

    def _ranges_overlap(self, a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return max(a_start, b_start) <= min(a_end, b_end)

    def _terms(self, text: str) -> List[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "các",
            "câu",
            "cho",
            "của",
            "from",
            "gì",
            "is",
            "là",
            "nội",
            "of",
            "the",
            "to",
            "trong",
            "từ",
            "về",
            "what",
            "what's",
        }
        terms = re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)
        return [term for term in terms if term not in stopwords and (len(term) > 1 or term.isdigit())]

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
