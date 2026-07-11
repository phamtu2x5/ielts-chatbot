import json
import re
import threading
from functools import wraps
from typing import Callable, Dict, List, TypeVar

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings

DATA_DIR = settings.rag_data_dir
INDEX_PATH = DATA_DIR / "embeddings.npy"
DOCS_PATH = DATA_DIR / "documents.json"

T = TypeVar("T")


def synchronized(method: Callable[..., T]) -> Callable[..., T]:
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class LocalVectorStore:
    def __init__(self) -> None:
        self.model_name = settings.embedding_model_name
        self.min_score = settings.rag_min_score
        self._lock = threading.RLock()
        self._embedding_lock = threading.Lock()
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
        docs_exist = DOCS_PATH.exists()
        index_exists = INDEX_PATH.exists()
        if docs_exist != index_exists:
            raise RuntimeError(
                "RAG index is incomplete: documents.json and embeddings.npy must exist together. "
                "Remove the incomplete index and upload the documents again."
            )
        if not docs_exist:
            return

        try:
            docs = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
            embeddings = np.load(INDEX_PATH, allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("RAG index cannot be loaded. Rebuild it by uploading the documents again.") from exc

        if not isinstance(docs, list) or embeddings.ndim != 2 or len(docs) != embeddings.shape[0]:
            raise RuntimeError(
                "RAG index is inconsistent: document and embedding counts do not match. "
                "Rebuild it by uploading the documents again."
            )
        self._docs = docs
        self._embeddings = np.asarray(embeddings, dtype=np.float32)

    def _save_state(self, docs: List[Dict], embeddings: np.ndarray) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        docs_temp = DOCS_PATH.with_suffix(".json.tmp")
        index_temp = INDEX_PATH.with_suffix(".npy.tmp")
        try:
            docs_temp.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
            with index_temp.open("wb") as handle:
                np.save(handle, embeddings)
            index_temp.replace(INDEX_PATH)
            docs_temp.replace(DOCS_PATH)
        finally:
            docs_temp.unlink(missing_ok=True)
            index_temp.unlink(missing_ok=True)

    def _embed(self, texts: List[str]) -> np.ndarray:
        with self._embedding_lock:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)

    @synchronized
    def warmup(self) -> Dict:
        embedding = self._embed(["IELTS document retrieval warmup"])[0]
        return {"embedding_model": self.model_name, "embedding_dimensions": int(embedding.shape[0])}

    def upsert(self, chunks: List[Dict], source_file: str) -> int:
        if not chunks:
            return 0
        if any(not (chunk.get("retrieval_text") or chunk.get("text")) for chunk in chunks):
            raise ValueError("Every RAG chunk must contain text or retrieval_text.")

        texts = [chunk.get("retrieval_text") or chunk["text"] for chunk in chunks]
        new_embeddings = self._embed(texts)
        with self._lock:
            keep_indices = [idx for idx, doc in enumerate(self._docs) if doc.get("source_file") != source_file]
            kept_docs = [self._docs[idx] for idx in keep_indices]
            kept_embeddings = (
                self._embeddings[keep_indices]
                if self._embeddings.size
                else np.empty((0, 0), dtype=np.float32)
            )

            if kept_embeddings.size == 0:
                combined_embeddings = new_embeddings
            else:
                if kept_embeddings.shape[1] != new_embeddings.shape[1]:
                    raise RuntimeError(
                        "Embedding dimensions changed. Clear the RAG index before switching embedding models."
                    )
                combined_embeddings = np.vstack([kept_embeddings, new_embeddings])

            combined_docs = kept_docs + chunks
            self._save_state(combined_docs, combined_embeddings)
            self._docs = combined_docs
            self._embeddings = combined_embeddings
        return len(chunks)

    @synchronized
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self._docs or self._embeddings.size == 0:
            return []

        return self._dense_search(query, top_k=max(1, min(top_k, 50)), min_score=self.min_score)

    @synchronized
    def probe(self, query: str, top_k: int = 3) -> Dict:
        if not self._docs:
            return {"results": [], "has_hits": False, "has_strong_hits": False}

        is_overview = self._is_document_overview_query(query)
        has_document_intent = is_overview or self._has_document_intent(query)
        if is_overview:
            results = self.overview(top_k=settings.rag_overview_top_k)
            for result in results:
                result["probe_dense_score"] = float(result.get("score", 0.0))
                result["probe_keyword_score"] = 0.0
                result["probe_question_score"] = 0.0
                result["probe_overview_score"] = 1.0
            return {
                "results": results,
                "has_hits": bool(results),
                "has_strong_hits": bool(results),
                "has_document_intent": True,
                "is_overview": True,
                "top_score": results[0].get("score", 0.0) if results else 0.0,
                "top_keyword_score": 0.0,
                "top_question_score": 0.0,
                "top_overview_score": 1.0 if results else 0.0,
            }

        top_k = max(1, min(top_k, 50))
        probe_min_dense = settings.rag_probe_min_dense_score
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
            merged[key]["probe_overview_score"] = 0.0

        for result in keyword_results:
            key = result.get("chunk_id") or f"keyword-{result.get('chunk_index')}"
            if key not in merged:
                merged[key] = dict(result)
                merged[key]["score"] = 0.0
                merged[key]["probe_dense_score"] = 0.0
                merged[key]["probe_question_score"] = 0.0
                merged[key]["probe_overview_score"] = 0.0
            merged[key]["probe_keyword_score"] = float(result.get("keyword_score", 0.0))

        for result in question_results:
            key = result.get("chunk_id") or f"question-{result.get('chunk_index')}"
            if key not in merged:
                merged[key] = dict(result)
                merged[key]["score"] = 0.0
                merged[key]["probe_dense_score"] = 0.0
                merged[key]["probe_keyword_score"] = 0.0
                merged[key]["probe_overview_score"] = 0.0
            merged[key]["probe_question_score"] = float(result.get("question_score", 0.0))

        results = sorted(
            merged.values(),
            key=lambda item: (
                item.get("probe_overview_score", 0.0),
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
                    or results[0].get("probe_overview_score", 0.0) >= 1
                    or results[0].get("probe_dense_score", 0.0) >= self.min_score
                )
            ),
            "has_document_intent": has_document_intent,
            "is_overview": False,
            "top_score": results[0].get("score", 0.0) if results else 0.0,
            "top_keyword_score": results[0].get("probe_keyword_score", 0.0) if results else 0.0,
            "top_question_score": results[0].get("probe_question_score", 0.0) if results else 0.0,
            "top_overview_score": 0.0,
        }

    @synchronized
    def probe_with_catalog(self, query: str, top_k: int = 3) -> tuple[Dict, List[Dict]]:
        return self.probe(query, top_k), self.document_catalog()

    @synchronized
    def overview(self, top_k: int = 8) -> List[Dict]:
        top_k = max(1, min(top_k, 50))
        outline_docs = [
            doc
            for doc in self._docs
            if doc.get("metadata", {}).get("unit_type") in {"document_outline", "passage_summary"}
        ]
        if outline_docs:
            results = []
            for doc in sorted(outline_docs, key=lambda item: item.get("chunk_index", 0))[:top_k]:
                item = dict(doc)
                item["score"] = 0.0
                item["overview_score"] = 1.0
                results.append(item)
            passage_docs = [
                doc
                for doc in sorted(self._docs, key=lambda item: item.get("chunk_index", 0))
                if doc.get("metadata", {}).get("unit_type") == "passage"
            ]
            seen_passages = set()
            for doc in passage_docs:
                passage_number = doc.get("metadata", {}).get("passage_number")
                if passage_number in seen_passages:
                    continue
                seen_passages.add(passage_number)
                item = dict(doc)
                item["score"] = 0.0
                item["overview_score"] = 0.8
                results.append(item)
                if len(results) >= top_k:
                    break
            return results

        docs = sorted(self._docs, key=lambda doc: (min(doc.get("pages") or [999]), doc.get("chunk_index", 0)))
        content_docs = [doc for doc in docs if doc.get("token_count", 0) >= 80]
        selected = content_docs[:top_k] if content_docs else docs[:top_k]
        results = []
        for doc in selected:
            item = dict(doc)
            item["score"] = 0.0
            item["overview_score"] = 1.0
            results.append(item)
        return results

    @synchronized
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
                    "unit_types": set(),
                    "passage_numbers": set(),
                },
            )
            entry["chunks"] += 1
            entry["pages"].update(doc.get("pages") or [])
            if doc.get("document_id"):
                entry["document_ids"].add(doc["document_id"])
            mime_type = doc.get("metadata", {}).get("mime_type")
            if mime_type:
                entry["mime_types"].add(mime_type)
            unit_type = doc.get("metadata", {}).get("unit_type")
            if unit_type:
                entry["unit_types"].add(unit_type)
            passage_number = doc.get("metadata", {}).get("passage_number")
            if passage_number:
                entry["passage_numbers"].add(passage_number)

        return [
            {
                "source_file": item["source_file"],
                "chunks": item["chunks"],
                "pages": sorted(item["pages"]),
                "document_ids": sorted(item["document_ids"]),
                "mime_types": sorted(item["mime_types"]),
                "unit_types": sorted(item["unit_types"]),
                "passage_numbers": sorted(item["passage_numbers"]),
            }
            for item in catalog.values()
        ]

    @synchronized
    def passage_context_for_sources(self, sources: List[Dict], max_chunks_per_passage: int = 3) -> List[Dict]:
        passage_numbers = {
            source.get("metadata", {}).get("passage_number")
            for source in sources
            if source.get("metadata", {}).get("passage_number")
        }
        if not passage_numbers:
            return []

        results: list[Dict] = []
        counts: dict[int, int] = {}
        for doc in sorted(self._docs, key=lambda item: item.get("chunk_index", 0)):
            metadata = doc.get("metadata", {})
            passage_number = metadata.get("passage_number")
            if metadata.get("unit_type") != "passage" or passage_number not in passage_numbers:
                continue
            count = counts.get(passage_number, 0)
            if count >= max_chunks_per_passage:
                continue
            item = dict(doc)
            item["score"] = 0.0
            item["context_expansion_score"] = 1.0
            results.append(item)
            counts[passage_number] = count + 1
        return results

    def _dense_search(self, query: str, top_k: int, min_score: float) -> List[Dict]:
        query_embedding = self._embed([query])[0]
        if self._embeddings.shape[1] != query_embedding.shape[0]:
            raise RuntimeError(
                "Stored embeddings are incompatible with the configured embedding model. Rebuild the RAG index."
            )
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
            text = doc.get("retrieval_text") or doc.get("text", "")
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

        header_results = []
        numeric_results = []
        for doc in self._docs:
            metadata_score = self._question_metadata_match_score(doc, ranges)
            text = doc.get("text", "")
            header_score = 0.0
            numeric_score = 0.0
            for start, end in ranges:
                header_score += self._question_header_match_score(text, start, end)
                numeric_score += self._question_number_match_score(text, start, end)
            if metadata_score > 0:
                item = dict(doc)
                item["question_score"] = metadata_score + header_score + numeric_score
                header_results.append(item)
            elif header_score > 0:
                item = dict(doc)
                item["question_score"] = header_score + numeric_score
                header_results.append(item)
            elif numeric_score > 0:
                item = dict(doc)
                item["question_score"] = numeric_score
                numeric_results.append(item)

        results = header_results if header_results else numeric_results
        return sorted(results, key=lambda item: item["question_score"], reverse=True)[:top_k]

    def _question_metadata_match_score(self, doc: Dict, ranges: List[tuple[int, int]]) -> float:
        metadata = doc.get("metadata", {})
        question_range = metadata.get("question_range")
        if not isinstance(question_range, list) or len(question_range) != 2:
            return 0.0
        chunk_start, chunk_end = int(question_range[0]), int(question_range[1])
        score = 0.0
        unit_type = metadata.get("unit_type")
        for start, end in ranges:
            if not self._ranges_overlap(start, end, chunk_start, chunk_end):
                continue
            overlap = min(end, chunk_end) - max(start, chunk_start) + 1
            exact_bonus = 20.0 if start == chunk_start and end == chunk_end else 0.0
            unit_bonus = 8.0 if unit_type == "question_group" else 4.0 if unit_type == "question" else 0.0
            score += 30.0 + overlap + exact_bonus + unit_bonus
        return score

    def _question_ranges(self, query: str) -> List[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        normalized = query.lower().replace("đến", "to").replace("tới", "to")
        for match in re.finditer(r"(?:questions?|câu hỏi|câu)\s*(\d{1,2})(?:\s*(?:-|–|to)\s*(\d{1,2}))?", normalized):
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if start > end:
                start, end = end, start
            ranges.append((start, end))
        return ranges

    def _question_header_match_score(self, text: str, start: int, end: int) -> float:
        score = 0.0
        lowered = text.lower()
        for header_start, header_end in self._question_headers(lowered):
            if self._ranges_overlap(start, end, header_start, header_end):
                exact_bonus = 4.0 if start == header_start and end == header_end else 0.0
                score += 20.0 + exact_bonus
        return score

    def _question_number_match_score(self, text: str, start: int, end: int) -> float:
        score = 0.0
        lowered = text.lower()
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

    def _has_document_intent(self, query: str) -> bool:
        lowered = query.lower()
        markers = [
            "tài liệu",
            "file",
            "pdf",
            "docx",
            "trong bài",
            "bài đọc",
            "passage",
            "question",
            "questions",
            "câu hỏi",
            "trang",
            "page",
            "bảng",
            "table",
            "flow",
            "nội dung",
        ]
        return any(marker in lowered for marker in markers)

    def _is_document_overview_query(self, query: str) -> bool:
        lowered = query.lower()
        overview_markers = [
            "nội dung tài liệu",
            "nội dung của tài liệu",
            "nội dung tài liệu trên",
            "nội dung file",
            "nội dung của file",
            "tài liệu là gì",
            "tài liệu trên là gì",
            "tài liệu này là gì",
            "tài liệu này nói gì",
            "tài liệu trên nói gì",
            "file này là gì",
            "file trên là gì",
            "pdf này là gì",
            "pdf trên là gì",
            "tóm tắt tài liệu",
            "tổng quan tài liệu",
            "summary of the document",
            "summarize the document",
        ]
        if any(marker in lowered for marker in overview_markers):
            return True
        has_content_word = "nội dung" in lowered or "tóm tắt" in lowered or "tổng quan" in lowered
        has_document_word = any(marker in lowered for marker in ["tài liệu", "file", "pdf", "document"])
        has_reference_word = any(marker in lowered for marker in ["này", "trên", "đó", "uploaded", "đã tải"])
        return has_content_word and has_document_word and has_reference_word

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

    @synchronized
    def delete_source(self, source_file: str) -> int:
        if not self._docs:
            return 0

        keep_indices = [idx for idx, doc in enumerate(self._docs) if doc.get("source_file") != source_file]
        removed = len(self._docs) - len(keep_indices)
        if removed == 0:
            return 0

        docs = [self._docs[idx] for idx in keep_indices]
        embeddings = self._embeddings[keep_indices] if self._embeddings.size else self._embeddings
        self._save_state(docs, embeddings)
        self._docs = docs
        self._embeddings = embeddings
        return removed

    @synchronized
    def stats(self) -> Dict:
        return {
            "documents": len({doc.get("source_file") for doc in self._docs}),
            "chunks": len(self._docs),
            "embedding_model": self.model_name,
        }


_store: LocalVectorStore | None = None
_store_lock = threading.Lock()


def get_store() -> LocalVectorStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = LocalVectorStore()
    return _store
