import json
import re
import threading
import time
from functools import wraps
from typing import Callable, Dict, List, TypeVar

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings
from .structured_store import StructuredDocumentStore

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
        self.last_upsert_timing: Dict = {}
        self.structured_store = StructuredDocumentStore(lambda: self._docs)
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

    def upsert(
        self,
        chunks: List[Dict],
        source_file: str,
    ) -> int:
        started = time.perf_counter()
        if not chunks:
            self.last_upsert_timing = {"chunks": 0, "total_seconds": 0.0}
            return 0
        if any(not (chunk.get("retrieval_text") or chunk.get("text")) for chunk in chunks):
            raise ValueError("Every RAG chunk must contain text or retrieval_text.")

        texts = [chunk.get("retrieval_text") or chunk["text"] for chunk in chunks]
        embedding_started = time.perf_counter()
        new_embeddings = self._embed(texts)
        embedding_seconds = self._elapsed(embedding_started)
        with self._lock:
            merge_started = time.perf_counter()
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
            merge_seconds = self._elapsed(merge_started)
            save_started = time.perf_counter()
            self._save_state(combined_docs, combined_embeddings)
            save_seconds = self._elapsed(save_started)
            self._docs = combined_docs
            self._embeddings = combined_embeddings
            self.last_upsert_timing = {
                "embedding_seconds": embedding_seconds,
                "merge_seconds": merge_seconds,
                "save_seconds": save_seconds,
                "chunks": len(chunks),
                "total_seconds": self._elapsed(started),
            }
        return len(chunks)

    @synchronized
    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: List[str] | None = None,
    ) -> List[Dict]:
        if not self._docs or self._embeddings.size == 0:
            return []

        return self._dense_search(
            query,
            top_k=max(1, min(top_k, 50)),
            min_score=self.min_score,
            document_ids=document_ids,
        )

    @synchronized
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: List[str] | None = None,
        unit_types: List[str] | None = None,
        passage_numbers: List[int] | None = None,
    ) -> List[Dict]:
        if not self._docs:
            return []

        top_k = max(1, min(top_k, 50))
        candidate_k = min(max(top_k * 2, top_k), 50)
        dense_results = (
            self._dense_search(
                query,
                top_k=candidate_k,
                min_score=settings.rag_probe_min_dense_score,
                document_ids=document_ids,
                unit_types=unit_types,
                passage_numbers=passage_numbers,
            )
            if self._embeddings.size
            else []
        )
        keyword_results = self._keyword_search(
            query,
            top_k=candidate_k,
            document_ids=document_ids,
            unit_types=unit_types,
            passage_numbers=passage_numbers,
        )
        return self._fuse_ranked_results(
            dense_results=dense_results,
            keyword_results=keyword_results,
            question_results=[],
            top_k=top_k,
        )

    @synchronized
    def probe(
        self,
        query: str,
        top_k: int = 3,
        document_ids: List[str] | None = None,
    ) -> Dict:
        if not self._docs:
            return {"results": [], "has_hits": False, "has_strong_hits": False}

        is_overview = self._is_document_overview_query(query)
        has_document_intent = is_overview or self._has_document_intent(query)
        if is_overview:
            results = self.overview(
                top_k=settings.rag_overview_top_k,
                document_ids=document_ids,
            )
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
        dense_results = (
            self._dense_search(
                query,
                top_k=top_k,
                min_score=probe_min_dense,
                document_ids=document_ids,
            )
            if self._embeddings.size
            else []
        )
        keyword_results = self._keyword_search(query, top_k=top_k, document_ids=document_ids)
        question_results = self._question_range_search(query, top_k=top_k, document_ids=document_ids)

        results = self._fuse_ranked_results(
            dense_results=dense_results,
            keyword_results=keyword_results,
            question_results=question_results,
            top_k=top_k,
        )

        return {
            "results": results,
            "has_hits": bool(results),
            "has_strong_hits": any(
                result.get("probe_keyword_score", 0.0) >= 2
                or result.get("probe_question_score", 0.0) >= 1
                or result.get("probe_overview_score", 0.0) >= 1
                or result.get("probe_dense_score", 0.0) >= self.min_score
                for result in results
            ),
            "has_document_intent": has_document_intent,
            "is_overview": False,
            "top_score": results[0].get("score", 0.0) if results else 0.0,
            "top_fused_score": results[0].get("rrf_score", 0.0) if results else 0.0,
            "top_keyword_score": results[0].get("probe_keyword_score", 0.0) if results else 0.0,
            "top_question_score": results[0].get("probe_question_score", 0.0) if results else 0.0,
            "top_overview_score": 0.0,
        }

    @synchronized
    def probe_with_catalog(
        self,
        query: str,
        top_k: int = 3,
        document_ids: List[str] | None = None,
    ) -> tuple[Dict, List[Dict]]:
        return self.probe(query, top_k, document_ids), self.document_catalog(document_ids)

    @synchronized
    def overview(self, top_k: int = 8, document_ids: List[str] | None = None) -> List[Dict]:
        return self.structured_store.overview(top_k=top_k, document_ids=document_ids)

    @synchronized
    def structured_lookup(
        self,
        query: str,
        intent: str,
        top_k: int = 8,
        document_ids: List[str] | None = None,
    ) -> List[Dict]:
        return self.structured_store.structured_lookup(
            query=query,
            intent=intent,
            top_k=top_k,
            document_ids=document_ids,
        )

    @synchronized
    def document_catalog(self, document_ids: List[str] | None = None) -> List[Dict]:
        return self.structured_store.document_catalog(document_ids=document_ids)

    @synchronized
    def question_context_for_sources(
        self,
        sources: List[Dict],
        top_k: int = 8,
        document_ids: List[str] | None = None,
    ) -> List[Dict]:
        return self.structured_store.question_context_for_sources(
            sources=sources,
            top_k=top_k,
            document_ids=document_ids,
        )

    @synchronized
    def passage_context_for_sources(
        self,
        sources: List[Dict],
        max_chunks_per_passage: int = 3,
        document_ids: List[str] | None = None,
    ) -> List[Dict]:
        return self.structured_store.passage_context_for_sources(
            sources=sources,
            max_chunks_per_passage=max_chunks_per_passage,
            document_ids=document_ids,
        )

    @synchronized
    def writing_context_for_sources(
        self,
        sources: List[Dict],
        top_k: int = 4,
        document_ids: List[str] | None = None,
    ) -> List[Dict]:
        return self.structured_store.writing_context_for_sources(
            sources=sources,
            top_k=top_k,
            document_ids=document_ids,
        )

    def _dense_search(
        self,
        query: str,
        top_k: int,
        min_score: float,
        document_ids: List[str] | None = None,
        unit_types: List[str] | None = None,
        passage_numbers: List[int] | None = None,
    ) -> List[Dict]:
        query_embedding = self._embed([query])[0]
        if self._embeddings.shape[1] != query_embedding.shape[0]:
            raise RuntimeError(
                "Stored embeddings are incompatible with the configured embedding model. Rebuild the RAG index."
            )
        candidate_indices = self._candidate_indices(
            document_ids=document_ids,
            unit_types=unit_types,
            passage_numbers=passage_numbers,
        )
        if not candidate_indices:
            return []
        scores = self._embeddings @ query_embedding
        order = sorted(candidate_indices, key=lambda index: scores[index], reverse=True)[:top_k]

        results = []
        for idx in order:
            score = float(scores[idx])
            if score < min_score:
                continue
            doc = dict(self._docs[int(idx)])
            doc["score"] = score
            results.append(doc)
        return results

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        document_ids: List[str] | None = None,
        unit_types: List[str] | None = None,
        passage_numbers: List[int] | None = None,
    ) -> List[Dict]:
        query_terms = self._terms(query)
        if not query_terms:
            return []

        results = []
        candidate_indices = self._candidate_indices(
            document_ids=document_ids,
            unit_types=unit_types,
            passage_numbers=passage_numbers,
        )
        for index in candidate_indices:
            doc = self._docs[index]
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

    def _candidate_indices(
        self,
        document_ids: List[str] | None = None,
        unit_types: List[str] | None = None,
        passage_numbers: List[int] | None = None,
    ) -> List[int]:
        allowed_documents = set(document_ids or [])
        allowed_units = set(unit_types or [])
        allowed_passages = set(passage_numbers or [])
        return [
            index
            for index, doc in enumerate(self._docs)
            if (not allowed_documents or doc.get("document_id") in allowed_documents)
            and (
                not allowed_units
                or doc.get("metadata", {}).get("unit_type") in allowed_units
            )
            and (
                not allowed_passages
                or doc.get("metadata", {}).get("passage_number") in allowed_passages
            )
        ]

    def _fuse_ranked_results(
        self,
        dense_results: List[Dict],
        keyword_results: List[Dict],
        question_results: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        merged: dict[str, Dict] = {}
        rankings = [
            ("dense", dense_results, "score", "probe_dense_score"),
            ("keyword", keyword_results, "keyword_score", "probe_keyword_score"),
            ("question", question_results, "question_score", "probe_question_score"),
        ]
        for method, results, raw_score_key, debug_score_key in rankings:
            for rank, result in enumerate(results, 1):
                key = result.get("chunk_id") or f"{method}-{result.get('chunk_index')}"
                item = merged.setdefault(key, dict(result))
                item.setdefault("score", 0.0)
                item.setdefault("probe_dense_score", 0.0)
                item.setdefault("probe_keyword_score", 0.0)
                item.setdefault("probe_question_score", 0.0)
                item.setdefault("probe_overview_score", 0.0)
                item.setdefault("retrieval_methods", [])
                item["retrieval_method"] = "hybrid"
                item[debug_score_key] = float(result.get(raw_score_key, 0.0))
                item["rrf_score"] = float(item.get("rrf_score", 0.0)) + 1.0 / (
                    settings.rag_rrf_k + rank
                )
                if method not in item["retrieval_methods"]:
                    item["retrieval_methods"].append(method)

        return sorted(
            merged.values(),
            key=lambda item: (
                item.get("rrf_score", 0.0),
                item.get("probe_question_score", 0.0),
                item.get("probe_keyword_score", 0.0),
                item.get("probe_dense_score", 0.0),
            ),
            reverse=True,
        )[:top_k]

    def _question_range_search(
        self,
        query: str,
        top_k: int,
        document_ids: List[str] | None = None,
    ) -> List[Dict]:
        ranges = self._question_ranges(query)
        if not ranges:
            return []

        header_results = []
        numeric_results = []
        allowed = set(document_ids or [])
        for doc in self._docs:
            if allowed and doc.get("document_id") not in allowed:
                continue
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
        for match in re.finditer(
            r"(?:questions?|question|câu hỏi|câu)\s*(?:từ\s+)?(\d{1,2})"
            r"(?:\s*(?:-|–|to)\s*(?:questions?|question|câu hỏi|câu)?\s*(\d{1,2}))?",
            normalized,
        ):
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
            "ảnh",
            "hình",
            "image",
            "writing",
            "task 1",
            "task 2",
            "đề trên",
            "đề writing",
            "tỷ lệ",
            "giá trị",
            "số liệu",
            "có nhắc đến",
            "đã tải",
            "uploaded",
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

    def _elapsed(self, started: float) -> float:
        return round(time.perf_counter() - started, 3)

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
