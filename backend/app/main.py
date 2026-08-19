import asyncio
import json
import logging
import re
import secrets
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import aiofiles
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import settings
from .document_scope import (
    DocumentScope,
    order_metadata_values,
    rank_document_candidates,
    resolve_document_scope,
)
from .document_pipeline import DocumentProcessor
from .intent import (
    dedupe_sources,
    filter_sources_for_intent,
    has_explicit_no_solution_constraint,
    parse_passage_number,
    parse_question_ranges,
    semantic_intent_decision,
)
from .llm import (
    OLLAMA_MODEL,
    OLLAMA_NUM_PREDICT,
    ROUTING_INTENTS,
    OllamaRequestError,
    RouteGatewayDecision,
    classify_direct_source,
    classify_rag_intent,
    conversation_language,
    direct_chat_messages,
    direct_answer_prompt,
    extract_user_facts,
    is_direct_writing_request,
    query_ollama_chat,
    query_ollama,
    rag_prompt,
    response_output_contract,
    response_output_issues,
    response_language_debug,
    response_output_penalty,
    response_retry_prompt,
    resolve_rag_target,
    classify_chat_route,
    select_best_writing_output,
    select_best_response_output,
    stream_ollama,
    translation_retry_prompt,
    writing_output_contract,
    writing_output_issues,
    writing_output_penalty,
    writing_retry_prompt,
)
from .rag import get_store, get_store_manager
from .resource_debug import resource_delta, resource_snapshot
from .schemas import (
    ChatAffinity,
    ChatConversationState,
    ChatMessage,
    ChatRequest,
    ChatUserFact,
    SearchRequest,
    SearchResponse,
    SessionDeleteResponse,
    SessionExpireResponse,
    StatsResponse,
    UploadResponse,
)
from .structured_store import canonical_chunk_id
from .table_operations import (
    comparison_row_facts,
    comparison_row,
    format_number,
    table_cell_value,
    table_change_calculations,
    table_summary_facts,
)


logger = logging.getLogger(__name__)
SESSION_CLEANUP_ERRORS = 0


async def session_cleanup_loop() -> None:
    global SESSION_CLEANUP_ERRORS
    while True:
        await asyncio.sleep(60)
        try:
            deleted = await run_in_threadpool(get_store_manager().cleanup_expired)
            REQUEST_RATE_LIMITER.prune(
                max(settings.chat_rate_window_seconds, settings.upload_rate_window_seconds)
            )
            if deleted:
                logger.info("Expired RAG sessions removed: %s", deleted)
        except Exception:
            SESSION_CLEANUP_ERRORS += 1
            logger.exception("Session RAG cleanup failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(get_store_manager().cleanup_expired)
    cleanup_task = asyncio.create_task(session_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task

UPLOAD_DIR = settings.upload_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENT_PROCESSOR = DocumentProcessor()
LAST_WARMUP_STATUS: dict[str, Any] | None = None
CHAT_CONCURRENCY = asyncio.Semaphore(settings.chat_max_concurrency)
UPLOAD_CONCURRENCY = asyncio.Semaphore(settings.upload_max_concurrency)


class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def check(self, session_id: UUID, action: str, limit: int, window_seconds: int) -> None:
        now = time.monotonic()
        events = self._events[(str(session_id), action)]
        cutoff = now - window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= limit:
            retry_after = max(1, int(window_seconds - (now - events[0])) + 1)
            raise HTTPException(
                status_code=429,
                detail="Bạn thao tác quá nhanh. Vui lòng chờ một chút rồi thử lại.",
                headers={"Retry-After": str(retry_after)},
            )
        events.append(now)

    def prune(self, max_age_seconds: int) -> None:
        cutoff = time.monotonic() - max_age_seconds
        for key, events in list(self._events.items()):
            while events and events[0] <= cutoff:
                events.popleft()
            if not events:
                self._events.pop(key, None)

    def clear_session(self, session_id: UUID) -> None:
        normalized = str(session_id)
        for key in [key for key in self._events if key[0] == normalized]:
            self._events.pop(key, None)

    def stats(self) -> dict[str, int]:
        return {
            "buckets": len(self._events),
            "events": sum(len(events) for events in self._events.values()),
        }


REQUEST_RATE_LIMITER = SlidingWindowRateLimiter()


class SessionChatLockPool:
    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._locks: dict[str, asyncio.Lock] = {}
        self._users: dict[str, int] = {}

    @asynccontextmanager
    async def hold(self, session_id: UUID) -> AsyncIterator[None]:
        key = str(session_id)
        async with self._guard:
            lock = self._locks.setdefault(key, asyncio.Lock())
            self._users[key] = self._users.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            async with self._guard:
                remaining = self._users.get(key, 1) - 1
                if remaining <= 0:
                    self._users.pop(key, None)
                    self._locks.pop(key, None)
                else:
                    self._users[key] = remaining


SESSION_CHAT_LOCKS = SessionChatLockPool()


def enforce_chat_rate(req: ChatRequest) -> None:
    REQUEST_RATE_LIMITER.check(
        req.session_id,
        "chat",
        settings.chat_rate_limit,
        settings.chat_rate_window_seconds,
    )


def enforce_upload_rate(session_id: UUID = Form(...)) -> None:
    REQUEST_RATE_LIMITER.check(
        session_id,
        "upload",
        settings.upload_rate_limit,
        settings.upload_rate_window_seconds,
    )


def require_api_auth(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_auth_required:
        return
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        token,
        settings.api_auth_token,
    ):
        raise HTTPException(
            status_code=401,
            detail="API authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )


app = FastAPI(
    title="Standalone IELTS Chatbot",
    version="1.1.0",
    lifespan=lifespan,
    docs_url=None if settings.api_auth_required else "/docs",
    redoc_url=None if settings.api_auth_required else "/redoc",
    openapi_url=None if settings.api_auth_required else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


def ollama_failure_detail(exc: Exception) -> dict[str, Any]:
    if not settings.debug_payloads:
        return {
            "message": "Không thể kết nối hoặc nhận câu trả lời từ Ollama.",
            "kind": exc.kind if isinstance(exc, OllamaRequestError) else type(exc).__name__,
        }
    diagnostic = (
        exc.debug_detail()
        if isinstance(exc, OllamaRequestError)
        else {"kind": type(exc).__name__, "message": str(exc)[:500]}
    )
    return {
        "message": "Không thể kết nối hoặc nhận câu trả lời từ Ollama.",
        "ollama": diagnostic,
    }


def stream_event(event_type: str, **payload: Any) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


def format_context(sources: list[dict], max_chars_per_source: int | None = None) -> str:
    parts = []
    for index, source in enumerate(sources, 1):
        source_file = source.get("source_file", "unknown")
        pages = source.get("pages") or []
        text = source.get("display_text") or source.get("text", "")
        if max_chars_per_source and len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rsplit(" ", 1)[0] + " ..."
        unit_type = source.get("metadata", {}).get("unit_type")
        role = {
            "question_group": "QUESTION INSTRUCTIONS",
            "question": "QUESTION",
            "passage": "PASSAGE EVIDENCE",
            "document_outline": "DOCUMENT OUTLINE",
            "writing_prompt": "WRITING PROMPT",
            "writing_task": "WRITING TASK",
            "sample_answer": "SAMPLE ANSWER",
            "writing_table": "STRUCTURED TABLE",
            "table": "STRUCTURED TABLE",
            "flowchart": "STRUCTURED FLOWCHART",
            "diagram": "STRUCTURED DIAGRAM",
        }.get(unit_type, "DOCUMENT CONTEXT")
        parts.append(
            f"--- {role} {index} ---\n"
            f"File: {source_file}\n"
            f"Pages: {', '.join(str(page) for page in pages) if pages else 'unknown'}\n"
            f"{text}"
        )
    return "\n\n".join(parts)


def format_solve_context(
    sources: list[dict[str, Any]],
    report: dict[str, Any],
) -> str:
    """Keep each question contract next to only its selected evidence."""
    sources_by_chunk_id = {
        source.get("chunk_id"): source
        for source in sources
        if source.get("chunk_id")
    }
    evidence_by_question = {
        item.get("question_number"): item
        for item in report.get("evidence_by_question") or []
    }
    packets: list[str] = []
    for target in report.get("question_targets") or []:
        number = target.get("question_number")
        lines = [f"=== SOLVE PACKET: QUESTION {number} ==="]
        instructions = str(target.get("instructions") or "").strip()
        if instructions:
            lines.append(f"Instructions: {instructions}")
        lines.append(
            "Question: "
            + str(target.get("question_stem") or target.get("question_text") or "").strip()
        )
        answer_contract = target.get("answer_contract") or {}
        question_type = str(answer_contract.get("kind") or target.get("question_type") or "unknown")
        lines.append(f"Question type: {question_type}")
        options = target.get("answer_options") or []
        if options:
            lines.append("Answer options:")
            lines.extend(
                f"- {option.get('label')}. {str(option.get('text') or '').strip()}"
                for option in options
                if isinstance(option, dict)
            )
        word_limit = target.get("word_limit")
        if word_limit:
            lines.append(f"Maximum answer words: {word_limit}")
        allowed_labels = answer_contract.get("allowed_labels") or []
        if allowed_labels:
            lines.append(f"Required answer label: exactly one of {', '.join(allowed_labels)}")
        relationship_map = answer_contract.get("relationship_map") or {}
        if relationship_map:
            lines.append(
                "Required relationship mapping: "
                + "; ".join(
                    f"{relationship} -> {label}"
                    for relationship, label in relationship_map.items()
                )
            )

        evidence_debug = evidence_by_question.get(number)
        evidence_ids = (
            evidence_debug.get("selected_chunk_ids", [])
            if evidence_debug is not None
            else target.get("evidence_chunk_ids", [])
        )
        lines.append("Passage evidence selected for this question:")
        evidence_added = False
        for index, chunk_id in enumerate(evidence_ids, 1):
            source = sources_by_chunk_id.get(chunk_id)
            if not source:
                continue
            evidence_added = True
            pages = source.get("pages") or []
            page_label = ", ".join(str(page) for page in pages) if pages else "unknown"
            lines.extend(
                [
                    f"[Evidence {number}.{index}] File: {source.get('source_file', 'unknown')}; "
                    f"Pages: {page_label}",
                    _source_text(source),
                ]
            )
        if not evidence_added:
            lines.append("(No passage evidence was selected.)")
        packets.append("\n".join(lines))
    return "\n\n".join(packets) or format_context(sources)


@dataclass(frozen=True)
class DocumentCatalogContext:
    text: str
    document_refs: dict[str, str]
    included_document_ids: tuple[str, ...] = ()
    omitted_document_ids: tuple[str, ...] = ()


def format_document_catalog_context(
    catalog: list[dict],
    message: str = "",
) -> DocumentCatalogContext:
    lines: list[str] = []
    document_refs: dict[str, str] = {}
    if catalog:
        lines.append("Most relevant uploaded document candidates:")
        for index, item in enumerate(catalog, 1):
            document_id = next(iter(item.get("document_ids") or []), "")
            if not document_id:
                continue
            reference = f"D{index}"
            fields = [f"- {reference}: {item.get('source_file', 'unknown')}"]
            for label, key in (
                ("document_types", "document_types"),
                ("task_types", "task_types"),
                ("visual_types", "visual_types"),
                ("section_titles", "section_titles"),
                ("table_columns", "table_columns"),
                ("target_descriptors", "target_descriptors"),
                ("mime_types", "mime_types"),
                ("unit_types", "unit_types"),
            ):
                values = order_metadata_values(message, item.get(key) or [])
                if not values:
                    continue
                included_values: list[str] = []
                for value in values:
                    field = f"{label}={'; '.join([*included_values, value])}"
                    if len(" | ".join([*fields, field])) > settings.target_catalog_document_chars:
                        break
                    included_values.append(value)
                if included_values:
                    fields.append(f"{label}={'; '.join(included_values)}")
            line = " | ".join(fields)
            if len("\n".join([*lines, line])) > settings.target_catalog_chars:
                break
            document_refs[reference] = document_id
            lines.append(line)
    else:
        lines.append("Available uploaded documents: none")

    return DocumentCatalogContext(
        text="\n".join(lines),
        document_refs=document_refs,
        included_document_ids=tuple(document_refs.values()),
        omitted_document_ids=tuple(
            document_id
            for item in catalog
            for document_id in item.get("document_ids") or []
            if document_id and document_id not in document_refs.values()
        ),
    )


def format_route_environment_context(
    catalog: list[dict[str, Any]],
    attached_document_ids: list[str] | None = None,
    *,
    catalog_reference_match: bool = False,
) -> str:
    """Expose source-dependency signals to Patch 0 without document identities."""
    attached = set(attached_document_ids or [])
    available_ids = {
        str(document_id)
        for item in catalog
        for document_id in item.get("document_ids") or []
    }
    return json.dumps(
        {
            "documents_available": bool(available_ids),
            "attached_this_turn": bool(attached.intersection(available_ids)),
            "catalog_reference_match": catalog_reference_match,
        },
        ensure_ascii=False,
    )


def evidence_query_for_sources(sources: list[dict[str, Any]], fallback: str) -> str:
    question_sources = [
        source
        for source in sources
        if source.get("metadata", {}).get("unit_type") == "question"
    ]
    candidates = question_sources or [
        source
        for source in sources
        if source.get("metadata", {}).get("unit_type") == "question_group"
    ]
    queries: list[str] = []
    for source in candidates:
        text = (source.get("display_text") or source.get("text") or "").strip()
        if not text:
            continue
        text = re.sub(r"^\s*\d{1,2}\s*[.)]\s*", "", text)
        option_matches = list(re.finditer(r"(?<![A-Za-z0-9])([A-H])(?=\s+\S)", text))
        if len(option_matches) >= 2:
            text = text[: option_matches[0].start()].strip()
        if text and text not in queries:
            queries.append(text)
    return " ".join(queries).strip() or fallback


def _source_text(source: dict[str, Any]) -> str:
    return (source.get("display_text") or source.get("text") or "").strip()


def _answer_option_labels(text: str) -> list[str]:
    return list(
        dict.fromkeys(
            re.findall(
                r"(?:^|\s)([A-H])(?:[.)]|\s+(?=\S))",
                text,
            )
        )
    )


def _split_answer_options(text: str) -> tuple[str, list[dict[str, str]]]:
    """Split an IELTS question into its stem and labelled answer options."""
    matches = list(
        re.finditer(
            r"(?<![A-Za-z0-9])([A-H])(?:[.)]|\s+)(?=\S)",
            text,
        )
    )
    if len(matches) < 2:
        return text.strip(), []

    sequences: list[list[re.Match[str]]] = []
    for start in range(len(matches) - 1):
        sequence = [matches[start]]
        for match in matches[start + 1 :]:
            if ord(match.group(1)) == ord(sequence[-1].group(1)) + 1:
                sequence.append(match)
            elif ord(match.group(1)) <= ord(sequence[-1].group(1)):
                break
        if len(sequence) >= 2:
            sequences.append(sequence)
    if not sequences:
        return text.strip(), []
    matches = max(sequences, key=lambda sequence: (len(sequence), sequence[0].start()))

    options: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        option_text = text[match.end() : end].strip()
        if option_text:
            options.append({"label": match.group(1), "text": option_text})
    if len(options) < 2:
        return text.strip(), []
    return text[: matches[0].start()].strip(), options


def _clean_group_answer_options(text: str) -> list[dict[str, str]]:
    """Extract a shared labelled list without carrying later instructions/questions."""
    _, options = _split_answer_options(text)
    cleaned: list[dict[str, str]] = []
    for option in options:
        option_text = re.split(
            r"(?i)\b(?:you\s+may|NB\b|questions?\s+\d{1,3})|(?<!\d)\d{1,3}\s*[.)]\s+",
            option["text"],
            maxsplit=1,
        )[0].strip()
        if option_text:
            cleaned.append({"label": option["label"], "text": option_text})
    return cleaned


def _normalize_solve_question_type(
    question_type: str,
    instructions: str,
    question_text: str,
    answer_options: list[dict[str, str]],
    word_limit: int | None,
) -> str:
    """Normalize answer contracts from parser metadata and generic IELTS instructions."""
    raw_type = question_type.strip().lower().replace("-", "_")
    contract_text = f"{instructions}\n{question_text}".casefold()
    if raw_type == "true_false_not_given" or (
        "true" in contract_text
        and "false" in contract_text
        and "not given" in contract_text
    ):
        return "true_false_not_given"
    if raw_type == "yes_no_not_given" or (
        "yes" in contract_text
        and "no" in contract_text
        and "not given" in contract_text
    ):
        return "yes_no_not_given"
    if raw_type == "matching" or (
        answer_options
        and (
            "match each" in contract_text
            or "match the" in contract_text
            or "with one of" in contract_text
        )
    ):
        return "matching"
    if raw_type == "multiple_choice" or answer_options or _question_requires_options(contract_text):
        return "multiple_choice"
    if raw_type in {"short_answer", "short_answer_examples"} or word_limit:
        return "short_answer"
    return raw_type or "unknown"


def _question_requires_options(text: str) -> bool:
    return bool(
        re.search(
            r"(?:from\s+the\s+list\s+below|"
            r"choose\s+(?:the\s+)?(?:correct\s+|appropriate\s+)?letters?|"
            r"which\s+(?:(?:two|three|four|\d+)\s+)?of\s+the\s+following|"
            r"list\s+of\s+headings)",
            text,
            flags=re.IGNORECASE,
        )
    )


def _solve_answer_contract(
    question_type: str,
    answer_options: list[dict[str, str]],
    word_limit: int | None,
) -> dict[str, Any]:
    relationship_map: dict[str, str] = {}
    allowed_labels: list[str] = []
    if question_type == "true_false_not_given":
        allowed_labels = ["TRUE", "FALSE", "NOT GIVEN"]
        relationship_map = {
            "supports": "TRUE",
            "contradicts": "FALSE",
            "absent": "NOT GIVEN",
        }
    elif question_type == "yes_no_not_given":
        allowed_labels = ["YES", "NO", "NOT GIVEN"]
        relationship_map = {
            "supports": "YES",
            "contradicts": "NO",
            "absent": "NOT GIVEN",
        }
    elif question_type in {"multiple_choice", "matching"}:
        allowed_labels = [option["label"] for option in answer_options]
    return {
        "kind": question_type,
        "allowed_labels": allowed_labels,
        "requires_single_label": bool(allowed_labels),
        "requires_options": question_type in {"multiple_choice", "matching"},
        "relationship_map": relationship_map,
        "word_limit": word_limit,
    }


def _question_word_limit(text: str) -> int | None:
    match = re.search(
        r"\bNO\s+MORE\s+THAN\s+(ONE|TWO|THREE|FOUR|\d+)\s+WORDS?\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"\b(ONE|TWO|THREE|FOUR|\d+)\s+WORDS?\s+ONLY\b",
            text,
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    value = match.group(1).upper()
    return {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4}.get(value, int(value) if value.isdigit() else None)


def solve_question_packets(
    message: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one grounded context packet for every exact requested question."""
    packets: list[dict[str, Any]] = []
    exact_sources = exact_question_sources(message, sources)
    for number in requested_question_numbers(message):
        matches = dedupe_sources([
            source
            for source in exact_sources
            if source.get("metadata", {}).get("question_range") == [number, number]
        ])
        if len(matches) != 1:
            continue
        question = matches[0]
        metadata = question.get("metadata", {})
        document_id = question.get("document_id")
        question_passage_number = metadata.get("passage_number")
        parent_id = metadata.get("parent_id")
        group_candidates = [
            source
            for source in sources
            if source.get("document_id") == document_id
            and source.get("metadata", {}).get("unit_type") == "question_group"
            and (
                canonical_chunk_id(source.get("chunk_id"), document_id)
                == canonical_chunk_id(parent_id, document_id)
                if parent_id
                else _range_contains(source.get("metadata", {}).get("question_range"), number)
            )
        ]
        group_candidates = dedupe_sources(group_candidates)
        group = group_candidates[0] if len(group_candidates) == 1 else None
        group_metadata = group.get("metadata", {}) if group else {}
        group_passage_number = group_metadata.get("passage_number")
        passage_number = (
            question_passage_number
            if question_passage_number is not None
            else group_passage_number
        )
        question_text = _source_text(question)
        instructions = str(group_metadata.get("instructions") or "").strip()
        question_without_number = re.sub(
            r"^\s*\d{1,3}\s*[.)]\s*",
            "",
            question_text,
        ).strip()
        question_stem, answer_options = _split_answer_options(question_without_number)
        source_question_type = str(
            metadata.get("question_type") or group_metadata.get("question_type") or "unknown"
        )
        word_limit = _question_word_limit(instructions or _source_text(group or {}))
        group_answer_options = _clean_group_answer_options(_source_text(group or {}))
        preliminary_options = answer_options or group_answer_options
        question_type = _normalize_solve_question_type(
            source_question_type,
            instructions,
            question_text,
            preliminary_options,
            word_limit,
        )
        if question_type == "matching" and not answer_options:
            answer_options = group_answer_options
        option_labels = [option["label"] for option in answer_options]
        if not option_labels:
            option_labels = _answer_option_labels(question_text)
        answer_contract = _solve_answer_contract(
            question_type,
            answer_options,
            word_limit,
        )
        evidence = [
            source
            for source in sources
            if source.get("document_id") == document_id
            and source.get("metadata", {}).get("unit_type") == "passage"
            and group is not None
            and passage_number is not None
            and not (
                question_passage_number is not None
                and group_passage_number is not None
                and question_passage_number != group_passage_number
            )
            and source.get("metadata", {}).get("passage_number") == passage_number
        ]
        evidence = dedupe_sources(evidence)
        warnings: list[str] = []
        if not group_candidates:
            warnings.append("missing_question_group")
        elif len(group_candidates) > 1:
            warnings.append("ambiguous_question_group")
        if (
            question_passage_number is not None
            and group_passage_number is not None
            and question_passage_number != group_passage_number
        ):
            warnings.append("question_group_passage_mismatch")
        if passage_number is None:
            warnings.append("missing_passage_link")
        if answer_contract["requires_options"] and len(option_labels) < 2:
            warnings.append("missing_answer_options")
        structural_warnings = {
            "missing_question_group",
            "ambiguous_question_group",
            "question_group_passage_mismatch",
            "missing_passage_link",
            "missing_answer_options",
        }
        packets.append(
            {
                "question_number": number,
                "question_chunk_id": question.get("chunk_id"),
                "question_text": question_text,
                "question_stem": question_stem,
                "question_type": question_type,
                "source_question_type": source_question_type,
                "instructions": instructions,
                "answer_option_labels": option_labels,
                "answer_options": answer_options,
                "answer_contract": answer_contract,
                "word_limit": word_limit,
                "document_id": document_id,
                "passage_number": passage_number,
                "question_passage_number": question_passage_number,
                "group_passage_number": group_passage_number,
                "parent_id": parent_id,
                "question_group_chunk_ids": [
                    source.get("chunk_id") for source in group_candidates
                ],
                "evidence_chunk_ids": [source.get("chunk_id") for source in evidence],
                "warnings": warnings,
                "context_ready": not any(
                    warning in structural_warnings for warning in warnings
                ),
            }
        )
    return packets


def evidence_query_for_solve_packet(packet: dict[str, Any], fallback: str) -> str:
    question_stem = re.sub(
        r"\s+",
        " ",
        str(packet.get("question_stem") or ""),
    ).strip()
    if question_stem:
        return question_stem

    question_text = re.sub(
        r"^\s*\d{1,3}\s*[.)]\s*",
        "",
        str(packet.get("question_text") or ""),
    ).strip()
    return question_text or fallback


def solve_sources_with_selected_evidence(
    sources: list[dict[str, Any]],
    question_context: list[dict[str, Any]],
    evidence_context: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep solve contracts and only passage chunks selected as evidence."""
    contract_sources = [
        source
        for source in sources + question_context
        if source.get("metadata", {}).get("unit_type") != "passage"
    ]
    return dedupe_sources(contract_sources + evidence_context)


def requested_question_numbers(message: str) -> list[int]:
    numbers: list[int] = []
    for start, end in parse_question_ranges(message):
        for number in range(start, end + 1):
            if number not in numbers:
                numbers.append(number)
    return numbers


def exact_question_sources(
    message: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    requested = set(requested_question_numbers(message))
    if not requested:
        return []
    results: list[dict[str, Any]] = []
    for source in sources:
        metadata = source.get("metadata", {})
        question_range = metadata.get("question_range")
        if (
            metadata.get("unit_type") != "question"
            or not isinstance(question_range, list)
            or len(question_range) != 2
        ):
            continue
        start, end = int(question_range[0]), int(question_range[1])
        if start == end and start in requested:
            results.append(source)
    return results


def compact_probe_debug(probe: dict) -> dict:
    return {
        "has_hits": probe.get("has_hits", False),
        "has_strong_hits": probe.get("has_strong_hits", False),
        "top_score": probe.get("top_score", 0.0),
        "top_fused_score": probe.get("top_fused_score", 0.0),
        "top_keyword_score": probe.get("top_keyword_score", 0.0),
        "top_question_score": probe.get("top_question_score", 0.0),
        "top_overview_score": probe.get("top_overview_score", 0.0),
        "has_document_intent": probe.get("has_document_intent", False),
        "is_overview": probe.get("is_overview", False),
        "results": [
            {
                "source_file": item.get("source_file"),
                "pages": item.get("pages"),
                "score": item.get("score", 0.0),
                "dense": item.get("probe_dense_score", 0.0),
                "keyword": item.get("probe_keyword_score", 0.0),
                "question": item.get("probe_question_score", 0.0),
                "overview": item.get("probe_overview_score", 0.0),
                "fused": item.get("rrf_score", 0.0),
                "methods": item.get("retrieval_methods", []),
                "chunk_id": item.get("chunk_id"),
                "unit_type": item.get("metadata", {}).get("unit_type"),
                "chunk_reason": item.get("metadata", {}).get("chunk_reason"),
                "passage_number": item.get("metadata", {}).get("passage_number"),
                "question_range": item.get("metadata", {}).get("question_range"),
                "parent_id": item.get("metadata", {}).get("parent_id"),
                "text_preview": " ".join(
                    (item.get("display_text") or item.get("text") or "").split()
                )[:220],
            }
            for item in (probe.get("results") or [])[:3]
        ],
    }


NO_RAG_MATCH_RESPONSE = (
    "Mình chưa tìm thấy nội dung phù hợp trong tài liệu đã upload để trả lời câu hỏi này. "
    "Bạn có thể hỏi rõ hơn theo tên bài, số trang, hoặc upload lại tài liệu nếu phần đó nằm trong bảng/ảnh chưa được trích xuất tốt."
)

NO_RAG_MATCH_RESPONSE_EN = (
    "I could not find relevant content in the uploaded material to answer this question. "
    "Please specify the document, section, or page, or upload the material again if the content is in a table or image that was not extracted correctly."
)

LEGACY_NO_RAG_MATCH_RESPONSES = (
    "Mình không tìm thấy thông tin này trong tài liệu đã chọn.",
    "I cannot find this information in the selected uploaded material.",
)

SYSTEM_NO_RAG_MATCH_RESPONSES = (
    NO_RAG_MATCH_RESPONSE,
    NO_RAG_MATCH_RESPONSE_EN,
    *LEGACY_NO_RAG_MATCH_RESPONSES,
)

AMBIGUOUS_DOCUMENT_RESPONSE = (
    "Mình chưa xác định được bạn đang hỏi tài liệu nào vì có nhiều file phù hợp. "
    "Vui lòng nêu tên file hoặc đính kèm lại đúng tài liệu cần hỏi."
)

INCOMPLETE_QUESTION_RESPONSE = (
    "Mình đã tìm thấy câu hỏi nhưng phần lựa chọn hoặc dữ liệu cần thiết để giải chưa được "
    "trích xuất đầy đủ. Vì vậy mình chưa thể xác định đáp án đáng tin cậy từ tài liệu hiện có."
)

ROUTE_UNDETERMINED_RESPONSE = (
    "Mình chưa xác định chắc chắn câu hỏi này có cần dùng tài liệu đã tải lên hay không. "
    "Vui lòng nói rõ bạn muốn hỏi kiến thức chung hay nội dung trong một tài liệu cụ thể."
)

INTENT_UNDETERMINED_RESPONSE = (
    "Mình chưa xác định được thao tác bạn muốn thực hiện với tài liệu. "
    "Vui lòng nói rõ bạn muốn xem, dịch, giải thích, trả lời câu hỏi hay phân tích nội dung."
)

TRANSLATION_VALIDATION_FAILURE_VI = (
    "Mình chưa tạo được bản dịch đầy đủ bằng tiếng Việt từ nội dung đã chọn. "
    "Vui lòng thử lại."
)

TRANSLATION_VALIDATION_FAILURE_EN = (
    "I could not produce a complete English translation from the selected content. "
    "Please try again."
)

LANGUAGE_VALIDATION_FAILURE_VI = (
    "Mình chưa tạo được câu trả lời đúng ngôn ngữ bạn yêu cầu. Vui lòng thử lại."
)

LANGUAGE_VALIDATION_FAILURE_EN = (
    "I could not produce the response in the language you requested. Please try again."
)

SOLVE_VALIDATION_FAILURE_VI = (
    "Mình chưa xác định được đáp án đáng tin cậy từ bằng chứng trong tài liệu. "
    "Vui lòng thử lại hoặc nêu rõ câu hỏi cần kiểm tra."
)

SOLVE_VALIDATION_FAILURE_EN = (
    "I could not determine a reliable answer from the available passage evidence. "
    "Please try again or specify the question you want checked."
)


def hard_validation_failure(
    query_intent: str,
    language: str | None,
    issues: list[str],
) -> str | None:
    """Return a safe response only for explicit language/translation contracts."""
    language_issue = any("not written in" in issue for issue in issues)
    missing_translation_items = query_intent == "translate_questions" and any(
        "missing question numbers" in issue for issue in issues
    )
    if not language_issue and not missing_translation_items:
        return None
    if query_intent in {"translate_questions", "translate_content"}:
        return (
            TRANSLATION_VALIDATION_FAILURE_EN
            if language == "English"
            else TRANSLATION_VALIDATION_FAILURE_VI
        )
    return LANGUAGE_VALIDATION_FAILURE_EN if language == "English" else LANGUAGE_VALIDATION_FAILURE_VI


def solve_validation_failure(message: str) -> str:
    return (
        SOLVE_VALIDATION_FAILURE_EN
        if conversation_language(message) == "English"
        else SOLVE_VALIDATION_FAILURE_VI
    )


def document_extraction_failure_detail(document: Any) -> str:
    metadata = document.metadata or {}
    ocr_engine = metadata.get("ocr_engine")
    ocr_metadata = metadata.get("ocr_metadata") or {}
    attempts = ocr_metadata.get("cascade_attempts") or []
    if not attempts and isinstance(ocr_metadata.get("attempt"), dict):
        attempts = [ocr_metadata["attempt"]]
    if not attempts and ocr_metadata.get("error"):
        attempts = [ocr_metadata]
    errors = [
        str(attempt.get("error"))
        for attempt in attempts
        if isinstance(attempt, dict) and attempt.get("error")
    ]
    reasons = [
        str(attempt.get("engine") or attempt.get("reason"))
        for attempt in attempts
        if isinstance(attempt, dict) and (attempt.get("engine") or attempt.get("reason"))
    ]

    if ocr_engine == "rapidocr_failed":
        if any("rapidocr_unavailable" in reason for reason in reasons):
            return (
                "RapidOCR chưa khả dụng trong môi trường backend hiện tại, nên ảnh chưa được OCR. "
                "Hãy cài đúng rapidocr/torch CUDA rồi restart backend."
            )
        if errors:
            return f"RapidOCR không trích xuất được ảnh. Lỗi OCR đầu tiên: {errors[0][:300]}"

    return "Không trích xuất được văn bản từ tài liệu. File có thể quá mờ, không có chữ, hoặc OCR chưa phù hợp."


def generation_fallback(prepared: "ChatPreparation", message: str = "") -> str:
    if prepared.route_used.startswith("vector_rag"):
        return NO_RAG_MATCH_RESPONSE
    if conversation_language(message) == "English":
        return "What would you like help with? Please describe your request more specifically."
    return "Bạn muốn mình hỗ trợ nội dung gì? Hãy mô tả yêu cầu cụ thể hơn nhé."


def no_rag_match_response(message: str) -> str:
    contract = response_output_contract(
        message,
        "semantic_qa",
        allow_solution=False,
    )
    return NO_RAG_MATCH_RESPONSE_EN if contract.language == "English" else NO_RAG_MATCH_RESPONSE


def remove_appended_no_match_response(text: str) -> tuple[str, str | None]:
    stripped = text.strip()
    for response in SYSTEM_NO_RAG_MATCH_RESPONSES:
        marker = f"\n\n{response}"
        if stripped.endswith(marker):
            substantive = stripped[: -len(marker)].rstrip()
            if substantive:
                return substantive, response
    return text, None


def pending_no_match_suffix_length(text: str) -> int:
    """Hold only text that may become a system-owned no-match suffix."""
    best = 0
    for response in SYSTEM_NO_RAG_MATCH_RESPONSES:
        marker = f"\n\n{response}"
        max_prefix = min(len(text), len(marker))
        for size in range(max_prefix, best, -1):
            if text.endswith(marker[:size]):
                best = size
                break
        marker_start = text.rfind(marker)
        if marker_start >= 0 and not text[marker_start + len(marker) :].strip():
            best = max(best, len(text) - marker_start)
    return best


def generation_temperature(prepared: "ChatPreparation") -> float:
    if is_writing_response(prepared):
        return 0.1
    if prepared.route_used.startswith("vector_rag"):
        return 0.2
    return settings.ollama_direct_temperature


def _solve_answer_segment(
    text: str,
    question_number: int,
    requested_numbers: list[int],
) -> str:
    if len(requested_numbers) == 1:
        return text
    marker = re.compile(
        rf"(?im)^\s*(?:question|câu(?:\s+hỏi)?)?\s*{question_number}\s*[.):]"
    )
    match = marker.search(text)
    if not match:
        return ""
    following = [
        re.compile(rf"(?im)^\s*(?:question|câu(?:\s+hỏi)?)?\s*{number}\s*[.):]").search(
            text,
            match.end(),
        )
        for number in requested_numbers
        if number != question_number
    ]
    ends = [candidate.start() for candidate in following if candidate]
    return text[match.start() : min(ends) if ends else len(text)]


def _solve_answer_head(segment: str, question_number: int) -> str:
    answer = re.sub(
        rf"(?is)^\s*(?:[-*]\s*)?(?:\*{{0,2}})?(?:question|câu(?:\s+hỏi)?)?\s*"
        rf"{question_number}\s*[.):]\s*(?:\*{{0,2}})?",
        "",
        segment,
        count=1,
    ).strip()
    first_line = next((line.strip() for line in answer.splitlines() if line.strip()), "")
    first_line = re.sub(r"(?i)^\s*(?:answer|đáp\s+án)\s*[:\-–—]?\s*", "", first_line)
    first_line = re.split(
        r"(?i)\s+(?:[-–—:]\s*)?(?:evidence|bằng\s+chứng|relationship|giải\s+thích)\s*:",
        first_line,
        maxsplit=1,
    )[0]
    return first_line.strip().strip("*_`")


def _solve_relationship(segment: str) -> str | None:
    match = re.search(
        r"(?im)^\s*(?:[-*]\s*)?(?:\*{0,2})relationship\s*:\s*"
        r"(?:\*{0,2})(supports|contradicts|absent)\b",
        segment,
    )
    return match.group(1).lower() if match else None


def _solve_evidence(segment: str) -> str:
    match = re.search(
        r"(?is)(?:^|\n)\s*(?:[-*]\s*)?(?:\*{0,2})"
        r"(?:evidence|bằng\s+chứng)\s*:\s*(?:\*{0,2})(.*?)"
        r"(?=\n\s*(?:[-*]\s*)?(?:\*{0,2})relationship\s*:|\Z)",
        segment,
    )
    return match.group(1).strip().strip('"“”') if match else ""


def _normalized_evidence_text(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))


def _paragraph_evidence_scope(text: str, label: str) -> str | None:
    start = re.search(
        rf"(?<![\w]){re.escape(label)}\s+(?=[A-Z][a-z])",
        text,
    )
    if not start:
        return None
    next_label = chr(ord(label) + 1) if label < "Z" else None
    end = (
        re.search(
            rf"(?<![\w]){re.escape(next_label)}\s+(?=[A-Z][a-z])",
            text[start.end() :],
        )
        if next_label
        else None
    )
    end_index = start.end() + end.start() if end else len(text)
    return text[start.end() : end_index]


def _evidence_source_texts(
    packet: dict[str, Any],
    report: dict[str, Any],
    sources: list[dict[str, Any]],
) -> list[str]:
    number = packet.get("question_number")
    selection = next(
        (
            item
            for item in report.get("evidence_by_question") or []
            if item.get("question_number") == number
        ),
        None,
    )
    selected_ids = set(
        (selection or {}).get("selected_chunk_ids")
        or packet.get("evidence_chunk_ids")
        or []
    )
    texts = [
        _source_text(source)
        for source in sources
        if source.get("chunk_id") in selected_ids
        and source.get("metadata", {}).get("unit_type") == "passage"
    ]
    paragraph_match = re.search(
        r"\bparagraph\s+([A-Z])\b",
        str(packet.get("question_stem") or packet.get("question_text") or ""),
        flags=re.IGNORECASE,
    )
    if not paragraph_match:
        return texts
    label = paragraph_match.group(1).upper()
    paragraph_texts = [
        scoped
        for text in texts
        if (scoped := _paragraph_evidence_scope(text, label))
    ]
    return paragraph_texts or texts


def _evidence_appears_in_sources(evidence: str, source_texts: list[str]) -> bool:
    normalized_sources = [_normalized_evidence_text(text) for text in source_texts]
    evidence_parts = [
        _normalized_evidence_text(part)
        for part in re.split(r"(?:\.{3,}|…)", evidence)
    ]
    evidence_parts = [part for part in evidence_parts if len(part.split()) >= 3]
    return bool(evidence_parts) and all(
        any(part in source for source in normalized_sources)
        for part in evidence_parts
    )


def _normalized_option_text(text: str) -> str:
    return " ".join(re.findall(r"[\w']+", text.casefold(), flags=re.UNICODE))


def _option_label_from_answer(
    answer_head: str,
    options: list[dict[str, str]],
) -> str | None:
    normalized_answer = _normalized_option_text(answer_head)
    if not normalized_answer:
        return None
    matches = []
    for option in options:
        option_text = _normalized_option_text(str(option.get("text") or ""))
        if option_text and (
            normalized_answer == option_text
            or option_text in normalized_answer
        ):
            matches.append(str(option.get("label") or "").upper())
    unique = list(dict.fromkeys(label for label in matches if label))
    return unique[0] if len(unique) == 1 else None


def _replace_solve_answer(
    text: str,
    question_number: int,
    requested_numbers: list[int],
    answer: str,
) -> tuple[str, bool]:
    marker = re.compile(
        rf"(?im)^(\s*(?:[-*]\s*)?(?:\*{{0,2}})?(?:question|câu(?:\s+hỏi)?)?\s*"
        rf"{question_number}\s*[.):]\s*(?:\*{{0,2}})?)[^\n]*"
    )
    replaced, count = marker.subn(lambda match: f"{match.group(1)}{answer}", text, count=1)
    if count:
        return replaced, replaced != text
    if len(requested_numbers) == 1:
        return f"Question {question_number}: {answer}\n{text.strip()}", True
    return text, False


def normalize_solve_output(
    text: str,
    report: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Apply only deterministic mappings declared by each solve contract."""
    normalized = text
    adjustments: list[dict[str, Any]] = []
    requested = list(report.get("requested_question_numbers") or [])
    for packet in report.get("question_targets") or []:
        number = int(packet["question_number"])
        segment = _solve_answer_segment(normalized, number, requested)
        if not segment:
            continue
        contract = packet.get("answer_contract") or {}
        answer: str | None = None
        adjustment_reason: str | None = None
        relationship = _solve_relationship(segment)
        relationship_map = contract.get("relationship_map") or {}
        if relationship and relationship in relationship_map:
            answer = relationship_map[relationship]
            adjustment_reason = "relationship_mapping"
        elif contract.get("kind") in {"multiple_choice", "matching"}:
            answer = _option_label_from_answer(
                _solve_answer_head(segment, number),
                packet.get("answer_options") or [],
            )
            adjustment_reason = "option_text_mapping"
        if not answer:
            continue
        normalized, changed = _replace_solve_answer(
            normalized,
            number,
            requested,
            answer,
        )
        if changed:
            adjustments.append(
                {
                    "question_number": number,
                    "answer": answer,
                    "reason": adjustment_reason,
                }
            )
    return normalized, adjustments


def _selected_answer_labels(answer_head: str, allowed_labels: list[str]) -> list[str]:
    selected: list[str] = []
    for label in sorted(allowed_labels, key=len, reverse=True):
        if re.search(rf"(?i)(?<![A-Z]){re.escape(label)}(?![A-Z])", answer_head):
            selected.append(label)
    return selected


def _effective_solve_contract(packet: dict[str, Any]) -> dict[str, Any]:
    contract = dict(packet.get("answer_contract") or {})
    question_type = str(contract.get("kind") or packet.get("question_type") or "unknown")
    allowed_labels = list(
        contract.get("allowed_labels")
        or packet.get("answer_option_labels")
        or []
    )
    if not allowed_labels and question_type == "true_false_not_given":
        allowed_labels = ["TRUE", "FALSE", "NOT GIVEN"]
    elif not allowed_labels and question_type == "yes_no_not_given":
        allowed_labels = ["YES", "NO", "NOT GIVEN"]
    contract.setdefault("kind", question_type)
    contract["allowed_labels"] = allowed_labels
    contract.setdefault(
        "requires_single_label",
        question_type in {
            "multiple_choice",
            "matching",
            "true_false_not_given",
            "yes_no_not_given",
        },
    )
    contract.setdefault("requires_options", question_type in {"multiple_choice", "matching"})
    return contract


def solve_output_issues(
    text: str,
    report: dict[str, Any],
    sources: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Validate solve response shape without judging semantic correctness."""
    issues: list[str] = []
    requested = list(report.get("requested_question_numbers") or [])
    for packet in report.get("question_targets") or []:
        number = int(packet["question_number"])
        segment = _solve_answer_segment(text, number, requested)
        if not segment:
            issues.append(f"Question {number} is missing from the response.")
            continue
        has_explicit_contract = bool(packet.get("answer_contract"))
        answer_contract = _effective_solve_contract(packet)
        question_type = answer_contract["kind"]
        answer_head = _solve_answer_head(segment, number)
        allowed_labels = answer_contract["allowed_labels"]
        selected: list[str] = []
        if answer_contract.get("requires_options") and len(allowed_labels) < 2:
            issues.append(f"Question {number} is missing its answer option contract.")
        elif answer_contract.get("requires_single_label") or question_type in {
            "multiple_choice",
            "matching",
            "true_false_not_given",
            "yes_no_not_given",
        }:
            selected = _selected_answer_labels(answer_head, allowed_labels)
            starts_with_label = any(
                re.match(rf"(?i)^\s*{re.escape(label)}(?:\b|\s)", answer_head)
                for label in allowed_labels
            )
            exact_contract_label = any(
                answer_head.strip().casefold() == label.casefold()
                for label in allowed_labels
            )
            valid_label = exact_contract_label if has_explicit_contract else starts_with_label
            if len(selected) != 1 or not valid_label:
                issues.append(f"Question {number} is missing a valid answer option label.")
        elif question_type == "short_answer" and packet.get("word_limit"):
            word_count = len(re.findall(r"\b[\w'-]+\b", answer_head, flags=re.UNICODE))
            if not answer_head:
                issues.append(f"Question {number} is missing a short answer.")
            elif word_count > int(packet["word_limit"]):
                issues.append(
                    f"Question {number} exceeds its {packet['word_limit']}-word answer limit."
                )
        if not has_explicit_contract:
            continue

        evidence = _solve_evidence(segment)
        relationship = _solve_relationship(segment)
        if not evidence:
            issues.append(f"Question {number} is missing an Evidence field.")
        if not relationship:
            issues.append(f"Question {number} is missing a valid Relationship field.")
        elif (
            question_type in {"multiple_choice", "matching", "short_answer"}
            and relationship != "supports"
        ):
            issues.append(f"Question {number} must use Relationship: supports for the selected answer.")
        if not evidence:
            continue

        if sources and relationship != "absent" and not _evidence_appears_in_sources(
            evidence,
            _evidence_source_texts(packet, report, sources),
        ):
            issues.append(
                f"Question {number} Evidence is not quoted from its selected passage evidence."
            )
    return issues


def select_best_solve_output(
    first: str,
    retry: str,
    contract: Any,
    report: dict[str, Any],
    sources: list[dict[str, Any]],
) -> str:
    return min(
        (first, retry),
        key=lambda text: (
            len(solve_output_issues(text, report, sources)),
            response_output_penalty(text, contract),
        ),
    )


def generation_candidate_debug(text: str, max_chars: int = 4_000) -> dict[str, Any]:
    """Keep generation diagnostics useful without copying unbounded model output."""
    return {
        "text": text[:max_chars],
        "char_count": len(text),
        "truncated": len(text) > max_chars,
    }


DirectGenerationRetry = Callable[
    [list[str]],
    Awaitable[tuple[str | None, str | None]],
]
INCOMPLETE_GENERATION_ISSUE = (
    "The response stopped because the model reached the output length limit."
)


def _select_complete_candidate(
    first: str,
    retry: str,
    *,
    first_incomplete: bool,
    retry_incomplete: bool,
    selector: Callable[[str, str], str],
) -> str:
    if first_incomplete != retry_incomplete:
        return retry if not retry_incomplete else first
    return selector(first, retry)


async def generate_answer(
    prepared: "ChatPreparation",
    message: str,
    *,
    initial_answer: str | None = None,
    initial_done_reason: str | None = None,
    direct_source_available: bool = False,
    direct_retry: DirectGenerationRetry | None = None,
) -> str:
    prompt = prepared.prompt or ""
    answer = initial_answer
    if answer is None:
        answer = await query_ollama(prompt, temperature=generation_temperature(prepared))
    selected_done_reason = initial_done_reason
    first_incomplete = initial_done_reason == "length"

    direct_writing = (
        prepared.route_used == "base_model"
        and prepared.query_intent == "direct"
        and is_direct_writing_request(message)
    )
    direct_writing_contract = writing_output_contract(message) if direct_writing else None
    apply_writing_contract = prepared.query_intent == "writing_generation" or bool(
        direct_writing_contract and direct_source_available
    )

    if apply_writing_contract:
        contract = direct_writing_contract or writing_output_contract(message)
        retryable_issues = writing_output_issues(answer, contract)
        issues = list(retryable_issues)
        if first_incomplete:
            issues.append(INCOMPLETE_GENERATION_ISSUE)
        generation_debug = prepared.debug.setdefault("generation", {})
        generation_debug["writing_contract"] = {
            "language": contract.language,
            "min_words": contract.min_words,
            "max_words": contract.max_words,
            "target_words": list(contract.target_words) if contract.target_words else None,
            "single_paragraph": contract.single_paragraph,
            "overview_only": contract.overview_only,
            "first_draft_issues": issues,
            "first_done_reason": initial_done_reason,
        }
        if retryable_issues:
            if direct_retry:
                retry, retry_done_reason = await direct_retry(
                    contract.prompt_lines()
                    + ["- Finish every requested section and do not stop mid-sentence."]
                )
                retry_available = retry is not None
                if retry is None:
                    retry = answer
                    retry_done_reason = initial_done_reason
            else:
                try:
                    retry = await query_ollama(
                        writing_retry_prompt(prompt, contract),
                        temperature=0.1,
                        max_attempts=1,
                    )
                    retry_done_reason = None
                    retry_available = True
                except OllamaRequestError as exc:
                    retry = answer
                    retry_done_reason = initial_done_reason
                    retry_available = False
                    generation_debug.update(
                        {
                            "contract_retry_status": "failed",
                            "contract_retry_error": exc.kind,
                        }
                    )
            retry_incomplete = retry_done_reason == "length"
            selected = _select_complete_candidate(
                answer,
                retry,
                first_incomplete=first_incomplete,
                retry_incomplete=retry_incomplete,
                selector=lambda first, second: select_best_writing_output(
                    first,
                    second,
                    contract,
                ),
            )
            generation_debug["retry_used"] = True
            generation_debug["retry_succeeded"] = retry_available
            generation_debug["retry_endpoint"] = "chat" if direct_retry else "generate"
            generation_debug["retry_done_reason"] = retry_done_reason
            generation_debug["candidate_penalties"] = {
                "first": list(writing_output_penalty(answer, contract)),
                "retry": list(writing_output_penalty(retry, contract)),
            }
            selected_is_first = selected == answer
            generation_debug["selected_candidate"] = "first" if selected_is_first else "retry"
            answer = selected
            selected_done_reason = (
                initial_done_reason if selected_is_first else retry_done_reason
            )
        else:
            generation_debug["retry_used"] = False
        final_issues = writing_output_issues(answer, contract)
        if selected_done_reason == "length":
            final_issues.append(INCOMPLETE_GENERATION_ISSUE)
        generation_debug["final_issues"] = final_issues
        if final_issues:
            generation_debug["validation_degraded"] = True
            failure = (
                None
                if direct_writing
                else hard_validation_failure(
                    prepared.query_intent,
                    contract.language,
                    final_issues,
                )
            )
            if failure:
                generation_debug["validation_failed_closed"] = True
                generation_debug["returned_validation_fallback"] = failure
                return failure
    else:
        allow_solution = bool(prepared.debug.get("intent_decision", {}).get("allow_solution", False))
        contract = response_output_contract(
            message,
            prepared.query_intent,
            allow_solution=allow_solution,
            explicit_no_solution=has_explicit_no_solution_constraint(message),
        )
        if direct_writing and not apply_writing_contract:
            contract = replace(contract, language=conversation_language(message))
        solve_report = (
            prepared.debug.get("retrieval", {}).get("solve_context_report", {})
            if prepared.query_intent == "solve_questions"
            else {}
        )
        first_raw_answer = answer
        first_cleanup: str | None = None
        if prepared.query_intent in {"semantic_qa", "explain_questions"}:
            answer, first_cleanup = remove_appended_no_match_response(answer)
        first_adjustments: list[dict[str, Any]] = []
        if solve_report:
            answer, first_adjustments = normalize_solve_output(answer, solve_report)
        first_solve_issues = (
            solve_output_issues(answer, solve_report, prepared.sources)
            if solve_report
            else []
        )
        issues = response_output_issues(answer, contract) + first_solve_issues
        if first_incomplete:
            issues.append(INCOMPLETE_GENERATION_ISSUE)
        generation_debug = prepared.debug.setdefault("generation", {})
        generation_debug["response_contract"] = {
            "language": contract.language,
            "enforce_language": contract.enforce_language,
            "forbid_solution": contract.forbid_solution,
            "allow_source_language_fields": contract.allow_source_language_fields,
            "required_question_numbers": list(contract.required_question_numbers),
            "first_draft_cleanup": first_cleanup,
            "plan_duration_value": contract.plan_duration_value,
            "plan_duration_unit": contract.plan_duration_unit,
            "max_daily_minutes": contract.max_daily_minutes,
            "first_draft_issues": issues,
            "first_draft_language": response_language_debug(
                answer,
                contract.language,
                allow_source_language_fields=contract.allow_source_language_fields,
            ),
            "first_done_reason": initial_done_reason,
        }
        if solve_report:
            generation_debug["solve_contract"] = {
                "question_packets": solve_report.get("question_targets", []),
                "first_candidate_raw": generation_candidate_debug(first_raw_answer),
                "first_candidate_normalized": generation_candidate_debug(answer),
                "first_draft_adjustments": first_adjustments,
                "first_draft_issues": first_solve_issues,
            }
        enforce_review_contract = requires_reviewed_generation(prepared, message)
        should_retry = bool(issues) and (
            prepared.query_intent == "translate_questions"
            or (
                enforce_review_contract
                and any("not written in" in issue for issue in issues)
            )
            or any("malformed Markdown table" in issue for issue in issues)
            or any("conversation role prefix" in issue for issue in issues)
            or any("plan timeline" in issue for issue in issues)
            or any("plan periods" in issue for issue in issues)
            or any("daily time limit" in issue for issue in issues)
            or bool(first_solve_issues)
            or (
                has_explicit_no_solution_constraint(message)
                and contract.forbid_solution
            )
        )
        if should_retry:
            retry_prompt = (
                translation_retry_prompt(prompt, contract)
                if prepared.query_intent in {"translate_questions", "translate_content"}
                else response_retry_prompt(
                    prompt,
                    contract,
                    prepared.query_intent,
                    previous_candidate=answer if solve_report else "",
                    validation_issues=issues if solve_report else None,
                )
            )
            retry_contract = contract.prompt_lines()
            if first_incomplete:
                retry_contract.append(
                    "- Finish every requested section and structure; do not stop mid-sentence."
                )
            if direct_retry:
                retry, retry_done_reason = await direct_retry(retry_contract)
                retry_available = retry is not None
                if retry is None:
                    retry = answer
                    retry_done_reason = initial_done_reason
            else:
                try:
                    retry = await query_ollama(
                        retry_prompt,
                        temperature=0.1,
                        max_attempts=1,
                    )
                    retry_done_reason = None
                    retry_available = True
                except OllamaRequestError as exc:
                    retry = answer
                    retry_done_reason = initial_done_reason
                    retry_available = False
                    generation_debug.update(
                        {
                            "contract_retry_status": "failed",
                            "contract_retry_error": exc.kind,
                        }
                    )
            retry_incomplete = retry_done_reason == "length"
            retry_raw = retry
            retry_cleanup: str | None = None
            if prepared.query_intent in {"semantic_qa", "explain_questions"}:
                retry, retry_cleanup = remove_appended_no_match_response(retry)
                if retry_cleanup:
                    generation_debug["retry_cleanup"] = retry_cleanup
            retry_adjustments: list[dict[str, Any]] = []
            if solve_report:
                retry, retry_adjustments = normalize_solve_output(retry, solve_report)
                generation_debug["solve_contract"].update(
                    {
                        "retry_candidate_raw": generation_candidate_debug(retry_raw),
                        "retry_candidate_normalized": generation_candidate_debug(retry),
                        "retry_adjustments": retry_adjustments,
                    }
                )
            if solve_report:
                selector = lambda first, second: select_best_solve_output(
                    first,
                    second,
                    contract,
                    solve_report,
                    prepared.sources,
                )
            else:
                selector = lambda first, second: select_best_response_output(
                    first,
                    second,
                    contract,
                )
            selected = _select_complete_candidate(
                answer,
                retry,
                first_incomplete=first_incomplete,
                retry_incomplete=retry_incomplete,
                selector=selector,
            )
            generation_debug["retry_used"] = True
            generation_debug["retry_succeeded"] = retry_available
            generation_debug["retry_endpoint"] = "chat" if direct_retry else "generate"
            generation_debug["retry_done_reason"] = retry_done_reason
            generation_debug["candidate_penalties"] = {
                "first": list(response_output_penalty(answer, contract)),
                "retry": list(response_output_penalty(retry, contract)),
            }
            selected_is_first = selected == answer
            generation_debug["selected_candidate"] = "first" if selected_is_first else "retry"
            generation_debug["candidate_language"] = {
                "first": response_language_debug(
                    answer,
                    contract.language,
                    allow_source_language_fields=contract.allow_source_language_fields,
                ),
                "retry": response_language_debug(
                    retry,
                    contract.language,
                    allow_source_language_fields=contract.allow_source_language_fields,
                ),
            }
            answer = selected
            selected_done_reason = (
                initial_done_reason if selected_is_first else retry_done_reason
            )
        else:
            generation_debug["retry_used"] = False
            if solve_report:
                generation_debug["selected_candidate"] = "first"
        if solve_report:
            solve_debug = generation_debug["solve_contract"]
            solve_debug["selected_candidate_output"] = generation_candidate_debug(answer)
            answer, final_adjustments = normalize_solve_output(answer, solve_report)
            solve_debug["final_adjustments"] = final_adjustments
            solve_debug["final_normalized_output"] = generation_candidate_debug(answer)
        final_solve_issues = (
            solve_output_issues(answer, solve_report, prepared.sources)
            if solve_report
            else []
        )
        final_issues = response_output_issues(answer, contract) + final_solve_issues
        if selected_done_reason == "length":
            final_issues.append(INCOMPLETE_GENERATION_ISSUE)
        if solve_report:
            generation_debug["solve_contract"]["final_issues"] = final_solve_issues
        if contract.forbid_solution and any("reveals or narrows" in issue for issue in final_issues):
            answer = (
                "Hãy đối chiếu từng câu với đúng đoạn liên quan, xác định từ khóa và điều kiện trong "
                "hướng dẫn, nhưng chưa chọn hoặc loại trừ bất kỳ đáp án nào."
            )
            generation_debug["safe_fallback_used"] = True
            final_issues = response_output_issues(answer, contract)
        generation_debug["final_issues"] = final_issues
        if final_solve_issues:
            generation_debug["validation_degraded"] = True
        elif final_issues:
            generation_debug["validation_degraded"] = True
        failure = hard_validation_failure(
            prepared.query_intent,
            contract.language,
            final_issues,
        )
        if failure:
            generation_debug["validation_failed_closed"] = True
            generation_debug["returned_validation_fallback"] = failure
            return failure
        if solve_report and final_solve_issues:
            failure = solve_validation_failure(message)
            generation_debug["solve_validation_failed_closed"] = True
            generation_debug["returned_validation_fallback"] = failure
            generation_debug["solve_contract"]["returned_output"] = generation_candidate_debug(
                failure
            )
            return failure
        if solve_report:
            generation_debug["solve_contract"]["returned_output"] = generation_candidate_debug(
                answer
            )
    if (
        direct_retry
        and selected_done_reason == "length"
        and not generation_debug.get("retry_succeeded", True)
    ):
        generation_debug["returned_incomplete_fallback"] = True
        return ""
    return answer


def requires_reviewed_generation(prepared: "ChatPreparation", message: str) -> bool:
    return (
        prepared.route_used == "base_model"
        or is_writing_response(prepared)
        or prepared.query_intent == "solve_questions"
        or prepared.query_intent == "translate_questions"
        or prepared.query_intent == "translate_content"
        or prepared.query_intent == "document_overview"
        or (
            has_explicit_no_solution_constraint(message)
            and not prepared.debug.get("intent_decision", {}).get("allow_solution", False)
        )
    )


def is_writing_response(
    prepared: "ChatPreparation",
    message: str = "",
) -> bool:
    return prepared.query_intent == "writing_generation" or bool(
        message
        and prepared.route_used == "base_model"
        and prepared.query_intent == "direct"
        and is_direct_writing_request(message)
    )


def response_buffer_reason(prepared: "ChatPreparation", message: str) -> str:
    if prepared.static_response is not None:
        return "static_response"
    if is_writing_response(prepared, message):
        return "writing_contract"
    if prepared.query_intent == "solve_questions":
        return "solve_contract"
    if prepared.query_intent in {"translate_questions", "translate_content"}:
        return "translation_contract"
    if prepared.query_intent == "document_overview":
        return "overview_contract"
    if has_explicit_no_solution_constraint(message):
        return "no_solution_contract"
    if prepared.route_used == "base_model":
        return "direct_output_contract"
    return "response_contract"


def response_chunks(text: str, size: int = 16) -> list[str]:
    if not text:
        return [""]
    chunks: list[str] = []
    current = ""
    for part in re.findall(r"\S+\s*|\s+", text):
        current += part
        visible = current.rstrip()
        if (
            len(current) >= size
            or "\n\n" in current
            or visible.endswith((".", "!", "?", ":", ";"))
        ):
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def response_chunk_delay(chunk: str) -> float:
    delay = min(0.12, max(0.035, len(chunk) / 190))
    visible = chunk.rstrip()
    if visible.endswith((".", "!", "?")):
        delay += 0.035
    elif visible.endswith((":", ";")) or "\n\n" in chunk:
        delay += 0.02
    return delay


async def buffered_response_chunks(text: str) -> AsyncIterator[str]:
    chunks = response_chunks(text)
    for index, chunk in enumerate(chunks):
        yield chunk
        if index + 1 < len(chunks):
            await asyncio.sleep(response_chunk_delay(chunk))


def user_profile_context(req: ChatRequest, max_chars: int = 1_200) -> str:
    if not req.conversation_state or not req.conversation_state.user_facts:
        return ""
    lines: list[str] = []
    length = 0
    for fact in reversed(req.conversation_state.user_facts):
        line = f"- {fact.key}: {fact.value}"
        if lines and length + len(line) + 1 > max_chars:
            break
        lines.append(line)
        length += len(line) + 1
    return "\n".join(reversed(lines))


def merge_user_facts(
    existing: list[ChatUserFact],
    updates: list[ChatUserFact],
    limit: int = 12,
) -> list[ChatUserFact]:
    merged = {fact.key: fact for fact in existing}
    order = [fact.key for fact in existing]
    for fact in updates:
        if fact.key in order:
            order.remove(fact.key)
        order.append(fact.key)
        merged[fact.key] = fact
    return [merged[key] for key in order[-limit:]]


async def collect_user_fact_updates(
    req: ChatRequest,
    prepared: "ChatPreparation",
) -> None:
    trusted_result = prepared.route_used in {
        "base_model",
        "vector_rag",
        "vector_rag_static",
    }
    if not trusted_result:
        prepared.debug["user_fact_extraction"] = {
            "attempted": False,
            "reason": "untrusted_result",
            "facts": [],
        }
        return
    decision = await extract_user_facts(req.message)
    prepared.user_fact_updates = list(decision.facts)
    prepared.debug["user_fact_extraction"] = decision.to_debug()


def conversation_state_for_result(
    req: ChatRequest,
    prepared: "ChatPreparation",
) -> ChatConversationState:
    previous_state = req.conversation_state or ChatConversationState()
    previous_affinity = previous_state.rag_affinity
    trusted_result = prepared.route_used in {
        "base_model",
        "vector_rag",
        "vector_rag_static",
    }
    if prepared.route_used == "base_model":
        route = "direct"
        affinity = previous_affinity
    elif not trusted_result:
        route = previous_state.last_route
        affinity = previous_affinity
    else:
        route = "rag"
        document_ids = list(
            dict.fromkeys(
                str(source.get("document_id"))
                for source in prepared.sources
                if source.get("document_id")
            )
        )
        passage_numbers = sorted(
            {
                int(source.get("metadata", {}).get("passage_number"))
                for source in prepared.sources
                if source.get("metadata", {}).get("passage_number") is not None
            }
        )
        question_ranges = []
        for source in prepared.sources:
            values = source.get("metadata", {}).get("question_range")
            if isinstance(values, list) and len(values) == 2 and values not in question_ranges:
                question_ranges.append([int(values[0]), int(values[1])])
        affinity = ChatAffinity(
            document_ids=document_ids or previous_affinity.document_ids,
            passage_numbers=passage_numbers,
            question_ranges=question_ranges,
        )
    state = ChatConversationState(
        last_route=route,
        last_intent=(
            prepared.query_intent if trusted_result else previous_state.last_intent
        ),
        user_facts=(
            merge_user_facts(previous_state.user_facts, prepared.user_fact_updates)
            if trusted_result
            else previous_state.user_facts
        ),
        rag_affinity=affinity,
    )
    prepared.debug["conversation_state"] = {
        "input": req.conversation_state.model_dump() if req.conversation_state else None,
        "result": {
            "route_used": prepared.route_used,
            "query_intent": prepared.query_intent,
            "trusted": trusted_result,
        },
        "output": state.model_dump(),
    }
    return state


def backend_session_request(req: ChatRequest) -> ChatRequest:
    payload = get_store_manager().read_session_memory(req.session_id)
    try:
        messages = [
            ChatMessage.model_validate(message)
            for message in payload.get("messages", [])
        ]
        raw_state = payload.get("conversation_state")
        state = ChatConversationState.model_validate(raw_state) if raw_state else None
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Session memory is invalid.") from exc
    return req.model_copy(
        update={
            "conversation_history": messages or None,
            "conversation_state": state,
        }
    )


def persist_session_turn(
    req: ChatRequest,
    prepared: "ChatPreparation",
    assistant_answer: str,
) -> None:
    if not assistant_answer.strip():
        return
    history = list(req.conversation_history or [])
    history.extend(
        [
            ChatMessage(role="user", content=req.message),
            ChatMessage(role="assistant", content=assistant_answer[:20_000]),
        ]
    )
    history = history[-20:]
    state = conversation_state_for_result(req, prepared)
    get_store_manager().write_session_memory(
        req.session_id,
        [message.model_dump() for message in history],
        state.model_dump(),
    )
    prepared.debug["session_memory"] = {
        "source": "backend",
        "messages": len(history),
        "user_facts": len(state.user_facts),
        "rag_document_affinity": len(state.rag_affinity.document_ids),
    }


def _markdown_table(table: dict[str, Any]) -> str:
    columns = table.get("columns") or []
    rows = table.get("rows") or []
    if not columns or not rows:
        return ""
    header = "| " + " | ".join(str(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = list(row) + [""] * max(0, len(columns) - len(row))
        body.append("| " + " | ".join(str(cell) for cell in cells[: len(columns)]) + " |")
    return "\n".join([header, separator, *body])


def _visual_incomplete_text(visual: dict[str, Any], source: dict[str, Any]) -> str:
    visual_type = visual.get("type", "visual")
    question_range = visual.get("question_range") or []
    range_label = f" Questions {question_range[0]}-{question_range[1]}" if len(question_range) == 2 else ""
    blanks = ", ".join(str(number) for number in visual.get("blank_question_numbers") or [])
    raw_text = visual.get("raw_text") or ""
    return (
        f"Mình đã nhận diện được {visual_type}{range_label}, nhưng chưa trích xuất đủ cấu trúc hàng/cột hoặc node/edge đáng tin cậy.\n\n"
        + (f"Các ô/câu trống nhận diện được: {blanks}.\n\n" if blanks else "")
        + (f"Nội dung OCR/native liên quan:\n{raw_text}\n\n" if raw_text else "")
        + f"Nguồn: {_source_label(source)}."
    )


def _source_label(source: dict[str, Any]) -> str:
    source_file = source.get("source_file", "unknown")
    pages = source.get("pages") or []
    if not pages:
        return source_file
    return f"{source_file}, trang {', '.join(str(page) for page in pages)}"


def _table_from_source(source: dict[str, Any]) -> dict[str, Any] | None:
    metadata = source.get("metadata", {})
    table = metadata.get("table")
    if isinstance(table, dict):
        return table
    return None


def _render_show_questions(sources: list[dict[str, Any]]) -> str | None:
    question_groups = [
        source
        for source in sources
        if source.get("metadata", {}).get("unit_type") == "question_group"
    ]
    if question_groups:
        lines = []
        for source in question_groups:
            text = (source.get("display_text") or source.get("text") or "").strip()
            if not text:
                continue
            lines.append(text)
            lines.append(f"Nguồn: {_source_label(source)}.")
        return "\n\n".join(lines).strip() or None

    questions = [
        source
        for source in sources
        if source.get("metadata", {}).get("unit_type") == "question"
    ]
    if not questions:
        return None
    questions = sorted(questions, key=lambda source: source.get("metadata", {}).get("question_start") or 999)
    lines = ["Nội dung câu hỏi:"]
    for source in questions:
        text = (source.get("display_text") or source.get("text") or "").strip()
        if text:
            lines.append(f"- {text}")
    lines.append(f"\nNguồn: {_source_label(questions[0])}.")
    return "\n".join(lines)


def _lookup_table_cell(message: str, sources: list[dict[str, Any]]) -> str | None:
    best_match: tuple[float, Any, dict[str, Any]] | None = None
    for source in sources:
        table = _table_from_source(source)
        metadata = source.get("metadata", {})
        if table:
            columns = table.get("columns") or []
            rows = table.get("rows") or []
        else:
            columns = metadata.get("table_columns") or []
            row = metadata.get("table_row")
            rows = [row] if isinstance(row, list) else []
        match = table_cell_value(message, {"columns": columns, "rows": rows})
        if match and (best_match is None or match[0] > best_match[0]):
            best_match = (match[0], match[1], source)
    if best_match is None:
        return None
    _, value, source = best_match
    return f"{value}\n\nNguồn: {_source_label(source)}."


def _full_table_source(sources: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    candidates = []
    for source in sources:
        table = _table_from_source(source)
        if not table or not table.get("columns") or not table.get("rows"):
            continue
        candidates.append((len(table.get("rows") or []), table, source))
    if not candidates:
        return None
    _, table, source = max(candidates, key=lambda item: item[0])
    return table, source


def _render_table_calculation(message: str, sources: list[dict[str, Any]]) -> str | None:
    selected = _full_table_source(sources)
    if not selected:
        return None
    table, source = selected
    result = table_change_calculations(message, table)
    if not result:
        return None
    lines = [
        f"- {item['label']}: {format_number(item['second'])} - {format_number(item['first'])} = {format_number(item['change'])}"
        for item in result["calculations"]
    ]
    direction = "giảm" if result["direction"] == "decrease" else "tăng"
    winner = result["winner"]
    lines.append(
        f"\n{winner['label']} có mức {direction} lớn nhất: {format_number(winner['change'])}."
    )
    lines.append(f"\nNguồn: {_source_label(source)}.")
    return "\n".join(lines)


def _render_table_comparison(message: str, sources: list[dict[str, Any]]) -> str | None:
    selected = _full_table_source(sources)
    if not selected:
        return None
    table, source = selected
    row = comparison_row(message, table)
    if not row:
        return None
    markdown = _markdown_table({"columns": table.get("columns") or [], "rows": [row]})
    if not markdown:
        return None
    facts = comparison_row_facts(table, row)
    comparison = "\n".join(f"- {fact}" for fact in facts)
    if comparison:
        return f"{markdown}\n\n{comparison}\n\nNguồn: {_source_label(source)}."
    return f"{markdown}\n\nNguồn: {_source_label(source)}."


def _render_writing_prompt(sources: list[dict[str, Any]]) -> str | None:
    for source in sources:
        if source.get("metadata", {}).get("unit_type") not in {"writing_prompt", "writing_task"}:
            continue
        text = (source.get("display_text") or source.get("text") or "").strip()
        if text:
            return f"{text}\n\nNguồn: {_source_label(source)}."
    return None


def _render_writing_inventory(sources: list[dict[str, Any]]) -> str | None:
    tasks = [source for source in sources if source.get("metadata", {}).get("unit_type") == "writing_task"]
    if not tasks:
        return None
    answer_keys = {
        str(source.get("metadata", {}).get("section_id", "")).removesuffix("-answer")
        for source in sources
        if source.get("metadata", {}).get("unit_type") == "sample_answer"
    }
    lines = ["Các đề và bài mẫu trong tài liệu:"]
    for source in sorted(tasks, key=lambda item: (min(item.get("pages") or [999]), item.get("chunk_index", 0))):
        text = (source.get("display_text") or source.get("text") or "").strip()
        title = next((line.strip() for line in text.splitlines() if line.strip()), "Writing task")
        section_key = str(source.get("metadata", {}).get("section_id", "")).removesuffix("-task")
        sample_label = "có bài mẫu" if section_key in answer_keys else "chưa thấy bài mẫu"
        lines.append(f"- Trang {', '.join(str(page) for page in source.get('pages') or [])}: {title} ({sample_label})")
    lines.append(f"\nNguồn: {_source_label(tasks[0])}.")
    return "\n".join(lines)


def solve_context_report(
    message: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    requested = requested_question_numbers(message)
    exact_sources = exact_question_sources(message, sources)
    question_targets = solve_question_packets(message, sources)
    found_numbers = [target["question_number"] for target in question_targets]
    ambiguous_questions: list[int] = []
    missing_groups = [
        target["question_number"]
        for target in question_targets
        if "missing_question_group" in target["warnings"]
    ]
    ambiguous_groups = [
        target["question_number"]
        for target in question_targets
        if "ambiguous_question_group" in target["warnings"]
    ]
    passage_mismatches = [
        target["question_number"]
        for target in question_targets
        if "question_group_passage_mismatch" in target["warnings"]
    ]
    missing_passage_links = [
        target["question_number"]
        for target in question_targets
        if "missing_passage_link" in target["warnings"]
    ]
    missing_evidence = [
        target["question_number"]
        for target in question_targets
        if not target["evidence_chunk_ids"]
    ]
    missing_options = [
        target["question_number"]
        for target in question_targets
        if "missing_answer_options" in target["warnings"]
    ]

    for number in requested:
        matches = [
            source
            for source in exact_sources
            if source.get("metadata", {}).get("question_range") == [number, number]
        ]
        if len(dedupe_sources(matches)) > 1:
            ambiguous_questions.append(number)

    exact_numbers = {
        source.get("metadata", {}).get("question_range", [None])[0]
        for source in exact_sources
    }
    missing_numbers = [number for number in requested if number not in exact_numbers]
    issues: list[str] = []
    if requested and missing_numbers:
        issues.append("missing_exact_questions")
    if ambiguous_questions:
        issues.append("ambiguous_exact_questions")
    if missing_groups:
        issues.append("missing_question_groups")
    if ambiguous_groups:
        issues.append("ambiguous_question_groups")
    if passage_mismatches:
        issues.append("question_group_passage_mismatch")
    if missing_passage_links:
        issues.append("missing_passage_links")
    if missing_evidence:
        issues.append("missing_passage_evidence")
    if missing_options:
        issues.append("missing_answer_options")
    return {
        "requested_question_numbers": requested,
        "found_question_numbers": found_numbers,
        "missing_question_numbers": missing_numbers,
        "ambiguous_question_numbers": ambiguous_questions,
        "missing_group_question_numbers": missing_groups,
        "ambiguous_group_question_numbers": ambiguous_groups,
        "passage_mismatch_question_numbers": passage_mismatches,
        "missing_passage_link_question_numbers": missing_passage_links,
        "missing_evidence_question_numbers": missing_evidence,
        "missing_option_question_numbers": missing_options,
        "question_targets": question_targets,
        "issues": issues,
    }


def _range_contains(value: Any, number: int) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    return int(value[0]) <= number <= int(value[1])


def solve_context_issue(
    sources: list[dict[str, Any]],
    message: str | None = None,
) -> str | None:
    question_text = "\n".join(
        (source.get("display_text") or source.get("text") or "").strip()
        for source in sources
        if source.get("metadata", {}).get("unit_type") in {"question", "question_group"}
    )
    if not question_text:
        return "missing_question"
    if message:
        report = solve_context_report(message, sources)
        if report["issues"]:
            return report["issues"][0]
    requires_options = _question_requires_options(question_text)
    if requires_options and len(_answer_option_labels(question_text)) < 2:
        return "missing_answer_options"
    return None


def writing_table_facts(sources: list[dict[str, Any]]) -> list[str]:
    selected = _full_table_source(sources)
    return table_summary_facts(selected[0]) if selected else []


def static_response_for_sources(message: str, query_intent: str, sources: list[dict[str, Any]]) -> str | None:
    if query_intent == "table_cell":
        return _lookup_table_cell(message, sources)
    if query_intent == "table_calculation":
        return _render_table_calculation(message, sources)
    if query_intent == "table_comparison":
        return _render_table_comparison(message, sources)
    if query_intent == "show_writing_prompt":
        return _render_writing_prompt(sources)
    if query_intent == "document_overview":
        inventory = _render_writing_inventory(sources)
        has_non_writing_sections = any(
            source.get("metadata", {}).get("unit_type")
            in {"document_outline", "passage", "passage_summary", "question_group"}
            for source in sources
        )
        if inventory and not has_non_writing_sections:
            return inventory

    if query_intent == "show_questions":
        questions = _render_show_questions(sources)
        if questions:
            return questions

    if query_intent in {"show_table", "extract_table"}:
        for source in sources:
            table = _table_from_source(source)
            table_markdown = _markdown_table(table) if table else ""
            if table_markdown:
                return f"Dưới đây là bảng mình trích xuất được từ tài liệu:\n\n{table_markdown}\n\nNguồn: {_source_label(source)}."
            if table:
                return _visual_incomplete_text(table, source)
        return (
            "Mình chưa có dữ liệu bảng đã được trích xuất theo cấu trúc cho phần này. "
            "Để tránh tự dựng sai hàng/cột hoặc ô trống, mình chưa hiển thị bảng."
        )

    if query_intent == "show_flowchart":
        for source in sources:
            metadata = source.get("metadata", {})
            flowchart = metadata.get("flowchart")
            if isinstance(flowchart, dict):
                nodes = flowchart.get("nodes") or []
                edges = flowchart.get("edges") or []
                if not nodes or not edges:
                    return _visual_incomplete_text(flowchart, source)
                lines = ["Mình tìm thấy cấu trúc flowchart:"]
                for node in nodes:
                    label = f"Question {node['question_number']} blank" if node.get("question_number") else node.get("text", "")
                    lines.append(f"- {node['id']}: {label}")
                for edge in edges:
                    lines.append(f"- edge: {edge['from']} -> {edge['to']}")
                lines.append(f"\nNguồn: {_source_label(source)}.")
                return "\n".join(lines)
        return (
            "Mình chưa có dữ liệu flowchart đã được trích xuất theo node/edge cho phần này. "
            "Để tránh tự tưởng tượng cấu trúc, mình chưa mô tả flowchart."
        )

    if query_intent == "show_diagram":
        for source in sources:
            diagram = source.get("metadata", {}).get("diagram")
            if not isinstance(diagram, dict):
                continue
            nodes = diagram.get("nodes") or []
            edges = diagram.get("edges") or []
            if not nodes or not edges:
                return _visual_incomplete_text(diagram, source)
            lines = ["Mình tìm thấy cấu trúc diagram:"]
            for node in nodes:
                label = f"Question {node['question_number']} blank" if node.get("question_number") else node.get("text", "")
                lines.append(f"- {node['id']}: {label}")
            for edge in edges:
                lines.append(f"- edge: {edge['from']} -> {edge['to']}")
            lines.append(f"\nNguồn: {_source_label(source)}.")
            return "\n".join(lines)
        return (
            "Mình chưa có dữ liệu diagram đã được trích xuất theo cấu trúc cho phần này. "
            "Để tránh tự tưởng tượng nhãn hoặc quan hệ, mình chưa mô tả diagram."
        )

    return None


def is_presence_check_query(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in ["có nhắc đến", "có nói về", "có đề cập", "mentions", "mention"])


def has_lexical_source_hit(sources: list[dict[str, Any]]) -> bool:
    for source in sources:
        if source.get("probe_keyword_score", 0.0) > 0 or source.get("keyword_score", 0.0) > 0:
            return True
        if source.get("probe_question_score", 0.0) > 0 or source.get("question_score", 0.0) > 0:
            return True
        if source.get("probe_overview_score", 0.0) > 0 or source.get("overview_score", 0.0) > 0:
            return True
    return False


def compact_final_context_debug(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": source.get("chunk_id"),
            "document_id": source.get("document_id"),
            "source_file": source.get("source_file"),
            "method": source.get("retrieval_method"),
            "unit_type": source.get("metadata", {}).get("unit_type"),
            "passage_number": source.get("metadata", {}).get("passage_number"),
            "question_range": source.get("metadata", {}).get("question_range"),
            "parent_id": source.get("metadata", {}).get("parent_id"),
            "pages": source.get("pages"),
        }
        for source in sources
    ]


@dataclass
class ChatPreparation:
    prompt: str | None
    static_response: str | None
    route_used: str
    sources: list[dict[str, Any]]
    debug: dict[str, Any]
    query_intent: str = "direct"
    user_fact_updates: list[ChatUserFact] = field(default_factory=list)


def affinity_retrieval_query(req: ChatRequest, use_affinity_context: bool) -> str:
    message = req.message.strip()
    if not use_affinity_context or not req.conversation_history:
        return message

    previous_user_message = next(
        (
            item.content.strip()
            for item in reversed(req.conversation_history)
            if item.role == "user" and item.content.strip()
        ),
        "",
    )
    if not previous_user_message:
        return message

    affinity_hints: list[str] = []
    affinity = conversation_affinity(req)
    if affinity:
        if affinity.passage_numbers:
            affinity_hints.append(
                "Passages: " + ", ".join(str(value) for value in affinity.passage_numbers)
            )
        if affinity.question_ranges:
            affinity_hints.append(
                "Question ranges: "
                + ", ".join(
                    f"{values[0]}-{values[1]}"
                    for values in affinity.question_ranges
                    if len(values) == 2
                )
            )
    suffix = f"\n{' | '.join(affinity_hints)}" if affinity_hints else ""
    return f"{previous_user_message}\nFollow-up: {message}{suffix}"


def conversation_affinity(req: ChatRequest) -> ChatAffinity | None:
    if req.conversation_state and req.conversation_state.rag_affinity.document_ids:
        return req.conversation_state.rag_affinity
    return None


QUESTION_TARGET_INTENTS = {
    "show_questions",
    "translate_questions",
    "explain_questions",
    "solve_questions",
}
STRUCTURED_TABLE_OPERATION_INTENTS = {
    "table_cell",
    "table_calculation",
    "table_comparison",
}
TABLE_UNIT_TYPES = {"table", "table_row", "writing_table"}


def allowed_rag_intents(
    message: str,
    catalog: list[dict[str, Any]],
    affinity: ChatAffinity | None,
) -> tuple[str, ...]:
    """Remove intents that require structured targets absent from this turn."""
    has_question_target = bool(parse_question_ranges(message)) or bool(
        affinity and affinity.question_ranges
    )
    unit_types = {
        str(unit_type)
        for item in catalog
        for unit_type in item.get("unit_types", [])
    }
    has_structured_table = bool(unit_types & TABLE_UNIT_TYPES)

    excluded: set[str] = set()
    if not has_question_target:
        excluded.update(QUESTION_TARGET_INTENTS)
    if not has_structured_table:
        excluded.update(STRUCTURED_TABLE_OPERATION_INTENTS)
    return tuple(intent for intent in ROUTING_INTENTS if intent not in excluded)


def gateway_state_context(req: ChatRequest) -> str:
    if not req.conversation_state:
        return ""
    previous_answer_source = {
        "rag": "uploaded_material",
        "direct": "conversation",
    }.get(req.conversation_state.last_route, "none")
    return json.dumps(
        {
            "previous_answer_source": previous_answer_source,
            "previous_answer_intent": req.conversation_state.last_intent,
        },
        ensure_ascii=False,
    )


def direct_conversation_source(req: ChatRequest) -> str:
    if not req.conversation_state or req.conversation_state.last_route != "direct":
        return "none"
    if not req.conversation_history or not any(
        message.role == "assistant" for message in req.conversation_history
    ):
        return "none"
    return "conversation"


async def direct_reviewed_generation_fallback(
    prepared: ChatPreparation,
    req: ChatRequest,
    reason: str,
    previous_answer_source: str,
    direct_source_available: bool = True,
) -> str:
    generation_debug = prepared.debug.setdefault("direct_generation", {})
    output_contract: list[str] | None = None
    if is_direct_writing_request(req.message) and direct_source_available:
        output_contract = [
            "- The requested task source is available. Apply the Writing constraints below."
        ] + writing_output_contract(req.message).prompt_lines()
    generation_debug.update(
        {
            "fallback_used": True,
            "fallback_reason": reason,
            "fallback_endpoint": "generate",
        }
    )
    fallback_response_debug = generation_debug.setdefault("fallback_response", {})
    try:
        answer = await query_ollama(
            direct_answer_prompt(
                req.message,
                req.conversation_history,
                user_profile_context(req),
                previous_answer_source=previous_answer_source,
                output_contract=output_contract,
            ),
            temperature=generation_temperature(prepared),
            max_attempts=1,
        )
    except Exception as exc:
        if isinstance(exc, OllamaRequestError) and exc.kind in {
            "empty_response",
            "prompt_echo",
            "role_continuation",
        }:
            generation_debug.update(
                {
                    "fallback_status": "exhausted",
                    "fallback_error": exc.kind,
                }
            )
            return ""
        generation_debug.update(
            {
                "fallback_status": "failed",
                "fallback_error": (
                    exc.kind if isinstance(exc, OllamaRequestError) else type(exc).__name__
                ),
            }
        )
        raise
    fallback_response_debug.update(
        {
            "response_length": len(answer),
            "num_predict": OLLAMA_NUM_PREDICT,
        }
    )
    generation_debug["fallback_status"] = "succeeded" if answer.strip() else "empty"
    return answer


def resolve_target_refs(
    references: tuple[str, ...],
    routing_context: DocumentCatalogContext,
    allowed_document_ids: list[str],
) -> tuple[list[str], list[str]]:
    allowed = set(allowed_document_ids)
    resolved: list[str] = []
    invalid: list[str] = []
    for reference in references:
        document_id = routing_context.document_refs.get(reference)
        if not document_id or document_id not in allowed:
            invalid.append(reference)
        elif document_id not in resolved:
            resolved.append(document_id)
    return resolved, invalid


def gateway_clarification_response(
    catalog: list[dict[str, Any]],
    candidate_document_ids: list[str] | None = None,
    *,
    no_match_response: str = NO_RAG_MATCH_RESPONSE,
) -> str:
    by_document_id = {
        str(document_id): item.get("source_file", "unknown")
        for item in catalog
        for document_id in item.get("document_ids") or []
    }
    files = [
        by_document_id[document_id]
        for document_id in candidate_document_ids or []
        if document_id in by_document_id
    ]
    if not files:
        files = [item.get("source_file", "unknown") for item in catalog]
    files = list(dict.fromkeys(files))[: settings.target_clarification_max_candidates]
    if not files:
        return no_match_response
    choices = "\n".join(f"- {name}" for name in files)
    return f"{AMBIGUOUS_DOCUMENT_RESPONSE}\n\nCác file phù hợp nhất:\n{choices}"


async def prepare_chat(req: ChatRequest) -> ChatPreparation:
    message = req.message.strip()
    store = get_store(req.session_id)
    route = "direct"
    sources: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    probe: dict[str, Any] = {"results": []}
    query_intent = "direct"
    intent_debug: dict[str, Any] = {}
    writing_parent_id: str | None = None
    evidence_query: str | None = None
    evidence_by_question: list[dict[str, Any]] = []
    evidence_per_question = 0
    solve_report: dict[str, Any] = {}
    gateway_debug: dict[str, Any]
    document_resolution_debug: dict[str, Any] = {}

    full_catalog = await run_in_threadpool(store.document_catalog)
    scope = resolve_document_scope(
        message,
        full_catalog,
        req.document_ids,
        req.document_scope,
    )
    affinity = conversation_affinity(req)
    allowed_scope_ids = scope.allowed_document_ids
    catalog = [
        item
        for item in full_catalog
        if any(document_id in allowed_scope_ids for document_id in item.get("document_ids", []))
    ]
    attached_document_ids = (
        req.document_ids
        if req.document_scope == "explicit" and req.document_ids
        else []
    )
    catalog_reference_match = bool(
        scope.document_grounded and scope.resolved_document_ids
    )
    route_environment_context = format_route_environment_context(
        full_catalog,
        attached_document_ids,
        catalog_reference_match=catalog_reference_match,
    )
    gateway_decision = await classify_chat_route(
        message,
        req.conversation_history,
        gateway_state_context(req),
        route_environment_context,
    )
    route = gateway_decision.route
    gateway_debug = {
        "used": True,
        **gateway_decision.to_debug(),
        "catalog_context": {
            "mode": "environment_with_reference_signal",
            "available_documents": len(full_catalog),
            "included_documents": 0,
            "omitted_documents": len(full_catalog),
            "attached_document_ids": attached_document_ids,
            "catalog_reference_match": catalog_reference_match,
        },
    }
    if catalog_reference_match and route != "rag":
        gateway_debug["classifier_route"] = route
        gateway_debug["route"] = "rag"
        gateway_debug["policy_override"] = "catalog_reference_match"
        route = "rag"

    if route == "direct":
        previous_answer_source = direct_conversation_source(req)
        return ChatPreparation(
            prompt=direct_answer_prompt(
                message,
                req.conversation_history,
                user_profile_context(req),
                previous_answer_source=previous_answer_source,
            ),
            static_response=None,
            route_used="base_model",
            sources=[],
            debug={
                "route_decision": "direct",
                "query_intent": "direct",
                "route_gateway": gateway_debug,
                "document_resolution": {"skipped": True, "reason": "direct_route"},
                "intent_classifier": {"skipped": True, "reason": "direct_route"},
                "direct_generation": {
                    "used": True,
                    "response_contract": "adaptive_direct_answer",
                    "primary_endpoint": "chat",
                    "fallback_endpoint": "generate",
                    "fallback_used": False,
                    "previous_answer_source": previous_answer_source,
                    "temperature": settings.ollama_direct_temperature,
                },
                "retrieval": {"skipped": True, "final_context": []},
                "conversation_state": req.conversation_state.model_dump() if req.conversation_state else None,
                "source_count": 0,
            },
            query_intent="direct",
        )

    has_safe_rag_fallback = bool(
        (scope.request_mode == "explicit" and scope.resolved_document_ids)
        or (scope.document_grounded and scope.resolved_document_ids)
    )
    if route == "undetermined" and not has_safe_rag_fallback:
        return ChatPreparation(
            prompt=None,
            static_response=ROUTE_UNDETERMINED_RESPONSE,
            route_used="route_undetermined",
            sources=[],
            debug={
                "route_decision": "undetermined",
                "query_intent": "route_undetermined",
                "route_gateway": gateway_debug,
                "document_resolution": {"skipped": True, "reason": "route_undetermined"},
                "intent_classifier": {"skipped": True, "reason": "route_undetermined"},
                "retrieval": {"skipped": True, "final_context": []},
                "conversation_state": req.conversation_state.model_dump() if req.conversation_state else None,
                "source_count": 0,
            },
            query_intent="route_undetermined",
        )
    if route == "undetermined":
        route = "rag"
        gateway_debug["fallback_reason"] = "valid_document_scope_or_rag_affinity"

    scope_ids = list(scope.resolved_document_ids)
    needs_target_model = not scope_ids and len(allowed_scope_ids) > 1
    ranked_candidates = (
        rank_document_candidates(
            message,
            catalog,
            settings.target_resolver_max_candidates,
            affinity.document_ids if affinity else None,
        )
        if needs_target_model
        else []
    )
    evidence_candidates = [
        candidate for candidate in ranked_candidates if candidate.matched_fields
    ]
    deterministic_candidate = (
        evidence_candidates[0] if len(evidence_candidates) == 1 else None
    )
    resolver_candidates = evidence_candidates or ranked_candidates
    resolver_catalog = (
        [candidate.entry for candidate in resolver_candidates]
        if resolver_candidates
        else catalog
    )
    gateway_context = format_document_catalog_context(resolver_catalog, message)
    affinity_document_refs = tuple(
        reference
        for reference, document_id in gateway_context.document_refs.items()
        if affinity and document_id in affinity.document_ids
    )
    has_target_evidence = bool(evidence_candidates or affinity_document_refs)
    target_decision = (
        await resolve_rag_target(
            message,
            gateway_context.text,
            req.conversation_history,
            affinity_document_refs,
        )
        if needs_target_model and deterministic_candidate is None and has_target_evidence
        else None
    )
    use_affinity_context = False
    clarification_document_ids: list[str] = []

    if not scope_ids and len(allowed_scope_ids) == 1:
        scope_ids = list(allowed_scope_ids)
        document_resolution_debug = {"method": "single_allowed_document"}
    elif scope_ids:
        document_resolution_debug = {"method": scope.method}
    elif deterministic_candidate is not None:
        scope_ids = [
            document_id
            for document_id in deterministic_candidate.entry.get("document_ids", [])
            if document_id in allowed_scope_ids
        ]
        document_resolution_debug = {
            "method": "unique_metadata_candidate",
            "matched_fields": list(deterministic_candidate.matched_fields),
            "score": round(deterministic_candidate.score, 3),
        }
    elif target_decision and target_decision.action == "all":
        scope_ids = list(allowed_scope_ids)
        document_resolution_debug = {"method": "semantic_target_all", **target_decision.to_debug()}
    elif target_decision and target_decision.action == "selected":
        scope_ids, invalid_refs = resolve_target_refs(
            target_decision.document_refs,
            gateway_context,
            allowed_scope_ids,
        )
        document_resolution_debug = {
            "method": "semantic_target",
            **target_decision.to_debug(),
            "invalid_refs": invalid_refs,
            "affinity_candidate_refs": list(affinity_document_refs),
        }
        use_affinity_context = bool(
            set(target_decision.document_refs).intersection(affinity_document_refs)
        )
        if use_affinity_context:
            document_resolution_debug["method"] = "semantic_target_with_affinity"
    else:
        if target_decision:
            clarification_document_ids, invalid_candidate_refs = resolve_target_refs(
                target_decision.candidate_refs,
                gateway_context,
                allowed_scope_ids,
            )
        else:
            invalid_candidate_refs = []
        if clarification_document_ids:
            clarification_set = set(clarification_document_ids)
            clarification_document_ids = [
                document_id
                for document_id in gateway_context.included_document_ids
                if document_id in clarification_set
            ]
        if not clarification_document_ids:
            clarification_document_ids = list(gateway_context.included_document_ids)[
                : settings.target_clarification_max_candidates
            ]
        document_resolution_debug = {
            "method": (
                "semantic_target_clarify"
                if has_target_evidence
                else "no_target_evidence"
            ),
            **(target_decision.to_debug() if target_decision else {}),
            "candidate_document_ids": clarification_document_ids,
            "invalid_candidate_refs": invalid_candidate_refs,
        }
    document_resolution_debug["catalog_context"] = {
        "order": "query_relevance_then_recent_ingestion_tie_break",
        "available_document_ids": list(allowed_scope_ids),
        "included_document_ids": list(gateway_context.included_document_ids),
        "omitted_document_ids": [
            document_id
            for document_id in allowed_scope_ids
            if document_id not in gateway_context.included_document_ids
        ],
        "ranked_candidates": [
            candidate.to_debug() for candidate in ranked_candidates
        ],
        "resolver_candidates": [
            candidate.to_debug() for candidate in resolver_candidates
        ],
    }

    if not scope_ids:
        debug = {
            "route_decision": "rag",
            "query_intent": "ambiguous_document" if len(allowed_scope_ids) > 1 else "document_no_match",
            "route_gateway": gateway_debug,
            "document_resolution": document_resolution_debug,
            "intent_classifier": {"skipped": True, "reason": "document_scope_unresolved"},
            "target_resolution": scope.to_debug(),
            "catalog": full_catalog,
            "probe": compact_probe_debug(probe),
            "retrieval": {
                "method": None,
                "structured_hits": 0,
                "before_filter_count": 0,
                "after_filter_count": 0,
                "final_context": [],
            },
            "source_count": 0,
        }
        return ChatPreparation(
            prompt=None,
            static_response=(
                gateway_clarification_response(
                    catalog,
                    clarification_document_ids,
                    no_match_response=no_rag_match_response(message),
                )
                if len(allowed_scope_ids) > 1
                else no_rag_match_response(message)
            ),
            route_used=(
                "vector_rag_ambiguous_document"
                if len(allowed_scope_ids) > 1
                else "vector_rag_no_match"
            ),
            sources=[],
            debug=debug,
            query_intent=("ambiguous_document" if len(allowed_scope_ids) > 1 else "document_no_match"),
        )

    scoped_catalog = [
        item
        for item in full_catalog
        if any(document_id in scope_ids for document_id in item.get("document_ids", []))
    ]
    intent_affinity = affinity if use_affinity_context else None
    candidate_intents = allowed_rag_intents(message, scoped_catalog, intent_affinity)
    intent_classifier = await classify_rag_intent(
        message,
        req.conversation_history,
        candidate_intents,
    )
    if intent_classifier.intent == "undetermined":
        classifier_debug = intent_classifier.to_debug()
        classifier_debug["allowed_intents"] = list(candidate_intents)
        classifier_debug["excluded_intents"] = [
            intent for intent in ROUTING_INTENTS if intent not in candidate_intents
        ]
        debug = {
            "route_decision": "rag",
            "query_intent": "intent_undetermined",
            "route_gateway": gateway_debug,
            "document_resolution": {
                **document_resolution_debug,
                "resolved_document_ids": scope_ids,
                "requested_scope": scope.to_debug(),
            },
            "intent_classifier": classifier_debug,
            "target_resolution": scope.to_debug(),
            "catalog": catalog,
            "probe": compact_probe_debug(probe),
            "retrieval": {"skipped": True, "final_context": []},
            "source_count": 0,
        }
        return ChatPreparation(
            prompt=None,
            static_response=INTENT_UNDETERMINED_RESPONSE,
            route_used="intent_undetermined",
            sources=[],
            debug=debug,
            query_intent="intent_undetermined",
        )

    intent_decision = semantic_intent_decision(
        message,
        intent_classifier.intent,
        1.0,
        "Semantic intent enum classifier.",
    )
    query_intent = intent_decision.intent
    intent_debug = intent_decision.to_debug()
    intent_debug["classifier"] = intent_classifier.to_debug()
    intent_debug["allowed_intents"] = list(candidate_intents)
    intent_debug["excluded_intents"] = [
        intent for intent in ROUTING_INTENTS if intent not in candidate_intents
    ]

    catalog = scoped_catalog

    retrieval_query = affinity_retrieval_query(req, use_affinity_context)
    probe_top_k = max(settings.rag_probe_top_k, settings.rag_top_k)

    if full_catalog and route == "rag" and query_intent == "semantic_qa":
        semantic_results = await run_in_threadpool(
            store.hybrid_search,
            retrieval_query,
            probe_top_k,
            scope_ids,
            None,
            None,
        )
        probe = {
            "results": semantic_results,
            "has_hits": bool(semantic_results),
            "has_strong_hits": bool(semantic_results),
            "has_document_intent": True,
            "is_overview": False,
            "top_score": semantic_results[0].get("score", 0.0) if semantic_results else 0.0,
            "top_fused_score": semantic_results[0].get("rrf_score", 0.0) if semantic_results else 0.0,
            "top_keyword_score": semantic_results[0].get("probe_keyword_score", 0.0) if semantic_results else 0.0,
            "top_question_score": 0.0,
            "top_overview_score": 0.0,
        }

    if route == "rag":
        evidence_candidate_count = 0
        evidence_context_count = 0
        solve_question_numbers = (
            requested_question_numbers(message)
            if query_intent == "solve_questions"
            else []
        )
        if query_intent == "document_overview":
            structured_top_k = 50
        elif query_intent == "solve_questions" and solve_question_numbers:
            structured_top_k = min(
                50,
                max(
                    settings.rag_top_k,
                    len(solve_question_numbers) + len(parse_question_ranges(message)) + 4,
                ),
            )
        else:
            structured_top_k = max(
                settings.rag_top_k,
                settings.rag_overview_top_k,
            )
        structured_sources = await run_in_threadpool(
            store.structured_lookup,
            retrieval_query,
            query_intent,
            structured_top_k,
            scope_ids,
        )
        retrieval_method = "structured" if structured_sources else None
        if structured_sources:
            source_limit = (
                50
                if query_intent == "document_overview"
                else structured_top_k
                if query_intent == "solve_questions"
                else settings.rag_top_k
            )
            sources = structured_sources[:source_limit]
        elif query_intent == "solve_questions" and solve_question_numbers:
            sources = []
            retrieval_method = "structured_question_no_match"
        elif query_intent == "document_overview":
            sources = await run_in_threadpool(
                store.overview,
                settings.rag_overview_top_k,
                scope_ids,
            )
            for source in sources:
                source["probe_overview_score"] = 1.0
            retrieval_method = "overview"
        elif (
            scope.request_mode == "explicit"
            and scope_ids
            and set(scope_ids).issubset(set(attached_document_ids))
            and query_intent in {"translate_content", "semantic_qa"}
        ):
            sources = await run_in_threadpool(
                store.document_chunks,
                settings.rag_top_k,
                scope_ids,
            )
            retrieval_method = "explicit_scope"
        elif probe.get("has_strong_hits"):
            sources = (probe.get("results") or [])[: settings.rag_top_k]
            retrieval_method = "probe"
        else:
            sources = []
            retrieval_method = "no_strong_document_match"
        before_filter_count = len(sources)
        sources = filter_sources_for_intent(sources, message, query_intent)
        if query_intent in {"semantic_qa", "writing_generation"} and any(
            source.get("metadata", {}).get("unit_type") in {"writing_task", "sample_answer"}
            for source in sources
        ):
            writing_context = await run_in_threadpool(
                store.writing_context_for_sources,
                sources,
                4,
                scope_ids,
            )
            if writing_context:
                sources = writing_context
                writing_parent_id = sources[0].get("metadata", {}).get("parent_id")
                retrieval_method = "writing_parent"
        if query_intent == "solve_questions" and sources:
            question_context = await run_in_threadpool(
                store.question_context_for_sources,
                sources,
                min(50, max(8, len(solve_question_numbers) * 2 + 4)),
                scope_ids,
            )
            question_sources = exact_question_sources(
                message,
                sources + question_context,
            )
            initial_packets = solve_question_packets(
                message,
                sources + question_context,
            )
            evidence_per_question = min(
                settings.rag_solve_evidence_per_question,
                max(
                    1,
                    settings.rag_solve_max_evidence // max(1, len(initial_packets)),
                ),
            )
            evidence_context: list[dict[str, Any]] = []
            evidence_queries: list[str] = []
            question_sources_by_chunk_id = {
                source.get("chunk_id"): source
                for source in question_sources
                if source.get("chunk_id")
            }
            for packet in initial_packets:
                question_number = packet.get("question_number")
                question_source = question_sources_by_chunk_id.get(
                    packet.get("question_chunk_id")
                )
                document_id = packet.get("document_id")
                passage_number = packet.get("passage_number")
                question_query = evidence_query_for_solve_packet(packet, message)
                evidence_queries.append(question_query)
                if question_source is None or not packet.get("context_ready"):
                    evidence_by_question.append(
                        {
                            "question_number": question_number,
                            "question_chunk_id": packet.get("question_chunk_id"),
                            "document_id": document_id,
                            "passage_number": passage_number,
                            "question_type": packet.get("question_type"),
                            "answer_option_labels": packet.get("answer_option_labels", []),
                            "word_limit": packet.get("word_limit"),
                            "query": question_query,
                            "candidate_chunk_ids": [],
                            "selected_chunk_ids": [],
                            "fallback_used": False,
                            "skipped_reason": (
                                "missing_question_source"
                                if question_source is None
                                else "invalid_question_structure"
                            ),
                            "warnings": packet.get("warnings", []),
                        }
                    )
                    continue
                pair_candidates = (
                    await run_in_threadpool(
                        store.hybrid_search,
                        question_query,
                        max(evidence_per_question * 2, 4),
                        [document_id],
                        ["passage"],
                        [passage_number],
                    )
                    if document_id and passage_number is not None
                    else []
                )
                selected = pair_candidates[:evidence_per_question]
                evidence_candidate_count += len(pair_candidates)
                remaining = max(
                    0,
                    settings.rag_solve_max_evidence - len(evidence_context),
                )
                selected = selected[:remaining]
                evidence_context.extend(selected)
                evidence_by_question.append(
                    {
                        "question_number": question_number,
                        "question_chunk_id": question_source.get("chunk_id"),
                        "document_id": document_id,
                        "passage_number": passage_number,
                        "question_type": packet.get("question_type"),
                        "answer_option_labels": packet.get("answer_option_labels", []),
                        "word_limit": packet.get("word_limit"),
                        "query": question_query,
                        "candidate_chunk_ids": [
                            source.get("chunk_id") for source in pair_candidates
                        ],
                        "candidate_scores": [
                            {
                                "chunk_id": source.get("chunk_id"),
                                "dense": source.get("probe_dense_score", 0.0),
                                "keyword": source.get("probe_keyword_score", 0.0),
                                "rrf": source.get("rrf_score", 0.0),
                                "methods": source.get("retrieval_methods", []),
                            }
                            for source in pair_candidates
                        ],
                        "selected_chunk_ids": [
                            source.get("chunk_id") for source in selected
                        ],
                        "fallback_used": False,
                    }
                )
            evidence_query = " | ".join(dict.fromkeys(evidence_queries)) or None
            evidence_context = dedupe_sources(evidence_context)
            evidence_context_count = len(evidence_context)
            sources = solve_sources_with_selected_evidence(
                sources,
                question_context,
                evidence_context,
            )
            solve_report = solve_context_report(message, sources)
            solve_report["evidence_by_question"] = evidence_by_question
        elif query_intent == "semantic_qa" and use_affinity_context and sources:
            passage_context = await run_in_threadpool(
                store.passage_context_for_sources,
                sources,
                3,
                scope_ids,
            )
            if passage_context:
                sources = dedupe_sources(sources + passage_context)
                retrieval_method = f"{retrieval_method or 'semantic'}_with_parent"
        sources = dedupe_sources(sources)
    else:
        structured_sources = []
        retrieval_method = None
        before_filter_count = 0
        evidence_candidate_count = 0
        evidence_context_count = 0

    if query_intent == "solve_questions" and not solve_report:
        solve_report = solve_context_report(message, sources)

    debug = {
        "route_decision": route,
        "query_intent": query_intent,
        "intent_decision": intent_debug,
        "route_gateway": gateway_debug,
        "document_resolution": {
            **document_resolution_debug,
            "resolved_document_ids": scope_ids,
            "requested_scope": scope.to_debug(),
        },
        "intent_classifier": intent_classifier.to_debug(),
        "target_resolution": scope.to_debug(),
        "catalog": catalog,
        "probe": compact_probe_debug(probe),
        "retrieval": {
            "method": retrieval_method,
            "structured_hits": len(structured_sources),
            "before_filter_count": before_filter_count,
            "after_filter_count": len(sources),
            "evidence_candidate_count": evidence_candidate_count,
            "evidence_context_count": evidence_context_count,
            "evidence_query": evidence_query,
            "evidence_per_question": evidence_per_question,
            "evidence_by_question": evidence_by_question,
            "solve_context_report": solve_report,
            "retrieval_query": retrieval_query,
            "writing_parent_id": writing_parent_id,
            "final_context": compact_final_context_debug(sources),
        },
        "source_count": len(sources),
        "conversation_state": req.conversation_state.model_dump() if req.conversation_state else None,
    }

    if sources:
        if query_intent == "solve_questions":
            context_issue = solve_context_issue(sources, message)
            if context_issue:
                debug["no_match_guard"] = context_issue
                return ChatPreparation(
                    prompt=None,
                    static_response=INCOMPLETE_QUESTION_RESPONSE,
                    route_used="vector_rag_no_match",
                    sources=sources,
                    debug=debug,
                    query_intent=query_intent,
                )
        if is_presence_check_query(message) and not has_lexical_source_hit(sources):
            debug["no_match_guard"] = "presence_check_without_lexical_hit"
            return ChatPreparation(
                prompt=None,
                static_response=no_rag_match_response(message),
                route_used="vector_rag_no_match",
                sources=sources,
                debug=debug,
                query_intent=query_intent,
            )
        static_response = static_response_for_sources(message, query_intent, sources)
        if static_response:
            debug["static_response"] = True
            return ChatPreparation(
                prompt=None,
                static_response=static_response,
                route_used="vector_rag_static",
                sources=sources,
                debug=debug,
                query_intent=query_intent,
            )
        deterministic_intents = {
            "show_questions",
            "show_table",
            "extract_table",
            "table_cell",
            "table_calculation",
            "table_comparison",
            "show_flowchart",
            "show_diagram",
            "show_writing_prompt",
        }
        if query_intent in deterministic_intents:
            debug["no_match_guard"] = "deterministic_intent_without_structured_response"
            return ChatPreparation(
                prompt=None,
                static_response=no_rag_match_response(message),
                route_used="vector_rag_no_match",
                sources=sources,
                debug=debug,
                query_intent=query_intent,
            )
        if query_intent == "solve_questions":
            context = format_solve_context(sources, solve_report)
        else:
            prompt_sources = sources
            if (
                query_intent == "explain_questions"
                and has_explicit_no_solution_constraint(message)
            ):
                prompt_sources = [
                    source
                    for source in sources
                    if source.get("metadata", {}).get("unit_type")
                    in {"question_group", "question"}
                ]
                debug["retrieval"]["no_solution_context"] = {
                    "source_count": len(prompt_sources),
                    "excluded_passage_evidence": len(sources) - len(prompt_sources),
                }
            context = (
                format_context(
                    prompt_sources,
                    max_chars_per_source=settings.rag_overview_source_chars,
                )
                if probe.get("is_overview")
                else format_context(prompt_sources)
            )
        if query_intent == "writing_generation":
            facts = writing_table_facts(sources)
            if facts:
                debug["retrieval"]["deterministic_table_facts"] = facts
                context += "\n\n[Deterministic table facts]\n" + "\n".join(
                    f"- {fact}" for fact in facts
                )
        return ChatPreparation(
            prompt=rag_prompt(
                message,
                context,
                req.conversation_history,
                query_intent=query_intent,
                allow_solution=bool(intent_debug.get("allow_solution")),
                writing_context=query_intent == "writing_generation",
                user_profile=user_profile_context(req),
            ),
            static_response=None,
            route_used="vector_rag",
            sources=sources,
            debug=debug,
            query_intent=query_intent,
        )

    if route == "rag":
        return ChatPreparation(
            prompt=None,
            static_response=(
                INCOMPLETE_QUESTION_RESPONSE
                if query_intent == "solve_questions"
                and solve_report.get("requested_question_numbers")
                else no_rag_match_response(message)
            ),
            route_used="vector_rag_no_match",
            sources=[],
            debug=debug,
            query_intent=query_intent,
        )

    return ChatPreparation(
        prompt=direct_answer_prompt(
            message,
            req.conversation_history,
            user_profile_context(req),
            previous_answer_source=direct_conversation_source(req),
        ),
        static_response=None,
        route_used="base_model",
        sources=[],
        debug=debug,
        query_intent=query_intent,
    )


@app.get("/health")
async def health() -> dict:
    stats = get_store_manager().runtime_stats()
    return {
        "status": "ok",
        "runtime_status": (LAST_WARMUP_STATUS or {}).get("status", "not_warmed"),
        "model_readiness": (LAST_WARMUP_STATUS or {}).get("components", {}),
        "rag_sessions_active": stats["active_sessions"],
        "rag_sessions_in_flight": stats["in_flight_sessions"],
        "rag_sessions_cached": stats["cached_sessions"],
        "rag_cache_evictions_total": stats["cache_evictions"],
        "rag_sessions_cleaned_total": stats["cleaned_sessions"],
        "rag_session_cleanup_errors": SESSION_CLEANUP_ERRORS,
    }


@app.get("/admin/stats", dependencies=[Depends(require_api_auth)])
async def admin_stats() -> dict:
    stats = await run_in_threadpool(get_store_manager().stats)
    return {
        "rag": stats,
        "rate_limits": REQUEST_RATE_LIMITER.stats(),
        "cleanup_errors": SESSION_CLEANUP_ERRORS,
        "limits": {
            "chat_rate": settings.chat_rate_limit,
            "chat_window_seconds": settings.chat_rate_window_seconds,
            "upload_rate": settings.upload_rate_limit,
            "upload_window_seconds": settings.upload_rate_window_seconds,
            "session_max_documents": settings.rag_session_max_documents,
            "session_max_chunks": settings.rag_session_max_chunks,
            "chat_concurrency": settings.chat_max_concurrency,
            "upload_concurrency": settings.upload_max_concurrency,
        },
        "backend_worker_mode": "single_process_required",
    }


@app.post("/warmup", dependencies=[Depends(require_api_auth)])
async def warmup() -> dict:
    global LAST_WARMUP_STATUS
    started = time.perf_counter()
    results = {}

    if settings.warmup_llm:
        llm_started = time.perf_counter()
        try:
            response = await query_ollama(
                direct_answer_prompt("Give me one concise IELTS Speaking tip."),
                temperature=0.2,
                num_predict=192,
            )
            direct_check = await classify_chat_route(
                "Explain a common technology concept in one sentence."
            )
            rag_check = await classify_chat_route("Summarize the content of the uploaded document.")
            intent_check = await classify_rag_intent("List Questions 1-4 without solving them.")
            gateway_ok = (
                bool(response.strip())
                and direct_check.route == "direct"
                and rag_check.route == "rag"
                and intent_check.intent in {
                    "show_questions",
                    "translate_questions",
                    "explain_questions",
                    "solve_questions",
                    "semantic_qa",
                    "document_overview",
                    "show_table",
                    "extract_table",
                    "table_cell",
                    "table_calculation",
                    "table_comparison",
                    "show_flowchart",
                    "show_diagram",
                    "show_writing_prompt",
                    "writing_generation",
                }
            )
            results["llm"] = {
                "ok": gateway_ok,
                "model": OLLAMA_MODEL,
                "duration_seconds": round(time.perf_counter() - llm_started, 2),
                "sample": response[:120],
                "gateway": {
                    "direct": direct_check.to_debug(),
                    "rag": rag_check.to_debug(),
                    "intent": intent_check.to_debug(),
                },
            }
        except Exception as exc:
            results["llm"] = {
                "ok": False,
                "error": str(exc),
                "diagnostic": ollama_failure_detail(exc).get("ollama"),
            }
    else:
        results["llm"] = {"skipped": True}

    if settings.warmup_embedding:
        embedding_started = time.perf_counter()
        try:
            embedding_result = await run_in_threadpool(get_store_manager().warmup)
            results["embedding"] = {
                "ok": True,
                "duration_seconds": round(time.perf_counter() - embedding_started, 2),
                **embedding_result,
            }
        except Exception as exc:
            results["embedding"] = {"ok": False, "error": str(exc)}
    else:
        results["embedding"] = {"skipped": True}

    layout_started = time.perf_counter()
    try:
        layout_result = await run_in_threadpool(DOCUMENT_PROCESSOR.warmup_layout)
        results["layout"] = {
            "ok": bool(layout_result.get("skipped") or layout_result.get("ok", False)),
            "duration_seconds": round(time.perf_counter() - layout_started, 2),
            **layout_result,
        }
    except Exception as exc:
        results["layout"] = {"ok": False, "error": str(exc)}

    ocr_started = time.perf_counter()
    try:
        ocr_result = await run_in_threadpool(DOCUMENT_PROCESSOR.warmup_ocr)
        results["ocr"] = {
            "ok": bool(ocr_result.get("skipped") or ocr_result.get("models_ready", False)),
            "duration_seconds": round(time.perf_counter() - ocr_started, 2),
            **ocr_result,
        }
    except Exception as exc:
        results["ocr"] = {"ok": False, "error": str(exc)}

    ok = all(component.get("ok", True) for component in results.values())
    status = "ok" if ok else "partial"
    LAST_WARMUP_STATUS = {
        "status": status,
        "components": {
            name: bool(result.get("ok", result.get("skipped", False)))
            for name, result in results.items()
        },
    }
    return {
        "status": status,
        "duration_seconds": round(time.perf_counter() - started, 2),
        "results": results,
    }


@app.post(
    "/chat/stream",
    dependencies=[Depends(require_api_auth), Depends(enforce_chat_rate)],
)
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung câu hỏi")
    await CHAT_CONCURRENCY.acquire()

    async def generate_for_request(active_req: ChatRequest):
        resource_debug = {"start": resource_snapshot()} if settings.debug_payloads else {}
        prepared: ChatPreparation | None = None
        delivered_parts: list[str] = []

        def token_event(token: str) -> str:
            delivered_parts.append(token)
            return stream_event("token", token=token)

        async def persist_successful_turn() -> None:
            if prepared is None:
                return
            await run_in_threadpool(
                persist_session_turn,
                active_req,
                prepared,
                "".join(delivered_parts),
            )

        def finish_resource_debug() -> None:
            if not settings.debug_payloads or "end" in resource_debug:
                return
            resource_debug["end"] = resource_snapshot()
            resource_debug["delta_mb"] = resource_delta(
                resource_debug["start"],
                resource_debug["end"],
            )

        def metadata_event() -> str:
            if prepared is None:
                raise RuntimeError("Chat preparation is not available.")
            prepared.debug["resources"] = resource_debug
            return stream_event(
                "metadata",
                route_used=prepared.route_used,
                sources=prepared.sources if settings.debug_payloads else [],
                debug=prepared.debug if settings.debug_payloads else None,
                conversation_state=(
                    conversation_state_for_result(active_req, prepared).model_dump()
                    if settings.debug_payloads
                    else None
                ),
            )

        try:
            yield stream_event("status", message="Đang phân tích câu hỏi...")
            prepared = await prepare_chat(active_req)
            prepared.debug["session_id"] = str(active_req.session_id)
            await collect_user_fact_updates(active_req, prepared)
            if prepared.static_response is not None:
                prepared.debug["delivery"] = {
                    "endpoint": "static",
                    "mode": "buffered_then_streamed",
                    "buffer_reason": "static_response",
                }
            elif requires_reviewed_generation(prepared, active_req.message):
                prepared.debug["delivery"] = {
                    "endpoint": (
                        "chat" if prepared.route_used == "base_model" else "generate"
                    ),
                    "mode": "buffered_then_streamed",
                    "buffer_reason": response_buffer_reason(prepared, active_req.message),
                }
            else:
                prepared.debug["delivery"] = {
                    "endpoint": "generate",
                    "mode": "live_stream",
                    "buffer_reason": None,
                }
            yield metadata_event()
            if prepared.static_response is not None:
                async for token in buffered_response_chunks(prepared.static_response):
                    yield token_event(token)
                finish_resource_debug()
                await persist_successful_turn()
                yield metadata_event()
                yield stream_event("done")
                return

            yield stream_event("status", message="Đang soạn câu trả lời...")
            if requires_reviewed_generation(prepared, active_req.message):
                direct_reviewed = (
                    prepared.query_intent == "direct"
                    and prepared.route_used == "base_model"
                )
                direct_fallback_enabled = (
                    direct_reviewed and settings.ollama_chat_fallback
                )
                if direct_reviewed:
                    previous_answer_source = direct_conversation_source(active_req)
                    generation_debug = prepared.debug.setdefault("direct_generation", {})
                    direct_writing = is_direct_writing_request(active_req.message)
                    direct_source_available = True
                    if direct_writing:
                        source_decision = await classify_direct_source(
                            active_req.message,
                            active_req.conversation_history,
                        )
                        source_debug = source_decision.to_debug()
                        source_debug["method"] = "semantic_current_and_history"
                        source_debug["history_included"] = bool(active_req.conversation_history)
                        source_debug["blocking"] = False
                        generation_debug["source_sufficiency"] = source_debug
                        direct_source_available = source_decision.source == "available"
                        if direct_source_available and active_req.conversation_history and any(
                            message.role == "assistant"
                            for message in active_req.conversation_history
                        ):
                            previous_answer_source = "conversation"
                        elif not direct_source_available:
                            previous_answer_source = "none"
                        generation_debug["previous_answer_source"] = previous_answer_source

                    primary_response_debug = generation_debug.setdefault("primary_response", {})
                    primary_output_contract: list[str] | None = None
                    if direct_writing and direct_source_available:
                        primary_output_contract = [
                            "- The requested task source is available. Apply the Writing constraints below."
                        ] + writing_output_contract(active_req.message).prompt_lines()
                    try:
                        initial_answer = await query_ollama_chat(
                            direct_chat_messages(
                                active_req.message,
                                active_req.conversation_history,
                                user_profile_context(active_req),
                                previous_answer_source=previous_answer_source,
                                output_contract=primary_output_contract,
                            ),
                            temperature=generation_temperature(prepared),
                            response_debug=primary_response_debug,
                        )
                    except OllamaRequestError as exc:
                        if not direct_fallback_enabled or exc.kind not in {
                            "empty_response",
                            "prompt_echo",
                            "role_continuation",
                        }:
                            raise
                        initial_answer = await direct_reviewed_generation_fallback(
                            prepared,
                            active_req,
                            exc.kind,
                            previous_answer_source,
                            direct_source_available,
                        )
                    active_response_debug = (
                        generation_debug.get("fallback_response", {})
                        if generation_debug.get("fallback_used")
                        else primary_response_debug
                    )

                    async def retry_direct_generation(
                        output_contract: list[str],
                    ) -> tuple[str | None, str | None]:
                        retry_response_debug = generation_debug.setdefault(
                            "contract_retry_response",
                            {},
                        )
                        generation_debug["contract_retry_endpoint"] = "chat"
                        try:
                            retry_answer = await query_ollama_chat(
                                direct_chat_messages(
                                    active_req.message,
                                    active_req.conversation_history,
                                    user_profile_context(active_req),
                                    previous_answer_source=previous_answer_source,
                                    output_contract=output_contract,
                                ),
                                temperature=0.1,
                                response_debug=retry_response_debug,
                            )
                        except OllamaRequestError as exc:
                            generation_debug.update(
                                {
                                    "contract_retry_status": "failed",
                                    "contract_retry_error": exc.kind,
                                }
                            )
                            return None, None
                        generation_debug["contract_retry_status"] = "succeeded"
                        return retry_answer, retry_response_debug.get("done_reason")

                    answer = (
                        await generate_answer(
                            prepared,
                            active_req.message,
                            initial_answer=initial_answer,
                            initial_done_reason=active_response_debug.get("done_reason"),
                            direct_source_available=direct_source_available,
                            direct_retry=retry_direct_generation,
                        )
                        if initial_answer.strip()
                        else ""
                    )
                else:
                    answer = await generate_answer(prepared, active_req.message)
                if not answer.strip():
                    answer = generation_fallback(prepared, active_req.message)
                async for token in buffered_response_chunks(answer):
                    yield token_event(token)
                finish_resource_debug()
                await persist_successful_turn()
                yield metadata_event()
                yield stream_event("done")
                return

            has_token = False
            streamed_substantive = False
            pending_suffix = ""
            stream_failure: OllamaRequestError | None = None
            temperature = generation_temperature(prepared)
            try:
                async for token in stream_ollama(
                    prepared.prompt or "",
                    temperature=temperature,
                ):
                    pending_suffix += token
                    pending_length = pending_no_match_suffix_length(pending_suffix)
                    if pending_length:
                        visible = pending_suffix[:-pending_length]
                        pending_suffix = pending_suffix[-pending_length:]
                    else:
                        visible = pending_suffix
                        pending_suffix = ""
                    if visible:
                        has_token = True
                        streamed_substantive = streamed_substantive or bool(visible.strip())
                        yield token_event(visible)
            except OllamaRequestError as exc:
                if exc.kind != "prompt_echo" or has_token:
                    raise
                stream_failure = exc
            stream_cleanup: str | None = None
            if streamed_substantive:
                for response in SYSTEM_NO_RAG_MATCH_RESPONSES:
                    if pending_suffix.rstrip() == f"\n\n{response}":
                        stream_cleanup = response
                        pending_suffix = ""
                        break
            if pending_suffix:
                has_token = True
                yield token_event(pending_suffix)
            if stream_cleanup:
                prepared.debug.setdefault("generation", {})[
                    "stream_cleanup"
                ] = stream_cleanup
            if not has_token:
                direct_fallback = (
                    prepared.query_intent == "direct" and settings.ollama_chat_fallback
                )
                fallback_endpoint = "chat" if direct_fallback else "generate"
                fallback_reason = stream_failure.kind if stream_failure else "empty_stream"
                logger.warning(
                    "Ollama stream failed before visible tokens (%s); retrying via %s endpoint",
                    fallback_reason,
                    fallback_endpoint,
                )
                generation_debug = prepared.debug.setdefault("direct_generation", {})
                try:
                    if direct_fallback:
                        fallback_answer = await query_ollama_chat(
                            direct_chat_messages(
                                active_req.message,
                                active_req.conversation_history,
                                user_profile_context(active_req),
                                previous_answer_source=direct_conversation_source(active_req),
                            ),
                            temperature=temperature,
                        )
                    else:
                        fallback_answer = await query_ollama(
                            prepared.prompt or "",
                            temperature=temperature,
                        )
                except Exception as exc:
                    generation_debug.update(
                        {
                            "fallback_used": True,
                            "fallback_reason": fallback_reason,
                            "fallback_endpoint": fallback_endpoint,
                            "fallback_status": "failed",
                            "fallback_error": (
                                exc.kind if isinstance(exc, OllamaRequestError) else type(exc).__name__
                            ),
                        }
                    )
                    raise
                if not fallback_answer.strip():
                    fallback_answer = generation_fallback(prepared, active_req.message)
                generation_debug.update(
                    {
                        "fallback_used": True,
                        "fallback_reason": fallback_reason,
                        "fallback_endpoint": fallback_endpoint,
                        "fallback_status": "succeeded",
                    }
                )
                yield metadata_event()
                prepared.debug["delivery"] = {
                    "endpoint": fallback_endpoint,
                    "mode": "buffered_then_streamed",
                    "buffer_reason": "stream_fallback",
                }
                async for token in buffered_response_chunks(fallback_answer):
                    yield token_event(token)
            finish_resource_debug()
            await persist_successful_turn()
            yield metadata_event()
            yield stream_event("done")
        except Exception as exc:
            logger.exception("Streaming chat failed")
            finish_resource_debug()
            if prepared is not None:
                yield metadata_event()
            failure_detail = ollama_failure_detail(exc)
            failure_detail["resources"] = resource_debug
            yield stream_event(
                "error",
                message="Không thể tạo câu trả lời lúc này. Vui lòng thử lại.",
                detail=failure_detail,
            )
    async def generate():
        manager = get_store_manager()
        operation_started = False
        try:
            async with SESSION_CHAT_LOCKS.hold(req.session_id):
                await run_in_threadpool(manager.begin_session_operation, req.session_id)
                operation_started = True
                active_req = await run_in_threadpool(backend_session_request, req)
                async for event in generate_for_request(active_req):
                    yield event
        except Exception as exc:
            logger.exception("Session chat lifecycle failed")
            yield stream_event(
                "error",
                message="Không thể sử dụng phiên này lúc này. Vui lòng thử lại.",
                detail=ollama_failure_detail(exc),
            )
        finally:
            if operation_started:
                await run_in_threadpool(manager.end_session_operation, req.session_id)
            CHAT_CONCURRENCY.release()

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/documents/upload",
    response_model=UploadResponse,
    dependencies=[Depends(require_api_auth), Depends(enforce_upload_rate)],
)
async def upload_document(
    session_id: UUID = Form(...),
    file: UploadFile = File(...),
) -> UploadResponse:
    upload_started = time.perf_counter()
    upload_timing: dict[str, Any] = {}
    timing_debug: dict[str, Any] = {"upload": {}}

    if not file.filename:
        raise HTTPException(status_code=400, detail="Tên tệp không hợp lệ")
    await UPLOAD_CONCURRENCY.acquire()

    safe_name = Path(file.filename).name
    request_id = uuid4().hex
    file_path = UPLOAD_DIR / f"{request_id}-{safe_name}"
    max_bytes = DOCUMENT_PROCESSOR.config.max_upload_mb * 1024 * 1024
    manager = get_store_manager()
    operation_started = False

    try:
        await run_in_threadpool(manager.begin_session_operation, session_id)
        operation_started = True
        save_started = time.perf_counter()
        total_bytes = 0
        async with aiofiles.open(file_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Tệp quá lớn. Giới hạn hiện tại là {DOCUMENT_PROCESSOR.config.max_upload_mb}MB.",
                    )
                await out.write(chunk)
        upload_timing["save_file_seconds"] = round(time.perf_counter() - save_started, 3)

        process_started = time.perf_counter()
        document, chunks = await run_in_threadpool(
            DOCUMENT_PROCESSOR.process_file,
            file_path,
            safe_name,
            file.content_type,
        )
        upload_timing["process_file_seconds"] = round(time.perf_counter() - process_started, 3)
        document_timing = document.metadata.get("timing", {})
        timing_debug = {
            "upload": dict(upload_timing),
            "extraction": document_timing.get("extraction", {}),
            "process_file": document_timing.get("process_file", {}),
            "chunking": document_timing.get("chunking", {}),
            "embedding": {},
        }
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=document_extraction_failure_detail(document),
            )

        store = get_store(session_id)
        projected_stats = await run_in_threadpool(
            store.projected_stats,
            [chunk.to_dict() for chunk in chunks],
            safe_name,
        )
        if projected_stats["documents"] > settings.rag_session_max_documents:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Phiên đã đạt giới hạn "
                    f"{settings.rag_session_max_documents} tài liệu. "
                    "Hãy làm mới phiên trước khi tải thêm."
                ),
            )
        if projected_stats["chunks"] > settings.rag_session_max_chunks:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Tổng nội dung tài liệu trong phiên vượt giới hạn xử lý. "
                    "Hãy làm mới phiên hoặc tải ít tài liệu hơn."
                ),
            )
        upsert_started = time.perf_counter()
        inserted = await run_in_threadpool(
            store.upsert,
            [chunk.to_dict() for chunk in chunks],
            safe_name,
        )
        upload_timing["upsert_seconds"] = round(time.perf_counter() - upsert_started, 3)
        upload_timing["total_seconds"] = round(time.perf_counter() - upload_started, 3)
        timing_debug["upload"] = dict(upload_timing)
        timing_debug["embedding"] = dict(store.last_upsert_timing)
        logger.info(
            "Document indexed",
            extra={
                "source_file": safe_name,
                "document_id": document.document_id,
                "chunks": inserted,
                "bytes": total_bytes,
            },
        )
        return UploadResponse(
            session_id=session_id,
            message=f"Processed {inserted} chunks",
            file_name=document.filename,
            document_id=document.document_id,
            document_type=document.metadata.get("document_type") or document.mime_type,
            chunks_processed=inserted,
            collection_stats=await run_in_threadpool(store.stats),
            debug=(
                {
                    "timing": timing_debug,
                    "extraction": document.metadata.get("extraction_report", {}),
                    "structure": document.metadata.get("ielts_structure", {}).get("diagnostics", {}),
                    "outline": document.metadata.get("ielts_structure", {}).get("outline", {}),
                }
                if settings.debug_payloads
                else None
            ),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Document processing failed for %s", safe_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected document processing failure for %s", safe_name)
        raise HTTPException(status_code=500, detail="Không thể xử lý tài liệu này.") from exc
    finally:
        if operation_started:
            await run_in_threadpool(manager.end_session_operation, session_id)
        UPLOAD_CONCURRENCY.release()
        file_path.unlink(missing_ok=True)


@app.post(
    "/rag/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_api_auth)],
)
async def search(req: SearchRequest) -> SearchResponse:
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung tìm kiếm")
    manager = get_store_manager()
    await run_in_threadpool(manager.begin_session_operation, req.session_id)
    try:
        results = await run_in_threadpool(
            get_store(req.session_id).search,
            query,
            req.top_k,
            req.document_ids,
        )
    finally:
        await run_in_threadpool(manager.end_session_operation, req.session_id)
    return SearchResponse(query=query, results=results)


@app.get(
    "/rag/stats",
    response_model=StatsResponse,
    dependencies=[Depends(require_api_auth)],
)
async def stats(session_id: UUID) -> StatsResponse:
    manager = get_store_manager()
    await run_in_threadpool(manager.begin_session_operation, session_id)
    try:
        session_stats = await run_in_threadpool(get_store(session_id).stats)
    finally:
        await run_in_threadpool(manager.end_session_operation, session_id)
    return StatsResponse(session_id=session_id, **session_stats)


@app.post(
    "/sessions/{session_id}/expire",
    response_model=SessionExpireResponse,
    dependencies=[Depends(require_api_auth)],
)
async def expire_session(session_id: UUID) -> SessionExpireResponse:
    manager = get_store_manager()
    scheduled = await run_in_threadpool(manager.schedule_session_expiration, session_id)
    return SessionExpireResponse(
        session_id=session_id,
        scheduled=scheduled,
        expires_in_seconds=manager.grace_ttl_seconds,
    )


@app.delete(
    "/sessions/{session_id}",
    response_model=SessionDeleteResponse,
    dependencies=[Depends(require_api_auth)],
)
async def delete_session(session_id: UUID) -> SessionDeleteResponse:
    deleted = await run_in_threadpool(get_store_manager().delete_session, session_id)
    REQUEST_RATE_LIMITER.clear_session(session_id)
    return SessionDeleteResponse(session_id=session_id, deleted=deleted)
