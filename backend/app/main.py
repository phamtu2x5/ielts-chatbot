from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import settings
from .llm import (
    OLLAMA_MODEL,
    OllamaRequestError,
    classify_direct_source,
    conversation_language,
    direct_answer_prompt,
    direct_chat_messages,
    extract_user_facts,
    is_direct_writing_request,
    query_ollama,
    query_ollama_chat,
    response_output_contract,
    response_output_issues,
    select_best_response_output,
    select_best_writing_output,
    writing_output_contract,
    writing_output_issues,
)
from .schemas import (
    ChatConversationState,
    ChatMessage,
    ChatRequest,
    ChatUserFact,
    SessionDeleteResponse,
    SessionExpireResponse,
)
from .session_store import get_session_store


logger = logging.getLogger(__name__)
LAST_WARMUP_STATUS: dict[str, Any] | None = None
SESSION_CLEANUP_ERRORS = 0


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


REQUEST_RATE_LIMITER = SlidingWindowRateLimiter()
SESSION_CHAT_LOCKS = SessionChatLockPool()
CHAT_CONCURRENCY = asyncio.Semaphore(settings.chat_max_concurrency)


async def session_cleanup_loop() -> None:
    global SESSION_CLEANUP_ERRORS
    while True:
        await asyncio.sleep(60)
        try:
            deleted = await run_in_threadpool(get_session_store().cleanup_expired)
            REQUEST_RATE_LIMITER.prune(settings.chat_rate_window_seconds)
            if deleted:
                logger.info("Expired chat sessions removed: %s", deleted)
        except Exception:
            SESSION_CLEANUP_ERRORS += 1
            logger.exception("Session cleanup failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(get_session_store().cleanup_expired)
    cleanup_task = asyncio.create_task(session_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


def enforce_chat_rate(req: ChatRequest) -> None:
    REQUEST_RATE_LIMITER.check(
        req.session_id,
        "chat",
        settings.chat_rate_limit,
        settings.chat_rate_window_seconds,
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
    title="IELTS Direct Chatbot",
    version="2.0.0",
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


def stream_event(event_type: str, **payload: Any) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


def ollama_failure_detail(exc: Exception) -> dict[str, Any]:
    result = {
        "message": "Không thể kết nối hoặc nhận câu trả lời từ Ollama.",
        "kind": exc.kind if isinstance(exc, OllamaRequestError) else type(exc).__name__,
    }
    if settings.debug_payloads and isinstance(exc, OllamaRequestError):
        result["ollama"] = exc.debug_detail()
    return result


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


def backend_session_request(req: ChatRequest) -> ChatRequest:
    payload = get_session_store().read_memory(req.session_id)
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


def conversation_state_for_result(
    req: ChatRequest,
    updates: list[ChatUserFact],
) -> ChatConversationState:
    previous = req.conversation_state or ChatConversationState()
    return ChatConversationState(
        user_facts=merge_user_facts(previous.user_facts, updates),
    )


def persist_session_turn(
    req: ChatRequest,
    assistant_answer: str,
    fact_updates: list[ChatUserFact],
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
    state = conversation_state_for_result(req, fact_updates)
    get_session_store().write_memory(
        req.session_id,
        [message.model_dump() for message in history[-20:]],
        state.model_dump(),
    )


def direct_conversation_source(req: ChatRequest) -> str:
    if not req.conversation_history:
        return "none"
    return (
        "conversation"
        if any(message.role == "assistant" for message in req.conversation_history)
        else "none"
    )


def generation_fallback(message: str) -> str:
    if conversation_language(message) == "English":
        return "What would you like help with? Please describe your request more specifically."
    return "Bạn muốn mình hỗ trợ nội dung gì? Hãy mô tả yêu cầu cụ thể hơn nhé."


def _select_complete_candidate(
    first: str,
    retry: str,
    *,
    first_incomplete: bool,
    retry_incomplete: bool,
    writing_contract: Any | None,
    response_contract: Any | None,
) -> str:
    if first_incomplete != retry_incomplete:
        return retry if not retry_incomplete else first
    if writing_contract is not None:
        return select_best_writing_output(first, retry, writing_contract)
    if response_contract is not None:
        return select_best_response_output(first, retry, response_contract)
    return first


async def generate_direct_answer(req: ChatRequest) -> tuple[str, dict[str, Any]]:
    debug: dict[str, Any] = {}
    previous_answer_source = direct_conversation_source(req)
    direct_writing = is_direct_writing_request(req.message)
    direct_source_available = True
    if direct_writing:
        source_decision = await classify_direct_source(
            req.message,
            req.conversation_history,
        )
        direct_source_available = source_decision.source == "available"
        if not direct_source_available:
            previous_answer_source = "none"
        if settings.debug_payloads:
            debug["source_sufficiency"] = source_decision.to_debug()

    writing_contract = (
        writing_output_contract(req.message)
        if direct_writing and direct_source_available
        else None
    )
    output_contract = None
    if writing_contract is not None:
        output_contract = [
            "- The requested task source is available. Apply the Writing constraints below."
        ] + writing_contract.prompt_lines()

    response_debug: dict[str, Any] = {}
    try:
        answer = await query_ollama_chat(
            direct_chat_messages(
                req.message,
                req.conversation_history,
                user_profile_context(req),
                previous_answer_source=previous_answer_source,
                output_contract=output_contract,
            ),
            temperature=settings.ollama_direct_temperature,
            response_debug=response_debug,
        )
        endpoint = "chat"
    except OllamaRequestError as exc:
        if not settings.ollama_chat_fallback or exc.kind not in {
            "empty_response",
            "prompt_echo",
            "role_continuation",
        }:
            raise
        answer = await query_ollama(
            direct_answer_prompt(
                req.message,
                req.conversation_history,
                user_profile_context(req),
                previous_answer_source=previous_answer_source,
                output_contract=output_contract,
            ),
            temperature=settings.ollama_direct_temperature,
            max_attempts=1,
        )
        endpoint = "generate_fallback"
        response_debug["fallback_reason"] = exc.kind

    first_incomplete = response_debug.get("done_reason") == "length"
    response_contract = None
    if writing_contract is not None:
        issues = writing_output_issues(answer, writing_contract)
    else:
        response_contract = response_output_contract(
            req.message,
            "direct",
            allow_solution=True,
        )
        issues = response_output_issues(answer, response_contract)
    if first_incomplete:
        issues.append("The response stopped because it reached the output length limit.")

    retryable = first_incomplete or bool(issues)
    retry_succeeded = False
    if retryable:
        retry_lines = (
            writing_contract.prompt_lines()
            if writing_contract is not None
            else response_contract.prompt_lines()
        )
        retry_lines.append("- Finish every requested section and do not stop mid-sentence.")
        retry_debug: dict[str, Any] = {}
        try:
            retry = await query_ollama_chat(
                direct_chat_messages(
                    req.message,
                    req.conversation_history,
                    user_profile_context(req),
                    previous_answer_source=previous_answer_source,
                    output_contract=retry_lines,
                ),
                temperature=0.1,
                response_debug=retry_debug,
            )
        except OllamaRequestError:
            retry = answer
        else:
            retry_succeeded = True
        answer = _select_complete_candidate(
            answer,
            retry,
            first_incomplete=first_incomplete,
            retry_incomplete=retry_debug.get("done_reason") == "length",
            writing_contract=writing_contract,
            response_contract=response_contract,
        )

    if settings.debug_payloads:
        debug.update(
            {
                "endpoint": endpoint,
                "previous_answer_source": previous_answer_source,
                "writing_contract": bool(writing_contract),
                "first_issues": issues,
                "retry_used": retryable,
                "retry_succeeded": retry_succeeded,
                "response": response_debug,
            }
        )
    return answer.strip() or generation_fallback(req.message), debug


@app.get("/health")
async def health() -> dict[str, Any]:
    stats = get_session_store().runtime_stats()
    return {
        "status": "ok",
        "runtime_status": (LAST_WARMUP_STATUS or {}).get("status", "not_warmed"),
        "model_readiness": (LAST_WARMUP_STATUS or {}).get("components", {}),
        "mode": "direct_only",
        "sessions_active": stats["active_sessions"],
        "sessions_in_flight": stats["in_flight_sessions"],
        "sessions_cleaned_total": stats["cleaned_sessions"],
        "session_cleanup_errors": SESSION_CLEANUP_ERRORS,
    }


@app.get("/admin/stats", dependencies=[Depends(require_api_auth)])
async def admin_stats() -> dict[str, Any]:
    return {
        "sessions": await run_in_threadpool(get_session_store().stats),
        "rate_limits": REQUEST_RATE_LIMITER.stats(),
        "cleanup_errors": SESSION_CLEANUP_ERRORS,
        "limits": {
            "chat_rate": settings.chat_rate_limit,
            "chat_window_seconds": settings.chat_rate_window_seconds,
            "chat_concurrency": settings.chat_max_concurrency,
        },
        "backend_worker_mode": "single_process_required",
    }


@app.post("/warmup", dependencies=[Depends(require_api_auth)])
async def warmup() -> dict[str, Any]:
    global LAST_WARMUP_STATUS
    started = time.perf_counter()
    if not settings.warmup_llm:
        result = {"skipped": True}
    else:
        try:
            sample = await query_ollama(
                direct_answer_prompt("Give me one concise IELTS Speaking tip."),
                temperature=0.2,
                num_predict=192,
            )
            result = {"ok": bool(sample.strip()), "model": OLLAMA_MODEL}
        except Exception as exc:
            result = {"ok": False, "error": ollama_failure_detail(exc)}
    status = "ok" if result.get("ok", result.get("skipped", False)) else "partial"
    LAST_WARMUP_STATUS = {"status": status, "components": {"llm": status == "ok"}}
    return {
        "status": status,
        "duration_seconds": round(time.perf_counter() - started, 2),
        "results": {"llm": result},
    }


@app.post(
    "/chat/stream",
    dependencies=[Depends(require_api_auth), Depends(enforce_chat_rate)],
)
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung câu hỏi")
    await CHAT_CONCURRENCY.acquire()

    async def generate() -> AsyncIterator[str]:
        store = get_session_store()
        operation_started = False
        try:
            async with SESSION_CHAT_LOCKS.hold(req.session_id):
                await run_in_threadpool(store.begin_operation, req.session_id)
                operation_started = True
                active_req = await run_in_threadpool(backend_session_request, req)
                yield stream_event("status", message="Đang phân tích câu hỏi...")
                fact_decision = await extract_user_facts(active_req.message)
                fact_updates = list(fact_decision.facts)
                yield stream_event(
                    "metadata",
                    route_used="base_model",
                    sources=[],
                    debug=None,
                    conversation_state=None,
                )
                yield stream_event("status", message="Đang soạn câu trả lời...")
                answer, generation_debug = await generate_direct_answer(active_req)
                delivered: list[str] = []
                async for token in buffered_response_chunks(answer):
                    delivered.append(token)
                    yield stream_event("token", token=token)
                await run_in_threadpool(
                    persist_session_turn,
                    active_req,
                    "".join(delivered),
                    fact_updates,
                )
                if settings.debug_payloads:
                    yield stream_event(
                        "metadata",
                        route_used="base_model",
                        sources=[],
                        debug={
                            "direct_generation": generation_debug,
                            "user_fact_extraction": fact_decision.to_debug(),
                        },
                        conversation_state=conversation_state_for_result(
                            active_req,
                            fact_updates,
                        ).model_dump(),
                    )
                yield stream_event("done")
        except Exception as exc:
            logger.exception("Direct chat failed")
            yield stream_event(
                "error",
                message="Không thể tạo câu trả lời lúc này. Vui lòng thử lại.",
                detail=ollama_failure_detail(exc),
            )
        finally:
            try:
                if operation_started:
                    await run_in_threadpool(store.end_operation, req.session_id)
            finally:
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
    "/sessions/{session_id}/expire",
    response_model=SessionExpireResponse,
    dependencies=[Depends(require_api_auth)],
)
async def expire_session(session_id: UUID) -> SessionExpireResponse:
    store = get_session_store()
    scheduled = await run_in_threadpool(store.schedule_expiration, session_id)
    return SessionExpireResponse(
        session_id=session_id,
        scheduled=scheduled,
        expires_in_seconds=store.grace_ttl_seconds,
    )


@app.delete(
    "/sessions/{session_id}",
    response_model=SessionDeleteResponse,
    dependencies=[Depends(require_api_auth)],
)
async def delete_session(session_id: UUID) -> SessionDeleteResponse:
    deleted = await run_in_threadpool(get_session_store().delete, session_id)
    REQUEST_RATE_LIMITER.clear_session(session_id)
    return SessionDeleteResponse(session_id=session_id, deleted=deleted)
