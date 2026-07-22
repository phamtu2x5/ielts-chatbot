import json
import sys
import tempfile
import types
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

from starlette.datastructures import Headers, UploadFile


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    import aiofiles  # noqa: F401
except ImportError:
    class _AsyncFile:
        def __init__(self, path: Path, mode: str) -> None:
            self._path = path
            self._mode = mode
            self._handle = None

        async def __aenter__(self):
            self._handle = self._path.open(self._mode)
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            self._handle.close()

        async def write(self, data: bytes) -> int:
            return self._handle.write(data)

    sys.modules["aiofiles"] = types.SimpleNamespace(
        open=lambda path, mode: _AsyncFile(Path(path), mode)
    )

try:
    __import__("dotenv")
except ImportError:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: False
    sys.modules["dotenv"] = dotenv_stub

try:
    __import__("sentence_transformers")
except ImportError:
    sentence_transformers_stub = types.ModuleType("sentence_transformers")
    sentence_transformers_stub.SentenceTransformer = object
    sys.modules["sentence_transformers"] = sentence_transformers_stub

from app import main
from app.document_pipeline.models import DocumentChunk, ProcessedDocument, ProcessedPage
from app.llm import IntentClassifierDecision, RouteGatewayDecision, TargetResolverDecision


def _gateway_decision(
    route: str,
    intent: str,
    *,
    document_refs: tuple[str, ...] = (),
    section_refs: tuple[str, ...] = (),
    reason: str = "test decision",
) -> RouteGatewayDecision:
    return RouteGatewayDecision(
        route="rag" if route == "clarify" else route,
        attempts=1,
        duration_seconds=0.01,
        raw_output_preview='{"route":"rag"}' if route != "direct" else '{"route":"direct"}',
        fallback_reason=None,
    )


class _FakeProcessor:
    class Config:
        max_upload_mb = 1

    config = Config()

    def process_file(
        self,
        file_path: Path,
        filename: str,
        content_type: str | None,
    ) -> tuple[ProcessedDocument, list[DocumentChunk]]:
        if file_path.read_bytes() != b"sample content":
            raise AssertionError("Upload content was not saved before extraction.")
        document = ProcessedDocument(
            document_id="doc-1",
            filename=filename,
            mime_type=content_type or "text/plain",
            parser_version="1.10.0",
            metadata={
                "document_type": "ielts_reading",
                "timing": {
                    "process_file": {"total_seconds": 0.01},
                    "chunking": {"chunks": 1},
                },
                "extraction_report": {"pages": []},
                "ielts_structure": {"diagnostics": {}, "outline": {}},
            },
            pages=[ProcessedPage(page_number=1, processing_route="text", quality_score=1.0)],
        )
        chunk = DocumentChunk(
            chunk_id="doc-1-c1",
            document_id="doc-1",
            source_file=filename,
            pages=[1],
            element_ids=[],
            heading_path=[],
            text="sample content",
            token_count=2,
            min_confidence=1.0,
            chunk_index=0,
            metadata={"parser_version": "1.10.0"},
        )
        return document, [chunk]


class _FakeStore:
    def __init__(self) -> None:
        self.last_upsert_timing = {"embedding_seconds": 0.01, "chunks": 1}
        self.received_chunks: list[dict] = []

    def upsert(self, chunks: list[dict], source_file: str) -> int:
        self.received_chunks = chunks
        if source_file != "sample.txt":
            raise AssertionError("Source filename was not passed to the store.")
        return len(chunks)

    def stats(self) -> dict:
        return {"documents": 1, "chunks": 1, "embedding_model": "test"}


class _FakeChatStore:
    def __init__(
        self,
        catalog: list[dict],
        *,
        has_document_intent: bool = True,
        top_question_score: float = 0.0,
    ) -> None:
        self.catalog = catalog
        self.has_document_intent = has_document_intent
        self.top_question_score = top_question_score
        self.probe_dense_flags: list[bool] = []
        self.probe_queries: list[str] = []
        self.probe_document_ids: list[list[str] | None] = []
        self.routing_document_ids: list[list[str] | None] = []
        self.routing_queries: list[str] = []
        self.routing_candidates: list[dict] = []

    def stats(self) -> dict:
        return {"documents": len(self.catalog), "chunks": len(self.catalog), "embedding_model": "test"}

    def document_catalog(self, document_ids=None) -> list[dict]:
        if document_ids is None:
            return self.catalog
        allowed = set(document_ids)
        return [
            item
            for item in self.catalog
            if allowed.intersection(item.get("document_ids", []))
        ]

    def probe_with_catalog(self, query, top_k, document_ids=None, include_dense=True):
        self.probe_dense_flags.append(include_dense)
        self.probe_queries.append(query)
        self.probe_document_ids.append(document_ids)
        return (
            {
                "results": [],
                "has_hits": False,
                "has_strong_hits": False,
                "has_document_intent": self.has_document_intent,
                "is_overview": False,
                "top_question_score": self.top_question_score,
            },
            self.document_catalog(document_ids),
        )

    def structured_lookup(self, query, intent, top_k, document_ids=None):
        return []

    def hybrid_search(
        self,
        query,
        top_k,
        document_ids=None,
        unit_types=None,
        passage_numbers=None,
    ):
        self.routing_document_ids.append(document_ids)
        self.routing_queries.append(query)
        return self.routing_candidates[:top_k]

    def search(self, query, top_k, document_ids=None):
        return []


class UploadIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.intent_patcher = patch.object(
            main,
            "classify_rag_intent",
            AsyncMock(
                return_value=IntentClassifierDecision(
                    intent="semantic_qa",
                    attempts=1,
                    duration_seconds=0.01,
                    raw_output_preview="semantic_qa",
                )
            ),
        )
        self.target_patcher = patch.object(
            main,
            "resolve_rag_target",
            AsyncMock(
                return_value=TargetResolverDecision(
                    document_refs=(),
                    action="clarify",
                    attempts=1,
                    duration_seconds=0.01,
                    raw_output_preview="CLARIFY",
                )
            ),
        )
        self.intent_patcher.start()
        self.target_patcher.start()

    async def asyncTearDown(self) -> None:
        self.intent_patcher.stop()
        self.target_patcher.stop()

    async def test_chat_stream_uses_the_same_completed_preparation_as_chat(self) -> None:
        prepared = main.ChatPreparation(
            prompt=None,
            static_response="Xin chào",
            route_used="base_model",
            sources=[],
            debug={"gateway": {"used": True, "decision": "direct"}},
        )

        prepare = AsyncMock(return_value=prepared)
        with patch.object(main, "prepare_chat", prepare):
            response = await main.chat_stream(main.ChatRequest(message="xin chào"))
            events = [json.loads(chunk) async for chunk in response.body_iterator]

        token_events = [event["token"] for event in events if event["type"] == "token"]
        self.assertEqual(token_events, ["Xin chào"])
        self.assertEqual(
            next(event for event in events if event["type"] == "metadata")["route_used"],
            "base_model",
        )
        prepare.assert_awaited_once()
        self.assertEqual(prepare.await_args.kwargs, {})

    async def test_direct_turn_preserves_previous_rag_affinity(self) -> None:
        prepared = main.ChatPreparation(
            prompt=None,
            static_response="Three tips.",
            route_used="base_model",
            sources=[],
            debug={},
            query_intent="direct",
        )
        request = main.ChatRequest(
            message="Give me three IELTS tips.",
            conversation_state={
                "last_route": "rag",
                "last_intent": "semantic_qa",
                "rag_affinity": {
                    "document_ids": ["doc-1"],
                    "passage_numbers": [2],
                    "question_ranges": [[14, 17]],
                },
                "user_profile": {
                    "current_band": 5.5,
                    "target_band": 6.5,
                    "study_duration_months": 3,
                },
            },
        )

        state = main.conversation_state_for_result(request, prepared)

        self.assertEqual(state.last_route, "direct")
        self.assertEqual(state.rag_affinity.document_ids, ["doc-1"])
        self.assertEqual(state.rag_affinity.passage_numbers, [2])
        self.assertEqual(state.user_profile.current_band, 5.5)
        self.assertEqual(state.user_profile.target_band, 6.5)
        self.assertEqual(state.user_profile.study_duration_months, 3)

    def test_user_profile_uses_only_facts_from_user_message_and_preserves_previous_values(self) -> None:
        initial_request = main.ChatRequest(
            message="Hiện tại tôi band 5.5, mục tiêu đạt 6.5 trong vòng 3 tháng.",
        )
        initial = main.user_profile_for_request(initial_request)

        self.assertEqual(initial.current_band, 5.5)
        self.assertEqual(initial.target_band, 6.5)
        self.assertEqual(initial.study_duration_months, 3)

        follow_up = main.ChatRequest(
            message="Bạn nhớ trình độ của tôi không?",
            conversation_history=[
                {"role": "assistant", "content": "Bạn đang ở band 8.0."},
            ],
            conversation_state={
                "last_route": "direct",
                "last_intent": "direct",
                "user_profile": initial.model_dump(),
            },
        )
        preserved = main.user_profile_for_request(follow_up)

        self.assertEqual(preserved.current_band, 5.5)
        self.assertEqual(preserved.target_band, 6.5)
        self.assertEqual(preserved.study_duration_months, 3)

    def test_target_band_phrase_does_not_overwrite_current_band(self) -> None:
        request = main.ChatRequest(
            message="Hiện tại tôi muốn lên band 6.5.",
            conversation_state={"user_profile": {"current_band": 5.5}},
        )

        profile = main.user_profile_for_request(request)

        self.assertEqual(profile.current_band, 5.5)
        self.assertEqual(profile.target_band, 6.5)

    async def test_direct_prompt_receives_authoritative_user_profile(self) -> None:
        with patch.object(
            main,
            "classify_chat_route",
            AsyncMock(return_value=_gateway_decision("direct", "direct")),
        ):
            prepared = await main.prepare_chat(
                main.ChatRequest(
                    message="Bạn nhớ tôi band bao nhiêu không?",
                    conversation_state={"user_profile": {"current_band": 5.5}},
                )
            )

        self.assertEqual(prepared.route_used, "base_model")
        self.assertIn("Current IELTS band: 5.5", prepared.prompt)
        self.assertEqual(prepared.debug["user_profile"], {"current_band": 5.5})

    async def test_gateway_failure_without_document_basis_returns_safe_http_200_result(self) -> None:
        catalog = [
            {"source_file": "reading.pdf", "document_ids": ["doc-1"], "mime_types": ["application/pdf"]}
        ]
        gateway = RouteGatewayDecision(
            route="undetermined",
            attempts=2,
            duration_seconds=0.1,
            raw_output_preview="",
            fallback_reason="empty_response",
        )
        with (
            patch.object(main, "get_store", return_value=_FakeChatStore(catalog)),
            patch.object(main, "classify_chat_route", AsyncMock(return_value=gateway)),
        ):
            response = await main.chat(main.ChatRequest(message="Tell me something useful."))

        self.assertEqual(response.route_used, "route_undetermined")
        self.assertEqual(response.conversation_state.last_route, "no_match")
        self.assertEqual(response.sources, [])

    async def test_single_explicit_document_never_calls_target_resolver(self) -> None:
        catalog = [
            {"source_file": "reading.pdf", "document_ids": ["doc-1"], "mime_types": ["application/pdf"]},
            {"source_file": "other.pdf", "document_ids": ["doc-2"], "mime_types": ["application/pdf"]},
        ]
        target = AsyncMock()
        gateway = AsyncMock(return_value=_gateway_decision("direct", "direct"))
        with (
            patch.object(main, "get_store", return_value=_FakeChatStore(catalog)),
            patch.object(main, "classify_chat_route", gateway),
            patch.object(main, "resolve_rag_target", target),
        ):
            prepared = await main.prepare_chat(
                main.ChatRequest(
                    message="Summarize this document.",
                    document_ids=["doc-1"],
                    document_scope="explicit",
                )
            )

        target.assert_not_awaited()
        gateway.assert_not_awaited()
        self.assertNotEqual(prepared.route_used, "vector_rag_ambiguous_document")
        self.assertEqual(prepared.debug["document_resolution"]["resolved_document_ids"], ["doc-1"])
        self.assertEqual(prepared.debug["route_gateway"]["reason"], "explicit_current_turn_document_scope")

    async def test_intent_failure_does_not_fall_back_to_semantic_qa(self) -> None:
        catalog = [
            {"source_file": "reading.pdf", "document_ids": ["doc-1"], "mime_types": ["application/pdf"]}
        ]
        failed_intent = IntentClassifierDecision(
            intent="undetermined",
            attempts=2,
            duration_seconds=0.1,
            raw_output_preview="",
            fallback_reason="empty_response",
        )
        with (
            patch.object(main, "get_store", return_value=_FakeChatStore(catalog)),
            patch.object(main, "classify_rag_intent", AsyncMock(return_value=failed_intent)),
        ):
            prepared = await main.prepare_chat(
                main.ChatRequest(
                    message="Tóm tắt tài liệu này.",
                    document_ids=["doc-1"],
                    document_scope="explicit",
                )
            )

        self.assertEqual(prepared.route_used, "intent_undetermined")
        self.assertEqual(prepared.static_response, main.INTENT_UNDETERMINED_RESPONSE)
        self.assertEqual(prepared.sources, [])

    async def test_chat_converts_gateway_failure_to_502(self) -> None:
        failure = main.OllamaRequestError("empty_response", "router returned no content")
        with patch.object(main, "prepare_chat", AsyncMock(side_effect=failure)):
            with self.assertRaises(main.HTTPException) as raised:
                await main.chat(main.ChatRequest(message="xin chào"))

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.detail["ollama"]["kind"], "empty_response")

    async def test_chat_preserves_explicit_http_errors(self) -> None:
        failure = main.HTTPException(status_code=409, detail="conflict")
        with patch.object(main, "prepare_chat", AsyncMock(side_effect=failure)):
            with self.assertRaises(main.HTTPException) as raised:
                await main.chat(main.ChatRequest(message="xin chào"))

        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(raised.exception.detail, "conflict")

    def test_evidence_query_prefers_child_question_and_removes_options(self) -> None:
        sources = [
            {
                "display_text": "Questions 11-13 Choose the correct letter. 11. Vintage wines are A mostly better. B often preferred. C often discussed.",
                "metadata": {"unit_type": "question_group"},
            },
            {
                "display_text": "11. Vintage wines are A mostly better. B often preferred. C often discussed. D more costly.",
                "metadata": {"unit_type": "question"},
            },
        ]

        query = main.evidence_query_for_sources(sources, "Trả lời Question 11")

        self.assertEqual(query, "Vintage wines are")

    def test_context_assigns_roles_without_source_prompt_tokens(self) -> None:
        context = main.format_context(
            [
                {
                    "source_file": "reading.pdf",
                    "pages": [2],
                    "display_text": "11. Vintage wines are...",
                    "metadata": {"unit_type": "question"},
                },
                {
                    "source_file": "reading.pdf",
                    "pages": [1, 2],
                    "display_text": "Passage evidence",
                    "metadata": {"unit_type": "passage"},
                },
            ]
        )

        self.assertIn("--- QUESTION 1 ---", context)
        self.assertIn("--- PASSAGE EVIDENCE 2 ---", context)
        self.assertNotIn("[Source", context)

    def test_mixed_document_overview_is_not_reduced_to_writing_inventory(self) -> None:
        sources = [
            {
                "document_id": "reading-doc",
                "source_file": "reading.pdf",
                "text": "Passage 1",
                "metadata": {"unit_type": "passage", "passage_number": 1},
            },
            {
                "document_id": "writing-doc",
                "source_file": "writing.pdf",
                "text": "IELTS Writing Task 1",
                "metadata": {"unit_type": "writing_task", "section_id": "task-1-task"},
            },
        ]

        response = main.static_response_for_sources(
            "Tóm tắt toàn bộ tài liệu đã tải.",
            "document_overview",
            sources,
        )

        self.assertIsNone(response)

    async def test_upload_connects_processor_chunks_and_store(self) -> None:
        store = _FakeStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            upload = UploadFile(
                file=BytesIO(b"sample content"),
                filename="sample.txt",
                headers=Headers({"content-type": "text/plain"}),
            )
            with (
                patch.object(main, "UPLOAD_DIR", Path(temp_dir)),
                patch.object(main, "DOCUMENT_PROCESSOR", _FakeProcessor()),
                patch.object(main, "get_store", return_value=store),
            ):
                response = await main.upload_document(upload)

            self.assertEqual(response.document_id, "doc-1")
            self.assertEqual(response.document_type, "ielts_reading")
            self.assertEqual(response.chunks_processed, 1)
            self.assertEqual(store.received_chunks[0]["metadata"]["parser_version"], "1.10.0")
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    async def test_document_query_without_sources_does_not_fall_back_to_base_model(self) -> None:
        catalog = [
            {
                "source_file": "sample.pdf",
                "document_ids": ["doc-1"],
                "mime_types": ["application/pdf"],
            }
        ]
        with (
            patch.object(main, "get_store", return_value=_FakeChatStore(catalog)),
            patch.object(
                main,
                "classify_chat_route",
                AsyncMock(return_value=_gateway_decision("rag", "show_questions", document_refs=("D1",))),
            ),
        ):
            prepared = await main.prepare_chat(
                main.ChatRequest(
                    message="Nội dung Questions 1-4 trong sample.pdf là gì?",
                    document_ids=["doc-1"],
                )
            )

        self.assertEqual(prepared.route_used, "vector_rag_no_match")
        self.assertIsNone(prepared.prompt)
        self.assertEqual(prepared.static_response, main.NO_RAG_MATCH_RESPONSE)

    async def test_explicit_document_query_uses_semantic_gateway(self) -> None:
        catalog = [
            {
                "source_file": "sample.pdf",
                "document_ids": ["doc-1"],
                "mime_types": ["application/pdf"],
            }
        ]
        gateway = AsyncMock(
            return_value=_gateway_decision("rag", "semantic_qa", document_refs=("D1",))
        )
        with (
            patch.object(main, "get_store", return_value=_FakeChatStore(catalog)),
            patch.object(main, "classify_chat_route", gateway),
        ):
            prepared = await main.prepare_chat(
                main.ChatRequest(
                    message="Trong tài liệu có nói gì về Mars?",
                    document_ids=["doc-1"],
                )
            )

        self.assertEqual(prepared.route_used, "vector_rag_no_match")
        gateway.assert_awaited_once()

    async def test_general_question_without_document_scope_routes_directly(self) -> None:
        catalog = [
            {
                "source_file": "reading.pdf",
                "document_ids": ["doc-1"],
                "mime_types": ["application/pdf"],
            }
        ]
        store = _FakeChatStore(catalog, has_document_intent=True)

        with (
            patch.object(main, "get_store", return_value=store),
            patch.object(
                main,
                "classify_chat_route",
                AsyncMock(
                    side_effect=[
                        _gateway_decision("direct", "direct", reason="greeting"),
                        _gateway_decision("direct", "direct", reason="general advice"),
                    ]
                ),
            ),
        ):
            greeting = await main.prepare_chat(
                main.ChatRequest(message="xin chào", document_ids=["doc-1"])
            )
            advice = await main.prepare_chat(
                main.ChatRequest(
                    message="Give me 3 IELTS Speaking Part 2 tips.",
                    document_ids=["doc-1"],
                )
            )

        self.assertEqual(greeting.route_used, "base_model")
        self.assertEqual(greeting.query_intent, "direct")
        self.assertIsNone(greeting.static_response)
        self.assertIn("Current user message:\nxin chào", greeting.prompt)
        self.assertEqual(advice.route_used, "base_model")
        self.assertEqual(advice.query_intent, "direct")
        self.assertIsNone(advice.static_response)
        self.assertIn("why it helps", advice.prompt)
        self.assertEqual(store.probe_dense_flags, [])

    async def test_generic_ielts_categories_do_not_trigger_document_ambiguity(self) -> None:
        catalog = [
            {
                "source_file": "IELTS READING TEST 2.pdf",
                "document_ids": ["doc-reading"],
                "mime_types": ["application/pdf"],
            },
            {
                "source_file": "IELTS Task 1 Essay.pdf",
                "document_ids": ["doc-writing"],
                "mime_types": ["application/pdf"],
            },
        ]
        store = _FakeChatStore(catalog, has_document_intent=True)
        gateway = AsyncMock(
            return_value=_gateway_decision(
                "direct",
                "direct",
                reason="general advice",
            )
        )

        with (
            patch.object(main, "get_store", return_value=store),
            patch.object(main, "classify_chat_route", gateway),
        ):
            for message in [
                "How can I improve my IELTS Reading speed?",
                "Explain TRUE/FALSE/NOT GIVEN in IELTS Reading.",
                "Give me an IELTS Writing Task 2 discussion essay structure.",
            ]:
                prepared = await main.prepare_chat(
                    main.ChatRequest(
                        message=message,
                        document_ids=["doc-reading", "doc-writing"],
                    )
                )
                self.assertEqual(prepared.route_used, "base_model")
                self.assertIsNone(prepared.static_response)
                self.assertIn(message, prepared.prompt)
                self.assertTrue(prepared.debug["document_resolution"]["skipped"])

        self.assertEqual(gateway.await_count, 3)

    async def test_structured_title_query_is_confirmed_by_semantic_gateway(self) -> None:
        catalog = [
            {
                "source_file": "reading-a.pdf",
                "document_ids": ["doc-a"],
                "mime_types": ["application/pdf"],
                "section_titles": ["Snow-makers"],
            },
            {
                "source_file": "reading-b.pdf",
                "document_ids": ["doc-b"],
                "mime_types": ["application/pdf"],
                "section_titles": ["Painters of Time"],
            },
        ]
        store = _FakeChatStore(catalog, has_document_intent=True)
        gateway = AsyncMock(
            return_value=_gateway_decision("rag", "semantic_qa", document_refs=("D1",))
        )

        with (
            patch.object(main, "get_store", return_value=store),
            patch.object(main, "classify_chat_route", gateway),
        ):
            prepared = await main.prepare_chat(
                main.ChatRequest(
                    message="What is Snow-makers about?",
                    document_ids=["doc-a", "doc-b"],
                )
            )

        gateway.assert_awaited_once()
        self.assertEqual(prepared.route_used, "vector_rag_no_match")
        self.assertEqual(
            prepared.debug["document_resolution"]["resolved_document_ids"],
            ["doc-a"],
        )

    async def test_follow_up_affinity_limits_retrieval_to_previous_document(self) -> None:
        catalog = [
            {"source_file": "reading-2.pdf", "document_ids": ["doc-2"], "mime_types": ["application/pdf"]},
            {"source_file": "reading-4.pdf", "document_ids": ["doc-4"], "mime_types": ["application/pdf"]},
        ]
        store = _FakeChatStore(catalog, has_document_intent=False)
        prepared = None
        with (
            patch.object(main, "get_store", return_value=store),
            patch.object(
                main,
                "classify_chat_route",
                AsyncMock(return_value=_gateway_decision("rag", "semantic_qa", reason="follow-up")),
            ),
        ):
            prepared = await main.prepare_chat(
                main.ChatRequest(
                    message="Tại sao?",
                    conversation_history=[
                        {"role": "user", "content": "Trả lời Question 4 trong Reading Test 2"}
                    ],
                    document_ids=["doc-2", "doc-4"],
                    affinity={
                        "document_ids": ["doc-2"],
                        "passage_numbers": [1],
                        "question_ranges": [[1, 4]],
                    },
                )
            )

        self.assertEqual(prepared.debug["target_resolution"]["method"], "conversation_affinity")
        self.assertTrue(all(ids == ["doc-2"] for ids in store.routing_document_ids))
        self.assertEqual(len(store.routing_queries), 1)
        self.assertIn("Trả lời Question 4", store.routing_queries[0])
        self.assertIn("Follow-up: Tại sao?", store.routing_queries[0])

    async def test_affinity_does_not_turn_a_new_direct_question_into_rag(self) -> None:
        catalog = [
            {"source_file": "reading-2.pdf", "document_ids": ["doc-2"], "mime_types": ["application/pdf"]},
            {"source_file": "reading-4.pdf", "document_ids": ["doc-4"], "mime_types": ["application/pdf"]},
        ]
        store = _FakeChatStore(catalog, has_document_intent=False)
        gateway = AsyncMock(
            return_value=_gateway_decision(
                "direct",
                "direct",
                reason="new general request",
            )
        )
        with (
            patch.object(main, "get_store", return_value=store),
            patch.object(main, "classify_chat_route", gateway),
        ):
            prepared = await main.prepare_chat(
                main.ChatRequest(
                    message="Give me three IELTS Speaking tips.",
                    conversation_history=[
                        {"role": "user", "content": "Trả lời Question 4 trong Reading Test 2"}
                    ],
                    document_ids=["doc-2", "doc-4"],
                    affinity={"document_ids": ["doc-2"]},
                )
            )

        self.assertEqual(prepared.route_used, "base_model")
        self.assertIsNone(prepared.static_response)
        self.assertIn("three IELTS Speaking tips", prepared.prompt)
        self.assertEqual(store.probe_queries, [])

    async def test_retrieval_score_does_not_bypass_gateway_for_direct_intent(self) -> None:
        catalog = [
            {"source_file": "reading.pdf", "document_ids": ["doc-1"], "mime_types": ["application/pdf"]}
        ]
        store = _FakeChatStore(
            catalog,
            has_document_intent=False,
            top_question_score=120.0,
        )
        gateway = AsyncMock(
            return_value=_gateway_decision(
                "direct",
                "direct",
                reason="general request",
            )
        )
        with (
            patch.object(main, "get_store", return_value=store),
            patch.object(main, "classify_chat_route", gateway),
        ):
            prepared = await main.prepare_chat(
                main.ChatRequest(message="Give me three IELTS Speaking tips.", document_ids=["doc-1"])
            )

        gateway.assert_awaited_once()
        self.assertEqual(prepared.route_used, "base_model")
        self.assertIsNone(prepared.static_response)
        self.assertIn("Give me three IELTS Speaking tips.", prepared.prompt)

    async def test_semantic_gateway_does_not_receive_catalog_or_retrieval_snippets(self) -> None:
        catalog = [
            {
                "source_file": "reading.pdf",
                "document_ids": ["doc-1"],
                "mime_types": ["application/pdf"],
                "document_types": ["ielts_reading"],
                "section_titles": ["Urban transport"],
            }
        ]
        store = _FakeChatStore(catalog)
        store.routing_candidates = [
            {
                "chunk_id": "passage-2",
                "document_id": "doc-1",
                "source_file": "reading.pdf",
                "pages": [3],
                "text": "Urban transport changed after the rail network expanded.",
                "metadata": {
                    "unit_type": "passage",
                    "passage_number": 2,
                    "passage_title": "Urban transport",
                },
            }
        ]
        gateway = AsyncMock(
            return_value=_gateway_decision(
                "rag",
                "semantic_qa",
                document_refs=("D1",),
                section_refs=("S1",),
            )
        )
        with (
            patch.object(main, "get_store", return_value=store),
            patch.object(main, "classify_chat_route", gateway),
        ):
            prepared = await main.prepare_chat(
                main.ChatRequest(message="Why did urban transport change?", document_ids=["doc-1"])
            )

        gateway_context = gateway.await_args.args[2]
        self.assertEqual(gateway_context, "")
        self.assertEqual(prepared.debug["document_resolution"]["resolved_document_ids"], ["doc-1"])
        self.assertEqual(store.routing_candidates, [
            {
                "chunk_id": "passage-2",
                "document_id": "doc-1",
                "source_file": "reading.pdf",
                "pages": [3],
                "text": "Urban transport changed after the rail network expanded.",
                "metadata": {
                    "unit_type": "passage",
                    "passage_number": 2,
                    "passage_title": "Urban transport",
                },
            }
        ])

    async def test_semantic_gateway_state_does_not_expose_document_references(self) -> None:
        catalog = [
            {"source_file": "reading.pdf", "document_ids": ["doc-1"], "mime_types": ["application/pdf"]}
        ]
        gateway = AsyncMock(return_value=_gateway_decision("direct", "direct"))
        with (
            patch.object(main, "get_store", return_value=_FakeChatStore(catalog)),
            patch.object(main, "classify_chat_route", gateway),
        ):
            await main.prepare_chat(
                main.ChatRequest(
                    message="Give me three IELTS tips.",
                    document_ids=["doc-1"],
                    conversation_state={
                        "last_route": "rag",
                        "last_intent": "semantic_qa",
                        "rag_affinity": {
                            "document_ids": ["doc-1"],
                            "passage_numbers": [2],
                            "question_ranges": [[14, 17]],
                        },
                    },
                )
            )

        state_context = gateway.await_args.args[2]
        self.assertIn('"last_route": "rag"', state_context)
        self.assertIn('"has_rag_affinity": true', state_context)
        self.assertNotIn("doc-1", state_context)
        self.assertNotIn("14", state_context)

    async def test_gateway_can_request_rag_with_an_explicit_document_scope(self) -> None:
        catalog = [
            {
                "source_file": "reading.pdf",
                "document_ids": ["doc-1"],
                "mime_types": ["application/pdf"],
            }
        ]
        store = _FakeChatStore(catalog, has_document_intent=False)
        with (
            patch.object(main, "get_store", return_value=store),
            patch.object(
                main,
                "classify_chat_route",
                AsyncMock(
                    return_value=_gateway_decision(
                        "rag",
                        "semantic_qa",
                        document_refs=("D1",),
                        reason="requires document facts",
                    )
                ),
            ),
        ):
            prepared = await main.prepare_chat(
                main.ChatRequest(
                    message="How did the fence affect kangaroos?",
                    document_ids=["doc-1"],
                )
            )

        self.assertEqual(prepared.route_used, "vector_rag_no_match")
        self.assertNotEqual(prepared.query_intent, "direct")
        self.assertFalse(prepared.debug["target_resolution"]["document_grounded"])
        self.assertEqual(prepared.debug["route_gateway"]["route"], "rag")
        self.assertEqual(store.routing_document_ids, [["doc-1"]])

    async def test_ambiguous_question_range_requests_a_document_choice(self) -> None:
        catalog = [
            {"source_file": "reading-2.pdf", "document_ids": ["doc-2"], "mime_types": ["application/pdf"]},
            {"source_file": "reading-4.pdf", "document_ids": ["doc-4"], "mime_types": ["application/pdf"]},
        ]
        with (
            patch.object(main, "get_store", return_value=_FakeChatStore(catalog)),
            patch.object(
                main,
                "classify_chat_route",
                AsyncMock(
                    return_value=_gateway_decision(
                        "clarify",
                        "show_questions",
                    )
                ),
            ),
        ):
            prepared = await main.prepare_chat(main.ChatRequest(message="Liệt kê Questions 1-4"))

        self.assertEqual(prepared.route_used, "vector_rag_ambiguous_document")
        self.assertIn("Vui lòng nêu tên file", prepared.static_response)

    async def test_static_table_operations_do_not_collapse_to_first_cell(self) -> None:
        source = {
            "source_file": "writing.png",
            "pages": [1],
            "metadata": {
                "unit_type": "writing_table",
                "table": {
                    "columns": [
                        "Country",
                        "Internet Access 2019 (%)",
                        "Internet Access 2024 (%)",
                        "Smartphone Ownership 2019 (%)",
                        "Smartphone Ownership 2024 (%)",
                    ],
                    "rows": [
                        ["A", 78, 96, 82, 99],
                        ["B", 61, 89, 67, 94],
                        ["C", 42, 75, 48, 83],
                    ],
                },
            },
        }

        cell = main.static_response_for_sources(
            "Smartphone Ownership của Country B năm 2024 là bao nhiêu?",
            "table_cell",
            [source],
        )
        calculation = main.static_response_for_sources(
            "Từ bảng, quốc gia nào tăng Internet Access nhiều nhất từ 2019 đến 2024? Trình bày phép tính.",
            "table_calculation",
            [source],
        )
        comparison = main.static_response_for_sources(
            "So sánh Internet Access và Smartphone Ownership của Country A trong cả hai năm.",
            "table_comparison",
            [source],
        )

        self.assertTrue(cell.startswith("94"))
        self.assertIn("A: 96 - 78 = 18", calculation)
        self.assertIn("B: 89 - 61 = 28", calculation)
        self.assertIn("C: 75 - 42 = 33", calculation)
        self.assertIn("| A | 78 | 96 | 82 | 99 |", comparison)

    def test_solve_context_rejects_missing_multiple_choice_options(self) -> None:
        incomplete = [
            {
                "text": "From the list below choose the most suitable title.",
                "metadata": {"unit_type": "question"},
            }
        ]
        complete = [
            {
                "text": "Choose the correct letter. A first title B second title C third title D fourth title",
                "metadata": {"unit_type": "question"},
            }
        ]

        self.assertEqual(main.solve_context_issue(incomplete), "missing_answer_options")
        self.assertIsNone(main.solve_context_issue(complete))

    async def test_solve_generation_uses_one_grounded_model_call(self) -> None:
        prepared = main.ChatPreparation(
            prompt="grounded solve prompt",
            static_response=None,
            route_used="vector_rag",
            sources=[],
            debug={"intent_decision": {"allow_solution": True}},
            query_intent="solve_questions",
        )
        model = AsyncMock(return_value="Đáp án C vì passage nêu trực tiếp chi tiết này.")

        with patch.object(main, "query_ollama", model):
            answer = await main.generate_answer(prepared, "Trả lời Question 11.")

        self.assertEqual(answer, "Đáp án C vì passage nêu trực tiếp chi tiết này.")
        self.assertEqual(model.await_count, 1)
        self.assertFalse(main.requires_reviewed_generation(prepared, "Trả lời Question 11."))

    async def test_solve_does_not_fall_back_to_semantic_passage_without_exact_question(self) -> None:
        catalog = [
            {
                "source_file": "reading.pdf",
                "document_ids": ["doc-1"],
                "mime_types": ["application/pdf"],
            }
        ]

        class StrongPassageStore(_FakeChatStore):
            def probe_with_catalog(self, query, top_k, document_ids=None, include_dense=True):
                passage = {
                    "document_id": "doc-1",
                    "source_file": "reading.pdf",
                    "text": "Semantically related passage text.",
                    "metadata": {"unit_type": "passage", "passage_number": 1},
                }
                return (
                    {
                        "results": [passage],
                        "has_hits": True,
                        "has_strong_hits": True,
                        "has_document_intent": True,
                        "is_overview": False,
                        "top_question_score": 0.0,
                    },
                    self.document_catalog(document_ids),
                )

        store = StrongPassageStore(catalog)
        with (
            patch.object(main, "get_store", return_value=store),
            patch.object(
                main,
                "classify_chat_route",
                AsyncMock(return_value=_gateway_decision("rag", "solve_questions")),
            ),
            patch.object(
                main,
                "classify_rag_intent",
                AsyncMock(
                    return_value=IntentClassifierDecision(
                        intent="solve_questions",
                        attempts=1,
                        duration_seconds=0.01,
                        raw_output_preview="solve_questions",
                    )
                ),
            ),
        ):
            prepared = await main.prepare_chat(
                main.ChatRequest(
                    message="Trả lời Question 11 và giải thích.",
                    document_ids=["doc-1"],
                )
            )

        self.assertEqual(prepared.route_used, "vector_rag_no_match")
        self.assertEqual(prepared.debug["retrieval"]["method"], "structured_question_no_match")
        self.assertEqual(prepared.debug["retrieval"]["structured_question_units"], 0)
        self.assertEqual(prepared.sources, [])

    def test_direct_generation_is_buffered_for_output_validation(self) -> None:
        prepared = main.ChatPreparation(
            prompt="direct plan prompt",
            static_response=None,
            route_used="base_model",
            sources=[],
            debug={},
            query_intent="direct",
        )

        self.assertTrue(main.requires_reviewed_generation(prepared, "Lập kế hoạch học trong 3 tháng."))

    def test_plain_direct_generation_uses_true_streaming(self) -> None:
        prepared = main.ChatPreparation(
            prompt="direct tips prompt",
            static_response=None,
            route_used="base_model",
            sources=[],
            debug={},
            query_intent="direct",
        )

        self.assertFalse(main.requires_reviewed_generation(prepared, "Cho tôi 3 tips học IELTS."))
        self.assertFalse(main.requires_reviewed_generation(prepared, "haha"))

    async def test_invalid_direct_stream_prefix_falls_back_to_reviewed_generation(self) -> None:
        prepared = main.ChatPreparation(
            prompt="direct conversation prompt",
            static_response=None,
            route_used="base_model",
            sources=[],
            debug={},
            query_intent="direct",
        )

        async def invalid_stream(*args, **kwargs):
            raise main.OllamaRequestError("role_prefix", "invalid role prefix")
            yield ""  # pragma: no cover

        with (
            patch.object(main, "prepare_chat", AsyncMock(return_value=prepared)),
            patch.object(main, "stream_ollama", invalid_stream),
            patch.object(main, "generate_answer", AsyncMock(return_value="Mình đang nghe đây.")) as fallback,
        ):
            response = await main.chat_stream(main.ChatRequest(message="haha"))
            events = [json.loads(chunk) async for chunk in response.body_iterator]

        self.assertEqual(
            [event["token"] for event in events if event["type"] == "token"],
            ["Mình đang nghe đây."],
        )
        self.assertEqual(events[-1]["type"], "done")
        fallback.assert_awaited_once_with(prepared, "haha")

    async def test_direct_generation_retries_a_multiline_markdown_table(self) -> None:
        prepared = main.ChatPreparation(
            prompt="direct plan prompt",
            static_response=None,
            route_used="base_model",
            sources=[],
            debug={},
            query_intent="direct",
        )
        malformed = """| Giai đoạn | Hoạt động | Thời lượng |
| --- | --- | --- |
| Tuần 1-4 | - Luyện nghe
- Luyện đọc | 60 phút |"""
        valid = """| Giai đoạn | Hoạt động | Thời lượng |
| --- | --- | --- |
| Tuần 1-4 | Luyện nghe; luyện đọc | 60 phút |"""
        model = AsyncMock(side_effect=[malformed, valid])

        with patch.object(main, "query_ollama", model):
            answer = await main.generate_answer(prepared, "Lập kế hoạch học trong 3 tháng.")

        self.assertEqual(answer, valid)
        self.assertEqual(model.await_count, 2)
        self.assertTrue(prepared.debug["generation"]["retry_used"])

    async def test_direct_generation_retries_a_role_prefixed_echo(self) -> None:
        prepared = main.ChatPreparation(
            prompt="direct conversation prompt",
            static_response=None,
            route_used="base_model",
            sources=[],
            debug={},
            query_intent="direct",
        )
        model = AsyncMock(side_effect=["User: haha", "Mình đang nghe đây."])

        with patch.object(main, "query_ollama", model):
            answer = await main.generate_answer(prepared, "haha")

        self.assertEqual(answer, "Mình đang nghe đây.")
        self.assertEqual(model.await_count, 2)
        self.assertTrue(prepared.debug["generation"]["retry_used"])

    async def test_writing_generation_rewrites_wrong_language_and_length(self) -> None:
        prepared = main.ChatPreparation(
            prompt="grounded writing prompt",
            static_response=None,
            route_used="vector_rag",
            sources=[],
            debug={"intent_decision": {"allow_solution": True}},
            query_intent="writing_generation",
        )
        corrected = " ".join(["word"] * 175)
        model = AsyncMock(
            side_effect=[
                "Bảng cho thấy các quốc gia đều tăng đáng kể.",
                corrected,
            ]
        )

        with patch.object(main, "query_ollama", model):
            answer = await main.generate_answer(
                prepared,
                "Viết bài IELTS Writing Task 1 dài 170-190 từ.",
            )

        self.assertEqual(answer, corrected)
        self.assertEqual(model.await_count, 2)
        self.assertEqual(prepared.debug["generation"]["final_issues"], [])
        retry_prompt = model.await_args_list[1].args[0]
        self.assertNotIn("Bảng cho thấy", retry_prompt)
        self.assertNotIn("previous draft", retry_prompt.lower())
        self.assertNotIn("below 170", retry_prompt.lower())
        self.assertEqual(prepared.debug["generation"]["selected_candidate"], "retry")

    async def test_writing_generation_fails_closed_when_both_candidates_are_invalid(self) -> None:
        prepared = main.ChatPreparation(
            prompt="grounded writing prompt",
            static_response=None,
            route_used="vector_rag",
            sources=[],
            debug={"intent_decision": {"allow_solution": True}},
            query_intent="writing_generation",
        )
        first = " ".join(["word"] * 165)
        retry = "Here is the revised essay: " + " ".join(["word"] * 175)
        model = AsyncMock(side_effect=[first, retry])

        with patch.object(main, "query_ollama", model):
            answer = await main.generate_answer(
                prepared,
                "Viết bài IELTS Writing Task 1 dài 170-190 từ.",
            )

        self.assertEqual(answer, main.WRITING_VALIDATION_FAILURE_RESPONSE)
        self.assertEqual(model.await_count, 2)
        self.assertEqual(prepared.debug["generation"]["selected_candidate"], "first")
        self.assertTrue(prepared.debug["generation"]["final_issues"])
        self.assertTrue(prepared.debug["generation"]["fail_closed"])

    async def test_translation_fails_closed_when_retry_is_still_incomplete(self) -> None:
        prepared = main.ChatPreparation(
            prompt="grounded translation prompt",
            static_response=None,
            route_used="vector_rag",
            sources=[],
            debug={"intent_decision": {"allow_solution": False}},
            query_intent="translate_questions",
        )
        model = AsyncMock(side_effect=["Question one only.", "Chỉ có câu một."])

        with patch.object(main, "query_ollama", model):
            answer = await main.generate_answer(prepared, "Dịch Questions 1-4 sang tiếng Việt.")

        self.assertEqual(answer, main.TRANSLATION_VALIDATION_FAILURE_RESPONSE)
        self.assertEqual(model.await_count, 2)
        self.assertTrue(prepared.debug["generation"]["fail_closed"])

    async def test_markdown_table_fails_closed_when_retry_remains_malformed(self) -> None:
        prepared = main.ChatPreparation(
            prompt="direct plan prompt",
            static_response=None,
            route_used="base_model",
            sources=[],
            debug={},
            query_intent="direct",
        )
        malformed = """| Giai đoạn | Hoạt động |
| --- | --- |
| Tuần 1-4 | - Luyện nghe
- Luyện đọc |"""
        model = AsyncMock(side_effect=[malformed, malformed])

        with patch.object(main, "query_ollama", model):
            answer = await main.generate_answer(prepared, "Lập kế hoạch học trong 3 tháng.")

        self.assertEqual(answer, main.FORMAT_VALIDATION_FAILURE_RESPONSE)
        self.assertEqual(model.await_count, 2)
        self.assertTrue(prepared.debug["generation"]["fail_closed"])

    async def test_writing_semantic_answer_defaults_to_english(self) -> None:
        prepared = main.ChatPreparation(
            prompt="grounded writing analysis prompt",
            static_response=None,
            route_used="vector_rag",
            sources=[],
            debug={
                "intent_decision": {"allow_solution": True},
                "retrieval": {"writing_parent_id": "writing-task-2"},
            },
            query_intent="semantic_qa",
        )
        model = AsyncMock(
            side_effect=[
                "Bài viết mô tả xu hướng tăng rõ rệt trong giai đoạn này.",
                "The paragraph describes a clear upward trend over the period.",
            ]
        )

        with patch.object(main, "query_ollama", model):
            answer = await main.generate_answer(prepared, "Describe the main trend in this Writing task.")

        self.assertEqual(answer, "The paragraph describes a clear upward trend over the period.")
        self.assertEqual(model.await_count, 2)

    async def test_explicit_no_solution_request_is_rewritten_without_blocking(self) -> None:
        prepared = main.ChatPreparation(
            prompt="grounded explanation prompt",
            static_response=None,
            route_used="vector_rag",
            sources=[],
            debug={"intent_decision": {"allow_solution": False}},
            query_intent="explain_questions",
        )
        model = AsyncMock(side_effect=["24: A", "Đối chiếu từng phát biểu với mô tả của hai phương pháp."])

        with patch.object(main, "query_ollama", model):
            answer = await main.generate_answer(
                prepared,
                "Giải thích Questions 24-27 nhưng không chọn đáp án.",
            )

        self.assertEqual(answer, "Đối chiếu từng phát biểu với mô tả của hai phương pháp.")
        self.assertEqual(model.await_count, 2)
        retry_prompt = model.await_args_list[1].args[0]
        self.assertIn("Do not select, infer, eliminate, or hint", retry_prompt)
        self.assertNotIn("24: A", retry_prompt)

    async def test_no_solution_matching_answer_is_rewritten(self) -> None:
        prepared = main.ChatPreparation(
            prompt="grounded explanation prompt",
            static_response=None,
            route_used="vector_rag",
            sources=[],
            debug={"intent_decision": {"allow_solution": False}},
            query_intent="explain_questions",
        )
        model = AsyncMock(
            side_effect=[
                "Câu 36 phù hợp với A Levitin.",
                "Đối chiếu từ khóa trong từng phát biểu với quan điểm của mỗi nhà khoa học.",
            ]
        )

        with patch.object(main, "query_ollama", model):
            answer = await main.generate_answer(
                prepared,
                "Giải thích cách làm Questions 36-40 nhưng không giải.",
            )

        self.assertEqual(
            answer,
            "Đối chiếu từ khóa trong từng phát biểu với quan điểm của mỗi nhà khoa học.",
        )
        self.assertEqual(model.await_count, 2)

    async def test_compliant_no_solution_response_is_not_generated_twice(self) -> None:
        prepared = main.ChatPreparation(
            prompt="grounded explanation prompt",
            static_response=None,
            route_used="vector_rag",
            sources=[],
            debug={"intent_decision": {"allow_solution": False}},
            query_intent="explain_questions",
        )
        model = AsyncMock(return_value="Đối chiếu từng phát biểu với thông tin trong passage.")

        with patch.object(main, "query_ollama", model):
            answer = await main.generate_answer(
                prepared,
                "Giải thích Questions 24-27 nhưng không chọn đáp án.",
            )

        self.assertEqual(answer, "Đối chiếu từng phát biểu với thông tin trong passage.")
        self.assertEqual(model.await_count, 1)

    async def test_translation_retries_with_language_and_range_contract(self) -> None:
        prepared = main.ChatPreparation(
            prompt="grounded translation prompt",
            static_response=None,
            route_used="vector_rag",
            sources=[],
            debug={"intent_decision": {"allow_solution": False}},
            query_intent="translate_questions",
        )
        translated = (
            "25. Cơ quan nào cung cấp số liệu du lịch toàn cầu?\n"
            "26. Ai thường được hưởng lợi về tài chính?\n"
            "27. Cuộc họp nào cung cấp các nguyên tắc?"
        )
        model = AsyncMock(
            side_effect=[
                "25. Which body provides global tourist numbers?",
                translated,
            ]
        )

        with patch.object(main, "query_ollama", model):
            answer = await main.generate_answer(
                prepared,
                "Dịch Questions 25-27 sang tiếng Việt, chưa trả lời.",
            )

        self.assertEqual(answer, translated)
        self.assertEqual(model.await_count, 2)
        self.assertEqual(prepared.debug["generation"]["final_issues"], [])

    async def test_no_solution_uses_safe_fallback_when_retry_still_leaks(self) -> None:
        prepared = main.ChatPreparation(
            prompt="grounded explanation prompt",
            static_response=None,
            route_used="vector_rag",
            sources=[],
            debug={"intent_decision": {"allow_solution": False}},
            query_intent="explain_questions",
        )
        model = AsyncMock(side_effect=["24: A", "25: B"])

        with patch.object(main, "query_ollama", model):
            answer = await main.generate_answer(
                prepared,
                "Giải thích Questions 24-27 nhưng không chọn đáp án.",
            )

        self.assertIn("chưa chọn hoặc loại trừ", answer)
        self.assertTrue(prepared.debug["generation"]["safe_fallback_used"])
        self.assertEqual(prepared.debug["generation"]["final_issues"], [])


if __name__ == "__main__":
    unittest.main()
