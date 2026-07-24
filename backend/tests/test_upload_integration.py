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
    def test_intent_candidates_exclude_question_actions_without_question_target(self) -> None:
        candidates = main.allowed_rag_intents(
            "Why does the author reject this idea?",
            [{"unit_types": ["passage", "question", "question_group"]}],
            None,
        )

        self.assertIn("semantic_qa", candidates)
        self.assertNotIn("solve_questions", candidates)
        self.assertNotIn("explain_questions", candidates)

    def test_intent_candidates_include_question_actions_for_range_or_affinity(self) -> None:
        explicit = main.allowed_rag_intents(
            "Explain Questions 11-13 without answering.",
            [{"unit_types": ["passage", "question", "question_group"]}],
            None,
        )
        follow_up = main.allowed_rag_intents(
            "Explain them without answering.",
            [{"unit_types": ["passage", "question", "question_group"]}],
            main.ChatAffinity(question_ranges=[[11, 13]]),
        )

        self.assertIn("explain_questions", explicit)
        self.assertIn("explain_questions", follow_up)
        self.assertIn("solve_questions", explicit)

    def test_intent_candidates_require_structured_table_for_table_operations(self) -> None:
        without_table = main.allowed_rag_intents(
            "Compare the facts discussed in this sample answer.",
            [{"unit_types": ["writing_task", "sample_answer"]}],
            None,
        )
        with_table = main.allowed_rag_intents(
            "Compare the values in this table.",
            [{"unit_types": ["writing_prompt", "writing_table", "table_row"]}],
            None,
        )

        self.assertIn("semantic_qa", without_table)
        self.assertNotIn("table_comparison", without_table)
        self.assertIn("table_comparison", with_table)
        self.assertIn("show_table", without_table)

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

    def test_chat_stream_is_the_only_chat_endpoint(self) -> None:
        paths = {route.path for route in main.app.routes}

        self.assertIn("/chat/stream", paths)
        self.assertNotIn("/chat", paths)
        self.assertIn("/documents/upload", paths)
        self.assertNotIn("/rag/upload-pdf", paths)

    async def test_chat_stream_uses_completed_preparation(self) -> None:
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

    async def test_empty_direct_stream_falls_back_to_chat_api(self) -> None:
        prepared = main.ChatPreparation(
            prompt="direct generate prompt",
            static_response=None,
            route_used="base_model",
            sources=[],
            debug={"direct_generation": {"fallback_used": False}},
            query_intent="direct",
        )

        async def empty_stream(*args, **kwargs):
            if False:
                yield ""

        chat_fallback = AsyncMock(return_value="Chào bạn.")
        generate_fallback = AsyncMock(return_value="wrong endpoint")
        request = main.ChatRequest(
            message="xin chào",
            conversation_history=[{"role": "user", "content": "hi"}],
        )
        with (
            patch.object(main, "prepare_chat", AsyncMock(return_value=prepared)),
            patch.object(main, "stream_ollama", empty_stream),
            patch.object(main, "query_ollama_chat", chat_fallback),
            patch.object(main, "query_ollama", generate_fallback),
        ):
            response = await main.chat_stream(request)
            events = [json.loads(chunk) async for chunk in response.body_iterator]

        self.assertEqual(
            [event["token"] for event in events if event["type"] == "token"],
            ["Chào bạn."],
        )
        chat_fallback.assert_awaited_once()
        generate_fallback.assert_not_awaited()
        messages = chat_fallback.await_args.args[0]
        self.assertEqual(messages[-1], {"role": "user", "content": "xin chào"})
        metadata = [event for event in events if event["type"] == "metadata"][-1]
        self.assertEqual(metadata["debug"]["direct_generation"]["fallback_endpoint"], "chat")
        self.assertEqual(metadata["debug"]["direct_generation"]["fallback_status"], "succeeded")

    async def test_empty_rag_stream_keeps_generate_fallback(self) -> None:
        prepared = main.ChatPreparation(
            prompt="grounded RAG prompt",
            static_response=None,
            route_used="vector_rag",
            sources=[],
            debug={},
            query_intent="semantic_qa",
        )

        async def empty_stream(*args, **kwargs):
            if False:
                yield ""

        chat_fallback = AsyncMock(return_value="wrong endpoint")
        generate_fallback = AsyncMock(return_value="Grounded answer.")
        with (
            patch.object(main, "prepare_chat", AsyncMock(return_value=prepared)),
            patch.object(main, "stream_ollama", empty_stream),
            patch.object(main, "query_ollama_chat", chat_fallback),
            patch.object(main, "query_ollama", generate_fallback),
        ):
            response = await main.chat_stream(main.ChatRequest(message="Passage nói gì?"))
            events = [json.loads(chunk) async for chunk in response.body_iterator]

        self.assertEqual(
            [event["token"] for event in events if event["type"] == "token"],
            ["Grounded answer."],
        )
        generate_fallback.assert_awaited_once_with("grounded RAG prompt", temperature=0.2)
        chat_fallback.assert_not_awaited()

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
            },
        )

        state = main.conversation_state_for_result(request, prepared)

        self.assertEqual(state.last_route, "direct")
        self.assertEqual(state.rag_affinity.document_ids, ["doc-1"])
        self.assertEqual(state.rag_affinity.passage_numbers, [2])

    async def test_gateway_failure_without_document_basis_streams_safe_result(self) -> None:
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
            response = await main.chat_stream(main.ChatRequest(message="Tell me something useful."))
            events = [json.loads(chunk) async for chunk in response.body_iterator]

        metadata = next(event for event in events if event["type"] == "metadata")
        self.assertEqual(metadata["route_used"], "route_undetermined")
        self.assertEqual(metadata["conversation_state"]["last_route"], "no_match")
        self.assertEqual(metadata["sources"], [])
        self.assertTrue(any(event["type"] == "done" for event in events))

    async def test_single_explicit_document_still_uses_direct_rag_gateway(self) -> None:
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
        gateway.assert_awaited_once()
        self.assertEqual(prepared.route_used, "base_model")
        self.assertEqual(prepared.debug["route_decision"], "direct")

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
            patch.object(
                main,
                "classify_chat_route",
                AsyncMock(return_value=_gateway_decision("rag", "semantic_qa")),
            ),
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

    async def test_chat_stream_reports_gateway_failure(self) -> None:
        failure = main.OllamaRequestError("empty_response", "router returned no content")
        with patch.object(main, "prepare_chat", AsyncMock(side_effect=failure)):
            response = await main.chat_stream(main.ChatRequest(message="xin chào"))
            events = [json.loads(chunk) async for chunk in response.body_iterator]

        error = next(event for event in events if event["type"] == "error")
        self.assertEqual(error["detail"]["ollama"]["kind"], "empty_response")

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

    async def test_resolver_selected_affinity_document_limits_follow_up_retrieval(self) -> None:
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
            patch.object(
                main,
                "resolve_rag_target",
                AsyncMock(
                    return_value=TargetResolverDecision(
                        document_refs=("D1",),
                        action="selected",
                        attempts=1,
                        duration_seconds=0.01,
                        raw_output_preview='{"action":"selected","document_refs":["D1"]}',
                    )
                ),
            ),
        ):
            prepared = await main.prepare_chat(
                main.ChatRequest(
                    message="Tại sao?",
                    conversation_history=[
                        {"role": "user", "content": "Trả lời Question 4 trong Reading Test 2"}
                    ],
                    conversation_state={
                        "last_route": "rag",
                        "last_intent": "solve_questions",
                        "rag_affinity": {
                            "document_ids": ["doc-2"],
                            "passage_numbers": [1],
                            "question_ranges": [[1, 4]],
                        },
                    },
                )
            )

        self.assertEqual(
            prepared.debug["document_resolution"]["method"],
            "semantic_target_with_affinity",
        )
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
                    conversation_state={
                        "last_route": "rag",
                        "last_intent": "solve_questions",
                        "rag_affinity": {"document_ids": ["doc-2"]},
                    },
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

    async def test_semantic_gateway_receives_bounded_catalog_but_not_retrieval_snippets(self) -> None:
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
                main.ChatRequest(
                    message="Why did urban transport change?",
                    document_ids=["doc-1"],
                    document_scope="explicit",
                )
            )

        state_context = gateway.await_args.args[2]
        document_context = gateway.await_args.args[3]
        self.assertEqual(state_context, "")
        self.assertIn("file=reading.pdf", document_context)
        self.assertIn("attached_this_turn=true", document_context)
        self.assertIn("sections=Urban transport", document_context)
        self.assertNotIn("rail network expanded", document_context)
        self.assertLessEqual(len(document_context), main.settings.route_catalog_chars)
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

    def test_target_catalog_context_includes_bounded_visual_descriptors(self) -> None:
        context = main.format_document_catalog_context(
            [
                {
                    "source_file": "writing.png",
                    "document_ids": ["doc-writing"],
                    "mime_types": ["image/png"],
                    "document_types": ["ielts_writing_task_1"],
                    "task_types": ["academic_task_1_table"],
                    "unit_types": ["writing_prompt", "writing_table"],
                    "section_titles": [],
                    "visual_types": ["table"],
                    "table_columns": [
                        "Country",
                        "Internet Access 2024",
                        "Smartphone Ownership 2024",
                    ],
                    "target_descriptors": [
                        "The table shows internet access and smartphone ownership in three countries."
                    ],
                }
            ]
        )

        self.assertEqual(context.document_refs, {"D1": "doc-writing"})
        self.assertIn("mime_types=image/png", context.text)
        self.assertIn("visual_types=table", context.text)
        self.assertIn("Smartphone Ownership 2024", context.text)
        self.assertIn("internet access and smartphone ownership", context.text)
        self.assertLessEqual(len(context.text), main.settings.target_catalog_chars)

    def test_route_catalog_context_is_bounded_per_document_and_in_total(self) -> None:
        catalog = [
            {
                "source_file": f"reading-{index}.pdf",
                "document_ids": [f"doc-{index}"],
                "document_types": ["ielts_reading"],
                "section_titles": ["A" * 1_000],
                "unit_types": ["passage", "question_group", "question"],
            }
            for index in range(20)
        ]

        context = main.format_route_catalog_context(catalog)

        self.assertLessEqual(len(context), main.settings.route_catalog_chars)
        self.assertIn("file=reading-0.pdf", context)
        document_lines = [line for line in context.splitlines() if line.startswith("- file=")]
        self.assertTrue(document_lines)
        self.assertTrue(
            all(len(line) <= main.settings.route_catalog_document_chars for line in document_lines)
        )

    def test_route_catalog_marks_only_documents_attached_in_current_turn(self) -> None:
        catalog = [
            {
                "source_file": "reading.pdf",
                "document_ids": ["doc-reading"],
                "document_types": ["ielts_reading"],
            },
            {
                "source_file": "writing.png",
                "document_ids": ["doc-writing"],
                "document_types": ["ielts_writing_task_1"],
            },
        ]

        context = main.format_route_catalog_context(catalog, ["doc-writing"])

        reading_line, writing_line = context.splitlines()
        self.assertNotIn("attached_this_turn", reading_line)
        self.assertIn("attached_this_turn=true", writing_line)

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
        document_context = gateway.await_args.args[3]
        self.assertIn('"last_route": "rag"', state_context)
        self.assertIn('"has_rag_affinity": true', state_context)
        self.assertNotIn("doc-1", state_context)
        self.assertNotIn("14", state_context)
        self.assertNotIn("attached_this_turn", document_context)

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
        model = AsyncMock(return_value="C because it is explicitly discussed.")

        with patch.object(main, "query_ollama", model):
            answer = await main.generate_answer(prepared, "Trả lời Question 11.")

        self.assertEqual(answer, "C because it is explicitly discussed.")
        self.assertEqual(model.await_count, 1)
        self.assertFalse(main.requires_reviewed_generation(prepared, "Trả lời Question 11."))

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

    async def test_writing_generation_keeps_best_non_meta_candidate_after_retry(self) -> None:
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

        self.assertEqual(answer, first)
        self.assertEqual(model.await_count, 2)
        self.assertEqual(prepared.debug["generation"]["selected_candidate"], "first")
        self.assertTrue(prepared.debug["generation"]["final_issues"])

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
