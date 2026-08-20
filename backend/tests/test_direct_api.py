import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app import main
from app.llm import DirectSourceDecision, UserFactExtractionDecision
from app.session_store import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_sessions_are_isolated_and_deleted_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(
                Path(temp_dir),
                grace_ttl_seconds=5,
                hard_ttl_seconds=30,
            )
            first = uuid4()
            second = uuid4()
            store.begin_operation(first)
            store.write_memory(first, [{"role": "user", "content": "one"}], {})
            store.end_operation(first)
            store.begin_operation(second)
            store.write_memory(second, [{"role": "user", "content": "two"}], {})
            store.end_operation(second)

            self.assertEqual(store.read_memory(first)["messages"][0]["content"], "one")
            self.assertEqual(store.read_memory(second)["messages"][0]["content"], "two")
            self.assertTrue(store.delete(first))
            self.assertEqual(store.read_memory(first)["messages"], [])
            self.assertEqual(store.read_memory(second)["messages"][0]["content"], "two")

    def test_active_session_deletion_waits_for_operation_to_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir), 5, 30)
            session_id = uuid4()
            store.begin_operation(session_id)
            self.assertTrue(store.delete(session_id))
            self.assertTrue((Path(temp_dir) / str(session_id)).exists())
            self.assertTrue(store.end_operation(session_id))
            self.assertFalse((Path(temp_dir) / str(session_id)).exists())


class DirectApiTests(unittest.TestCase):
    def test_direct_branch_exposes_only_production_routes(self) -> None:
        paths = {route.path for route in main.app.routes}
        self.assertIn("/chat/stream", paths)
        self.assertIn("/health", paths)
        self.assertIn("/warmup", paths)
        self.assertIn("/sessions/{session_id}", paths)
        self.assertNotIn("/documents/upload", paths)
        self.assertNotIn("/search", paths)

    def test_chat_stream_persists_backend_owned_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir), 5, 30)
            session_id = uuid4()
            no_facts = UserFactExtractionDecision(
                facts=(),
                attempted=False,
                attempts=0,
                duration_seconds=0.0,
                raw_output_preview="",
            )
            with (
                patch.object(main, "get_session_store", return_value=store),
                patch.object(main, "extract_user_facts", AsyncMock(return_value=no_facts)),
                patch.object(
                    main,
                    "generate_direct_answer",
                    AsyncMock(return_value=("Xin chào!", {})),
                ),
                TestClient(main.app) as client,
            ):
                headers = {}
                if main.settings.api_auth_required:
                    headers["Authorization"] = f"Bearer {main.settings.api_auth_token}"
                response = client.post(
                    "/chat/stream",
                    json={"session_id": str(session_id), "message": "xin chào"},
                    headers=headers,
                )

            self.assertEqual(response.status_code, 200)
            events = [json.loads(line) for line in response.text.splitlines()]
            self.assertEqual(events[-1]["type"], "done")
            self.assertEqual(
                "".join(event.get("token", "") for event in events),
                "Xin chào!",
            )
            memory = store.read_memory(session_id)
            self.assertEqual([item["role"] for item in memory["messages"]], ["user", "assistant"])


class DirectGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_simple_direct_answer_uses_chat_without_retry(self) -> None:
        request = main.ChatRequest(session_id=uuid4(), message="xin chào")
        with patch.object(
            main,
            "query_ollama_chat",
            AsyncMock(return_value="Xin chào! Mình có thể giúp gì cho bạn?"),
        ) as chat:
            answer, _ = await main.generate_direct_answer(request)

        self.assertEqual(answer, "Xin chào! Mình có thể giúp gì cho bạn?")
        self.assertEqual(chat.await_count, 1)

    async def test_writing_contract_retries_once_and_selects_valid_length(self) -> None:
        request = main.ChatRequest(
            session_id=uuid4(),
            message=(
                "Hãy viết một đoạn văn khoảng 150 từ bằng tiếng Anh về lợi ích "
                "của giáo dục trực tuyến."
            ),
        )
        source = DirectSourceDecision(
            source="available",
            attempts=1,
            duration_seconds=0.01,
            raw_output_preview='{"source":"available"}',
        )
        short = "Online learning helps students." * 10
        valid = " ".join(["Learning"] * 150) + "."
        with (
            patch.object(main, "classify_direct_source", AsyncMock(return_value=source)),
            patch.object(
                main,
                "query_ollama_chat",
                AsyncMock(side_effect=[short, valid]),
            ) as chat,
        ):
            answer, _ = await main.generate_direct_answer(request)

        self.assertEqual(answer, valid)
        self.assertEqual(chat.await_count, 2)


if __name__ == "__main__":
    unittest.main()
