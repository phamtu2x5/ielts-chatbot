from __future__ import annotations

import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from .config import settings


class SessionStore:
    EXPIRY_FILE = ".expires_at"
    MEMORY_FILE = "memory.json"

    def __init__(
        self,
        data_dir: Path | None = None,
        grace_ttl_seconds: int | None = None,
        hard_ttl_seconds: int | None = None,
    ) -> None:
        self.sessions_dir = data_dir or settings.session_data_dir
        self.grace_ttl_seconds = grace_ttl_seconds or settings.session_grace_ttl_seconds
        self.hard_ttl_seconds = hard_ttl_seconds or settings.session_hard_ttl_seconds
        self._lock = threading.RLock()
        self._expires_at: dict[str, float] = {}
        self._known_sessions: set[str] = set()
        self._active_operations: dict[str, int] = {}
        self._delete_pending: set[str] = set()
        self._cleanup_runs = 0
        self._cleaned_sessions = 0
        self._last_cleanup_at: str | None = None

    @staticmethod
    def normalize_session_id(session_id: UUID | str) -> str:
        try:
            return str(UUID(str(session_id)))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("session_id must be a valid UUID.") from exc

    def _session_dir(self, normalized: str) -> Path:
        return self.sessions_dir / normalized

    def _expiry_path(self, normalized: str) -> Path:
        return self._session_dir(normalized) / self.EXPIRY_FILE

    def _memory_path(self, normalized: str) -> Path:
        return self._session_dir(normalized) / self.MEMORY_FILE

    def _read_expiry_locked(self, normalized: str) -> float | None:
        if normalized in self._expires_at:
            return self._expires_at[normalized]
        try:
            expires_at = float(self._expiry_path(normalized).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        self._expires_at[normalized] = expires_at
        return expires_at

    def _is_expired_locked(self, normalized: str, now: float) -> bool:
        session_dir = self._session_dir(normalized)
        expires_at = self._read_expiry_locked(normalized)
        if expires_at is not None and expires_at <= now:
            return True
        try:
            last_access = session_dir.stat().st_mtime
        except OSError:
            return False
        return now - last_access >= self.hard_ttl_seconds

    def _touch_locked(self, normalized: str) -> None:
        session_dir = self._session_dir(normalized)
        session_dir.mkdir(parents=True, exist_ok=True)
        self._known_sessions.add(normalized)
        self._expires_at.pop(normalized, None)
        self._expiry_path(normalized).unlink(missing_ok=True)
        os.utime(session_dir, None)

    def _delete_locked(self, normalized: str) -> bool:
        session_dir = self._session_dir(normalized)
        self._expires_at.pop(normalized, None)
        self._known_sessions.discard(normalized)
        self._delete_pending.discard(normalized)
        existed = session_dir.exists()
        if existed:
            shutil.rmtree(session_dir)
        return existed

    def begin_operation(self, session_id: UUID | str) -> str:
        normalized = self.normalize_session_id(session_id)
        with self._lock:
            if normalized in self._delete_pending:
                raise RuntimeError("Session deletion is already in progress.")
            if self._is_expired_locked(normalized, time.time()):
                self._delete_locked(normalized)
            self._touch_locked(normalized)
            self._active_operations[normalized] = self._active_operations.get(normalized, 0) + 1
            return normalized

    def end_operation(self, session_id: UUID | str) -> bool:
        normalized = self.normalize_session_id(session_id)
        with self._lock:
            active = self._active_operations.get(normalized, 0)
            if active <= 1:
                self._active_operations.pop(normalized, None)
            else:
                self._active_operations[normalized] = active - 1
                return False
            if normalized not in self._delete_pending:
                return False
            return self._delete_locked(normalized)

    def read_memory(self, session_id: UUID | str) -> dict[str, Any]:
        normalized = self.normalize_session_id(session_id)
        with self._lock:
            try:
                payload = json.loads(self._memory_path(normalized).read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {"messages": [], "conversation_state": None}
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("Session memory cannot be loaded.") from exc
            if not isinstance(payload, dict):
                raise RuntimeError("Session memory is invalid.")
            return payload

    def write_memory(
        self,
        session_id: UUID | str,
        messages: list[dict[str, Any]],
        conversation_state: dict[str, Any],
    ) -> None:
        normalized = self.normalize_session_id(session_id)
        with self._lock:
            if normalized in self._delete_pending:
                return
            session_dir = self._session_dir(normalized)
            session_dir.mkdir(parents=True, exist_ok=True)
            path = self._memory_path(normalized)
            temp_path = path.with_suffix(".json.tmp")
            payload = {
                "messages": messages,
                "conversation_state": conversation_state,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                temp_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temp_path.replace(path)
            finally:
                temp_path.unlink(missing_ok=True)
            os.utime(session_dir, None)

    def schedule_expiration(self, session_id: UUID | str) -> bool:
        normalized = self.normalize_session_id(session_id)
        with self._lock:
            session_dir = self._session_dir(normalized)
            if not session_dir.exists():
                return False
            expires_at = time.time() + self.grace_ttl_seconds
            self._expires_at[normalized] = expires_at
            self._known_sessions.add(normalized)
            self._expiry_path(normalized).write_text(str(expires_at), encoding="utf-8")
            return True

    def delete(self, session_id: UUID | str) -> bool:
        normalized = self.normalize_session_id(session_id)
        with self._lock:
            if self._active_operations.get(normalized, 0):
                existed = self._session_dir(normalized).exists()
                self._delete_pending.add(normalized)
                return existed
            return self._delete_locked(normalized)

    def cleanup_expired(self, now: float | None = None) -> int:
        current_time = time.time() if now is None else now
        with self._lock:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            session_ids: set[str] = set()
            for session_dir in self.sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                try:
                    session_ids.add(self.normalize_session_id(session_dir.name))
                except ValueError:
                    continue
            self._known_sessions.update(session_ids)
            deleted = 0
            for normalized in session_ids:
                if not self._is_expired_locked(normalized, current_time):
                    continue
                if self._active_operations.get(normalized, 0):
                    self._delete_pending.add(normalized)
                else:
                    deleted += int(self._delete_locked(normalized))
            self._cleanup_runs += 1
            self._cleaned_sessions += deleted
            self._last_cleanup_at = datetime.now(timezone.utc).isoformat()
            return deleted

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            sessions = 0
            pending_expiry = 0
            storage_bytes = 0
            for session_dir in self.sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                try:
                    self.normalize_session_id(session_dir.name)
                except ValueError:
                    continue
                sessions += 1
                pending_expiry += int((session_dir / self.EXPIRY_FILE).exists())
                for path in session_dir.iterdir():
                    if path.is_file():
                        try:
                            storage_bytes += path.stat().st_size
                        except OSError:
                            pass
            return {
                "sessions": sessions,
                "active_sessions": max(0, sessions - pending_expiry),
                "pending_expiry": pending_expiry,
                "in_flight_sessions": len(self._active_operations),
                "active_operations": sum(self._active_operations.values()),
                "pending_deletions": len(self._delete_pending),
                "storage_bytes": storage_bytes,
                "cleanup_runs": self._cleanup_runs,
                "cleaned_sessions": self._cleaned_sessions,
                "last_cleanup_at": self._last_cleanup_at,
            }

    def runtime_stats(self) -> dict[str, Any]:
        with self._lock:
            pending_expiry = sum(
                1 for session_id in self._known_sessions if session_id in self._expires_at
            )
            return {
                "sessions": len(self._known_sessions),
                "active_sessions": max(0, len(self._known_sessions) - pending_expiry),
                "in_flight_sessions": len(self._active_operations),
                "active_operations": sum(self._active_operations.values()),
                "pending_deletions": len(self._delete_pending),
                "cleanup_runs": self._cleanup_runs,
                "cleaned_sessions": self._cleaned_sessions,
                "last_cleanup_at": self._last_cleanup_at,
            }


_store: SessionStore | None = None
_store_lock = threading.Lock()


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = SessionStore()
    return _store
