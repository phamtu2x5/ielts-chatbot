import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class IngestionDebugStore:
    """Persist the latest ingestion trace independently of the HTTP response."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    def start(
        self,
        request_id: str,
        source_file: str,
        pipeline: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        payload = {
            "schema_version": "1.0",
            "request_id": request_id,
            "source_file": source_file,
            "status": "processing",
            "stage": "save_file",
            "started_at": self._iso_time(now),
            "started_at_epoch": now,
            "updated_at": self._iso_time(now),
            "updated_at_epoch": now,
            "pipeline": pipeline,
            "timing": {},
        }
        with self._lock:
            self._write_unlocked(payload)
        return payload

    def update(self, request_id: str, **changes: Any) -> dict[str, Any] | None:
        with self._lock:
            payload = self._read_unlocked()
            if not payload or payload.get("request_id") != request_id:
                return None
            payload.update(changes)
            now = time.time()
            payload["updated_at"] = self._iso_time(now)
            payload["updated_at_epoch"] = now
            if payload.get("status") in {"completed", "failed"}:
                payload["finished_at"] = self._iso_time(now)
                payload["finished_at_epoch"] = now
                payload["elapsed_seconds"] = round(now - payload["started_at_epoch"], 3)
            self._write_unlocked(payload)
            return payload

    def read(self) -> dict[str, Any] | None:
        with self._lock:
            return self._read_unlocked()

    def _read_unlocked(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(self.path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _iso_time(self, value: float) -> str:
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
