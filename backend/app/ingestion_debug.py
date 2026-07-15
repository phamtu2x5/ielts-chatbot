import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class IngestionDebugStore:
    """Persist temporary ingestion diagnostics independently of the HTTP response."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.events_path = path.with_name("ingestion_events.jsonl")
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
            "temporary_diagnostics": True,
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
            "events_recorded": 0,
        }
        with self._lock:
            start_event = self._append_event_unlocked(payload, "ingestion_started", now, {})
            payload["last_event"] = start_event
            payload["events_recorded"] = 1
            self._write_unlocked(payload)
        return payload

    def event(self, request_id: str, event: str, **details: Any) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            payload = self._read_unlocked()
            event_payload = self._append_event_unlocked(payload, event, now, {
                "request_id": request_id,
                **details,
            })
            if payload and payload.get("request_id") == request_id:
                payload["stage"] = details.get("stage", event)
                payload["last_event"] = event_payload
                payload["events_recorded"] = int(payload.get("events_recorded", 0)) + 1
                payload["updated_at"] = self._iso_time(now)
                payload["updated_at_epoch"] = now
                self._write_unlocked(payload)
            return event_payload

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

    def _append_event_unlocked(
        self,
        summary: dict[str, Any] | None,
        event: str,
        now: float,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        request_id = details.pop("request_id", None) or (summary or {}).get("request_id")
        started_at = (summary or {}).get("started_at_epoch")
        payload = {
            "schema_version": "1.0",
            "temporary_diagnostics": True,
            "request_id": request_id,
            "source_file": (summary or {}).get("source_file"),
            "event": event,
            "timestamp": self._iso_time(now),
            "timestamp_epoch": now,
            "elapsed_seconds": round(now - started_at, 3) if started_at else None,
            **details,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def _iso_time(self, value: float) -> str:
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
