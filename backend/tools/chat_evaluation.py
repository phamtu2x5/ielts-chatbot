from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
DEFAULT_MANIFEST = BACKEND_DIR / "evaluation" / "chat_corpus_v2.json"
DEFAULT_CORPUS_DIR = REPO_DIR / "docs"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "chat_evaluation"

SOURCE_METADATA_FIELDS = (
    "unit_type",
    "chunk_reason",
    "passage_number",
    "question_range",
    "parent_id",
    "document_type",
    "task_type",
)


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_corpus(manifest: dict[str, Any], corpus_dir: Path) -> list[Path]:
    files: list[Path] = []
    errors: list[str] = []
    for document in manifest.get("documents", []):
        path = corpus_dir / document["filename"]
        if not path.is_file():
            errors.append(f"Missing corpus file: {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != document.get("sha256"):
            errors.append(f"SHA256 mismatch for {path.name}: {digest}")
        files.append(path)
    if errors:
        raise ValueError("\n".join(errors))
    return files


def request_json(
    method: str,
    url: str,
    payload: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 600.0,
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, data=payload, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail: dict[str, Any] = json.loads(body)
        except json.JSONDecodeError:
            detail = {"detail": body}
        return exc.code, detail


def request_ndjson(
    url: str,
    payload: bytes,
    timeout: float,
) -> tuple[int, list[dict[str, Any]]]:
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            events = [
                json.loads(line.decode("utf-8"))
                for line in response
                if line.strip()
            ]
            return response.status, events
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail: dict[str, Any] = json.loads(body)
        except json.JSONDecodeError:
            detail = {"detail": body}
        return exc.code, [{"type": "error", "detail": detail}]


def multipart_file(path: Path) -> tuple[bytes, str]:
    boundary = f"----ielts-chat-evaluation-{uuid4().hex}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    body = prefix + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, boundary


def upload_document(base_url: str, path: Path, timeout: float) -> dict[str, Any]:
    body, boundary = multipart_file(path)
    started = time.perf_counter()
    status, response = request_json(
        "POST",
        f"{base_url}/documents/upload",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout,
    )
    return {
        "filename": path.name,
        "http_status": status,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "response": response,
    }


def ask_chat(
    base_url: str,
    message: str,
    timeout: float,
    conversation_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "message": message,
            "document_ids": None,
            "document_scope": "available",
            "conversation_state": conversation_state,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    started = time.perf_counter()
    status, events = request_ndjson(
        f"{base_url}/chat/stream",
        payload,
        timeout,
    )
    response: dict[str, Any] = {
        "response": "",
        "route_used": None,
        "sources": [],
        "debug": {},
        "conversation_state": None,
    }
    stream_error: Any = None
    for event in events:
        event_type = event.get("type")
        if event_type == "metadata":
            response.update(
                {
                    "route_used": event.get("route_used"),
                    "sources": event.get("sources") or [],
                    "debug": event.get("debug") or {},
                    "conversation_state": event.get("conversation_state"),
                }
            )
        elif event_type == "token":
            response["response"] += str(event.get("token") or "")
        elif event_type == "error":
            stream_error = event.get("detail") or event.get("message") or event
    result = {
        "http_status": status,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "response": response,
    }
    if stream_error is not None:
        result["error"] = stream_error
    return result


def source_reference_id(source: dict[str, Any]) -> str:
    chunk_id = source.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    identity = {
        "source_file": source.get("source_file") or source.get("file"),
        "pages": source.get("pages"),
        "text": source.get("display_text") or source.get("text") or source.get("preview"),
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"source-{digest}"


def compact_source_reference(source: dict[str, Any]) -> dict[str, Any]:
    metadata = source.get("metadata") or {}
    compact = {
        "source_ref": source_reference_id(source),
        "source_file": source.get("source_file") or source.get("file"),
        "pages": source.get("pages"),
        "retrieval_method": source.get("retrieval_method"),
        "score": source.get("score"),
        "dense_score": source.get("probe_dense_score", source.get("dense_score")),
        "keyword_score": source.get("probe_keyword_score", source.get("keyword_score")),
        "question_score": source.get("probe_question_score", source.get("question_score")),
        "overview_score": source.get("probe_overview_score", source.get("overview_score")),
        **{field: metadata.get(field) for field in SOURCE_METADATA_FIELDS},
    }
    return {key: value for key, value in compact.items() if value is not None}


def source_index_entry(source: dict[str, Any]) -> dict[str, Any]:
    metadata = source.get("metadata") or {}
    entry = {
        "source_ref": source_reference_id(source),
        "chunk_id": source.get("chunk_id"),
        "source_file": source.get("source_file") or source.get("file"),
        "pages": source.get("pages"),
        "text": source.get("display_text") or source.get("text") or source.get("preview"),
        **{field: metadata.get(field) for field in SOURCE_METADATA_FIELDS},
    }
    return {key: value for key, value in entry.items() if value is not None}


def compact_case_debug(debug: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in debug.items() if key != "catalog"}


def compact_upload_result(result: dict[str, Any]) -> dict[str, Any]:
    response = result.get("response") or {}
    debug = response.get("debug") or {}
    compact_response = {
        key: response.get(key)
        for key in (
            "message",
            "detail",
            "file_name",
            "document_id",
            "document_type",
            "chunks_processed",
            "collection_stats",
        )
        if response.get(key) is not None
    }
    compact_debug = {
        key: debug.get(key)
        for key in ("timing", "structure")
        if debug.get(key) is not None
    }
    if compact_debug:
        compact_response["debug"] = compact_debug
    compact = {
        "filename": result.get("filename"),
        "http_status": result.get("http_status"),
        "duration_seconds": result.get("duration_seconds"),
        "error": result.get("error"),
        "response": compact_response,
    }
    return {key: value for key, value in compact.items() if value is not None}


def capture_case(
    case: dict[str, Any],
    result: dict[str, Any],
    source_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    response = result.get("response") or {}
    raw_sources = response.get("sources") or []
    if source_index is not None:
        for source in raw_sources:
            entry = source_index_entry(source)
            source_index.setdefault(entry["source_ref"], entry)
    return {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "expected_target_files": case.get("expected_target_files", []),
        "request_context": {
            "document_ids": None,
            "document_scope": "available",
            "conversation_state": None,
        },
        "http_status": result.get("http_status"),
        "duration_seconds": result.get("duration_seconds"),
        "request_error": result.get("error"),
        "error_detail": response if result.get("http_status") != 200 else None,
        "answer": response.get("response"),
        "route_used": response.get("route_used"),
        "conversation_state": response.get("conversation_state"),
        "resolved_document_ids": (
            (response.get("debug") or {})
            .get("document_resolution", {})
            .get("resolved_document_ids", [])
        ),
        "sources": [compact_source_reference(source) for source in raw_sources],
        "debug": compact_case_debug(response.get("debug") or {}),
    }


def select_cases(manifest: dict[str, Any], case_ids: list[str]) -> list[dict[str, Any]]:
    cases = manifest.get("cases", [])
    if not case_ids:
        return cases
    selected = [case for case in cases if case.get("id") in set(case_ids)]
    missing = sorted(set(case_ids) - {case["id"] for case in selected})
    if missing:
        raise ValueError(f"Unknown case IDs: {', '.join(missing)}")
    return selected


def merge_document_catalog(
    catalog_by_file: dict[str, dict[str, Any]],
    entries: list[dict[str, Any]],
) -> None:
    for entry in entries:
        source_file = str(entry.get("source_file") or "")
        if source_file:
            catalog_by_file[source_file] = entry


def run_capture(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    manifest = load_manifest(args.manifest)
    corpus_files = verify_corpus(manifest, args.corpus_dir)
    base_url = args.base_url.rstrip("/")
    upload_results: list[dict[str, Any]] = []

    if not args.skip_upload:
        for path in corpus_files:
            try:
                upload_results.append(upload_document(base_url, path, args.upload_timeout))
            except Exception as exc:
                upload_results.append({"filename": path.name, "http_status": None, "error": repr(exc)})

    case_results: list[dict[str, Any]] = []
    source_index: dict[str, dict[str, Any]] = {}
    document_catalog_by_file: dict[str, dict[str, Any]] = {}
    for case in select_cases(manifest, args.case):
        try:
            raw_result = ask_chat(
                base_url,
                case["query"],
                args.chat_timeout,
            )
        except Exception as exc:
            raw_result = {
                "http_status": None,
                "duration_seconds": None,
                "response": {},
                "error": repr(exc),
            }
        response_debug = (raw_result.get("response") or {}).get("debug") or {}
        merge_document_catalog(
            document_catalog_by_file,
            response_debug.get("catalog") or [],
        )
        case_results.append(
            capture_case(
                case,
                raw_result,
                source_index,
            )
        )

    upload_errors = sum(item.get("http_status") != 200 for item in upload_results)
    request_errors = sum(
        item.get("http_status") != 200 or bool(item.get("request_error"))
        for item in case_results
    )
    report = {
        "schema_version": "1.3",
        "question_set_name": manifest.get("name"),
        "manifest_schema_version": manifest.get("schema_version"),
        "base_url": base_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "uploads": [compact_upload_result(result) for result in upload_results],
        "document_catalog": list(document_catalog_by_file.values()),
        "source_index": list(source_index.values()),
        "summary": {
            "total_questions": len(case_results),
            "responses_collected": len(case_results) - request_errors,
            "request_errors": request_errors,
            "uploads_attempted": len(upload_results),
            "upload_errors": upload_errors,
            "unique_context_sources": len(source_index),
            "answer_assessment": "not_performed",
        },
        "cases": case_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output_dir / f"chat-review-capture-{timestamp}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect chatbot answers and RAG debug data for manual review."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:2222")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--case", action="append", default=[], help="Run one case ID; repeat as needed.")
    parser.add_argument("--upload-timeout", type=float, default=900.0)
    parser.add_argument("--chat-timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output_path, report = run_capture(args)
    except Exception as exc:
        print(f"Capture setup failed: {exc}")
        return 2
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"Report: {output_path}")
    return 1 if report["summary"]["request_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
