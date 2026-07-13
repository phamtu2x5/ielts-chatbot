import json
import subprocess
import sys
import time
from pathlib import Path


API_URL = "https://involvement-buzz-joseph-orlando.trycloudflare.com/api/chat"
OUT_DIR = Path(__file__).resolve().parent

QUESTIONS = [
    "Hiện tôi đã tải lên những tài liệu nào? Hãy mô tả ngắn từng tài liệu, không giải bài.",
    "Tài liệu IELTS Reading gồm những passage nào và mỗi passage có những nhóm câu hỏi nào?",
    "Tóm tắt nội dung của cả ba passage trong đề Reading. Không giải câu hỏi.",
    "Đề Writing trong ảnh yêu cầu người học làm gì? Chỉ giải thích yêu cầu, chưa viết bài.",
    "Dịch Questions 1–4 sang tiếng Việt nhưng không đưa đáp án.",
    "Giải thích yêu cầu và cách làm Questions 1–4, nhưng chưa giải từng câu.",
    "Trả lời Questions 1–4 và dẫn bằng chứng ngắn từ passage cho từng câu.",
    "Hiển thị lại toàn bộ bảng của Questions 5–10 theo dạng Markdown, giữ đúng hàng, cột và vị trí các ô trống. Không giải bài.",
    "Hiển thị cấu trúc flowchart của Questions 18–23. Hãy mô tả các node và hướng nối giữa chúng, chưa điền đáp án.",
    "Passage 1 giải thích màu đỏ hoặc trắng của rượu được tạo ra như thế nào?",
    "Tại sao tác giả của Passage 2 phản đối việc “envisioning” trong tổ chức? Hãy nêu ba lý do chính.",
    "Hãy trích xuất toàn bộ bảng số liệu trong ảnh Writing thành bảng Markdown. Chưa phân tích và chưa viết bài.",
    "Tỷ lệ sở hữu smartphone của nước B năm 2024 là bao nhiêu? Chỉ trả lời giá trị và nguồn.",
    "Quốc gia nào có mức tăng Internet Access lớn nhất từ 2019 đến 2024? Trình bày phép tính.",
    "Viết riêng một đoạn overview cho đề Writing, không viết introduction hoặc body paragraph.",
    "Viết một bài IELTS Writing Task 1 khoảng 170–190 từ dựa hoàn toàn trên bảng trong ảnh.",
    "Đề Reading có nhắc đến mạng xã hội và việc học tiếng Anh của sinh viên không?",
    "Trong Reading Passage 1, tác giả nói rằng smartphone ownership tăng mạnh nhất ở Country C đúng không?",
    "Task 2 ở cuối file Reading yêu cầu gì? Nó có giống đề Writing trong ảnh không?",
]


def post_chat(question: str) -> tuple[dict, float, int | None]:
    body = json.dumps(
        {"message": question, "conversation_history": []},
        ensure_ascii=False,
    )
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [
                "curl",
                "-s",
                "-w",
                "\n%{http_code}",
                "-X",
                "POST",
                API_URL,
                "-H",
                "Content-Type: application/json",
                "-d",
                body,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        duration = time.perf_counter() - started
        if result.returncode != 0:
            return {"error": result.stderr or result.stdout}, duration, None
        raw, _, status_text = result.stdout.rpartition("\n")
        status = int(status_text) if status_text.isdigit() else None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": raw}
        return payload, duration, status
    except Exception as exc:
        return {"error": repr(exc)}, time.perf_counter() - started, None


def source_preview(source: dict) -> dict:
    metadata = source.get("metadata") or {}
    return {
        "file": source.get("source_file"),
        "pages": source.get("pages"),
        "score": source.get("score"),
        "dense": source.get("probe_dense_score"),
        "keyword": source.get("probe_keyword_score"),
        "question": source.get("probe_question_score"),
        "overview": source.get("probe_overview_score"),
        "chunk_id": source.get("chunk_id"),
        "unit_type": metadata.get("unit_type"),
        "chunk_reason": metadata.get("chunk_reason"),
        "passage_number": metadata.get("passage_number"),
        "question_range": metadata.get("question_range"),
        "parent_id": metadata.get("parent_id"),
        "preview": (source.get("display_text") or source.get("text") or "")[:700],
    }


def make_flags(payload: dict) -> list[str]:
    flags = []
    response = payload.get("response") or ""
    route_used = payload.get("route_used")
    debug = payload.get("debug") or {}
    intent = debug.get("query_intent")
    sources = payload.get("sources") or []
    catalog = debug.get("catalog") or []
    if route_used == "base_model" and any(
        marker in (debug.get("probe") or {})
        for marker in ["has_hits", "has_document_intent"]
    ):
        flags.append("Route base_model; verify if this should have used RAG.")
    if route_used == "vector_rag" and not sources:
        flags.append("RAG route has no sources.")
    if not response.strip():
        flags.append("Empty response.")
    if intent in {"show_questions", "translate_questions", "explain_questions"}:
        lowered = response.lower()
        if any(answer in lowered for answer in ["→ true", "→ false", "→ not given", "đáp án"]):
            flags.append("Possible policy leak: answer labels appear in non-solve intent.")
    for item in catalog:
        passages = item.get("passage_numbers") or []
        if len(passages) > 4:
            flags.append(
                f"Suspicious passage_numbers for {item.get('source_file')}: {passages}"
            )
    return flags


def write_markdown(test_id: int, question: str, record: dict) -> None:
    payload = record.get("payload") or {}
    debug = payload.get("debug") or {}
    probe = debug.get("probe") or {}
    sources = payload.get("sources") or []
    flags = record.get("flags") or []
    lines = [
        f"# Test {test_id:02d}",
        "",
        "## Question",
        "",
        question,
        "",
        "## Answer",
        "",
        payload.get("response") or payload.get("error") or "",
        "",
        "## Route And Intent",
        "",
        f"- HTTP status: `{record.get('http_status')}`",
        f"- Duration seconds: `{record.get('duration_seconds'):.3f}`",
        f"- route_used: `{payload.get('route_used')}`",
        f"- route_decision: `{debug.get('route_decision')}`",
        f"- query_intent: `{debug.get('query_intent')}`",
        f"- source_count: `{debug.get('source_count')}`",
        "",
        "## Probe Summary",
        "",
        f"- has_hits: `{probe.get('has_hits')}`",
        f"- has_strong_hits: `{probe.get('has_strong_hits')}`",
        f"- has_document_intent: `{probe.get('has_document_intent')}`",
        f"- is_overview: `{probe.get('is_overview')}`",
        f"- top_score: `{probe.get('top_score')}`",
        f"- top_keyword_score: `{probe.get('top_keyword_score')}`",
        f"- top_question_score: `{probe.get('top_question_score')}`",
        f"- top_overview_score: `{probe.get('top_overview_score')}`",
        "",
        "## Flags",
        "",
    ]
    if flags:
        lines.extend(f"- {flag}" for flag in flags)
    else:
        lines.append("- None")
    lines.extend(["", "## Sources", ""])
    if sources:
        for index, source in enumerate(sources, 1):
            preview = source_preview(source)
            lines.extend(
                [
                    f"### Source {index}",
                    "",
                    f"- file: `{preview['file']}`",
                    f"- pages: `{preview['pages']}`",
                    f"- chunk_id: `{preview['chunk_id']}`",
                    f"- unit_type: `{preview['unit_type']}`",
                    f"- chunk_reason: `{preview['chunk_reason']}`",
                    f"- passage_number: `{preview['passage_number']}`",
                    f"- question_range: `{preview['question_range']}`",
                    f"- dense: `{preview['dense']}`",
                    f"- keyword: `{preview['keyword']}`",
                    f"- question: `{preview['question']}`",
                    f"- overview: `{preview['overview']}`",
                    "",
                    "```text",
                    preview["preview"],
                    "```",
                    "",
                ]
            )
    else:
        lines.append("- No sources returned.")
    (OUT_DIR / f"test_{test_id:02d}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        prepare_requests()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        collect_raw_outputs()
        return

    summary_rows = []
    for index, question in enumerate(QUESTIONS, 1):
        payload, duration, status = post_chat(question)
        record = {
            "test_id": index,
            "question": question,
            "http_status": status,
            "duration_seconds": duration,
            "payload": payload,
            "flags": make_flags(payload),
            "source_previews": [source_preview(source) for source in payload.get("sources", [])],
        }
        (OUT_DIR / f"test_{index:02d}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_markdown(index, question, record)
        debug = payload.get("debug") or {}
        summary_rows.append(
            {
                "test_id": index,
                "status": status,
                "duration": round(duration, 3),
                "route": payload.get("route_used"),
                "intent": debug.get("query_intent"),
                "source_count": debug.get("source_count"),
                "flags": record["flags"],
            }
        )

    summary_lines = [
        "# API Batch Test Summary",
        "",
        f"- API URL: `{API_URL}`",
        f"- Total tests: `{len(QUESTIONS)}`",
        "",
        "| Test | HTTP | Seconds | Route | Intent | Sources | Flags |",
        "|---:|---:|---:|---|---|---:|---|",
    ]
    for row in summary_rows:
        flags = "<br>".join(row["flags"]) if row["flags"] else ""
        summary_lines.append(
            f"| {row['test_id']:02d} | {row['status']} | {row['duration']} | "
            f"{row['route']} | {row['intent']} | {row['source_count']} | {flags} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def prepare_requests() -> None:
    for index, question in enumerate(QUESTIONS, 1):
        request_body = {"message": question, "conversation_history": []}
        (OUT_DIR / f"request_{index:02d}.json").write_text(
            json.dumps(request_body, ensure_ascii=False),
            encoding="utf-8",
        )


def collect_raw_outputs() -> None:
    summary_rows = []
    for index, question in enumerate(QUESTIONS, 1):
        body_path = OUT_DIR / f"raw_{index:02d}.body"
        raw_path = OUT_DIR / f"raw_{index:02d}.out"
        duration_path = OUT_DIR / f"duration_{index:02d}.txt"
        status_path = OUT_DIR / f"status_{index:02d}.txt"
        if body_path.exists():
            raw = body_path.read_text(encoding="utf-8", errors="replace")
            status_text = status_path.read_text(encoding="utf-8").strip() if status_path.exists() else "200"
            status = int(status_text) if status_text.isdigit() else None
        else:
            raw_text = raw_path.read_text(encoding="utf-8", errors="replace") if raw_path.exists() else ""
            raw, _, status_text = raw_text.rstrip("\n").rpartition("\n")
            status = int(status_text) if status_text.isdigit() else None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": raw}
        try:
            duration = float(duration_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            duration = 0.0
        record = {
            "test_id": index,
            "question": question,
            "http_status": status,
            "duration_seconds": duration,
            "payload": payload,
            "flags": make_flags(payload),
            "source_previews": [source_preview(source) for source in payload.get("sources", [])],
        }
        (OUT_DIR / f"test_{index:02d}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_markdown(index, question, record)
        debug = payload.get("debug") or {}
        summary_rows.append(
            {
                "test_id": index,
                "status": status,
                "duration": round(duration, 3),
                "route": payload.get("route_used"),
                "intent": debug.get("query_intent"),
                "source_count": debug.get("source_count"),
                "flags": record["flags"],
            }
        )

    summary_lines = [
        "# API Batch Test Summary",
        "",
        f"- API URL: `{API_URL}`",
        f"- Total tests: `{len(QUESTIONS)}`",
        "",
        "| Test | HTTP | Seconds | Route | Intent | Sources | Flags |",
        "|---:|---:|---:|---|---|---:|---|",
    ]
    for row in summary_rows:
        flags = "<br>".join(row["flags"]) if row["flags"] else ""
        summary_lines.append(
            f"| {row['test_id']:02d} | {row['status']} | {row['duration']} | "
            f"{row['route']} | {row['intent']} | {row['source_count']} | {flags} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
