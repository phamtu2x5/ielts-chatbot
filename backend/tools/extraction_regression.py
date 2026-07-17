from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.document_pipeline import DocumentProcessor


STATUS_ORDER = {"passed": 0, "unsupported": 1, "degraded": 2, "failed": 3}
PACKAGE_NAMES = (
    "PyMuPDF",
    "Pillow",
    "rapidocr",
    "torch",
    "doclayout-yolo",
    "huggingface-hub",
    "numpy",
)


@dataclass
class CheckResult:
    name: str
    status: str
    expected: Any = None
    actual: Any = None
    details: str | None = None
    pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "status": self.status,
            "expected": self.expected,
            "actual": self.actual,
            "pages": self.pages,
        }
        if self.details:
            result["details"] = self.details
        return result


class PipelineEventRecorder:
    """EXTRACTION_REGRESSION_TEMP: captures stage timing and memory without changing the pipeline."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.events: list[dict[str, Any]] = []

    def __call__(self, event: str, details: dict[str, Any]) -> None:
        self.events.append(
            {
                "event": event,
                "elapsed_seconds": round(time.perf_counter() - self.started, 3),
                "rss_mb": current_rss_mb(),
                "gpu": gpu_memory_snapshot(),
                "details": details,
            }
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run document extraction regression without the web stack.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BACKEND_DIR / "tests" / "fixtures" / "extraction_manifest.json",
    )
    parser.add_argument("--corpus-dir", type=Path, default=REPO_ROOT / "docs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    environment = environment_report()
    write_json(output_dir / "environment.json", environment)

    preflight = verify_fixtures(manifest, args.corpus_dir)
    write_json(output_dir / "preflight.json", preflight)
    if not preflight["ok"]:
        summary = base_summary(environment, manifest)
        summary.update({"status": "failed", "phase": "preflight", "preflight": preflight})
        write_json(output_dir / "regression_summary.json", summary)
        archive = None if args.no_archive else create_archive(output_dir, environment["git_commit"])
        print_locations(output_dir, archive)
        return 2

    if args.preflight_only:
        summary = base_summary(environment, manifest)
        summary.update({"status": "passed", "phase": "preflight", "preflight": preflight})
        write_json(output_dir / "regression_summary.json", summary)
        archive = None if args.no_archive else create_archive(output_dir, environment["git_commit"])
        print_locations(output_dir, archive)
        return 0

    processor = DocumentProcessor()
    warmup = run_warmup(processor, skip=args.skip_warmup)
    write_json(output_dir / "model_health.json", warmup)
    if warmup["status"] == "failed":
        summary = base_summary(environment, manifest)
        summary.update({"status": "failed", "phase": "warmup", "model_health": warmup})
        write_json(output_dir / "regression_summary.json", summary)
        archive = None if args.no_archive else create_archive(output_dir, environment["git_commit"])
        print_locations(output_dir, archive)
        return 1

    document_reports = []
    for fixture in manifest["documents"]:
        report = run_document(processor, args.corpus_dir / fixture["filename"], fixture, output_dir)
        document_reports.append(report)

    summary = build_summary(environment, manifest, preflight, warmup, document_reports)
    write_json(output_dir / "regression_summary.json", summary)
    write_json(output_dir / "timing.json", aggregate_timing(document_reports))
    archive = None if args.no_archive else create_archive(output_dir, environment["git_commit"])
    print_locations(output_dir, archive)
    return 0 if summary["status"] == "passed" else 1


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest.get("documents"), list) or not manifest["documents"]:
        raise ValueError("Extraction manifest must contain a non-empty documents list.")
    for fixture in manifest["documents"]:
        if not fixture.get("filename") or not fixture.get("sha256") or not fixture.get("kind"):
            raise ValueError("Every extraction fixture requires filename, sha256, and kind.")
    return manifest


def prepare_output_dir(requested: Path | None, overwrite: bool) -> Path:
    if requested is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        requested = BACKEND_DIR / "data" / "extraction_regression" / f"run-{timestamp}"
    requested = requested.resolve()
    if requested.exists() and any(requested.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {requested}")
        shutil.rmtree(requested)
    requested.mkdir(parents=True, exist_ok=True)
    return requested


def verify_fixtures(manifest: dict[str, Any], corpus_dir: Path) -> dict[str, Any]:
    results = []
    for fixture in manifest["documents"]:
        path = corpus_dir / fixture["filename"]
        result = {"filename": fixture["filename"], "expected_sha256": fixture["sha256"]}
        if not path.exists():
            result.update({"status": "missing", "actual_sha256": None})
        elif is_lfs_pointer(path):
            result.update({"status": "lfs_pointer_not_resolved", "actual_sha256": None})
        else:
            actual = sha256_file(path)
            result.update(
                {
                    "status": "passed" if actual == fixture["sha256"] else "hash_mismatch",
                    "actual_sha256": actual,
                    "bytes": path.stat().st_size,
                }
            )
        results.append(result)
    return {"ok": all(item["status"] == "passed" for item in results), "fixtures": results}


def run_warmup(processor: DocumentProcessor, skip: bool) -> dict[str, Any]:
    if skip:
        return {"status": "skipped"}
    result: dict[str, Any] = {"status": "passed"}
    for name, callback in (("layout", processor.warmup_layout), ("ocr", processor.warmup_ocr)):
        started = time.perf_counter()
        try:
            value = callback()
            ok = warmup_ok(name, value)
            result[name] = {
                "ok": ok,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "result": value,
                "gpu": gpu_memory_snapshot(),
            }
            if not ok:
                result["status"] = "failed"
        except Exception as exc:
            result[name] = {
                "ok": False,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            result["status"] = "failed"
    return result


def warmup_ok(name: str, result: dict[str, Any]) -> bool:
    if result.get("skipped"):
        return False
    if name == "ocr":
        return bool(result.get("models_ready"))
    return bool(result.get("ok"))


def run_document(
    processor: DocumentProcessor,
    source_path: Path,
    fixture: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    slug = safe_slug(source_path.stem)
    document_dir = output_root / "documents" / slug
    document_dir.mkdir(parents=True, exist_ok=True)
    recorder = PipelineEventRecorder()
    reset_gpu_peak_memory()
    started = time.perf_counter()
    try:
        document, chunks = processor.process_file(
            source_path,
            fixture["filename"],
            content_type=fixture.get("mime_type"),
            progress=recorder,
        )
        duration = round(time.perf_counter() - started, 3)
        canonical = document.to_dict()
        chunk_data = [chunk.to_dict() for chunk in chunks]
        checks = evaluate_document(canonical, chunk_data, fixture)
        status = status_from_checks(checks)
        report = {
            "filename": fixture["filename"],
            "document_id": document.document_id,
            "kind": fixture["kind"],
            "status": status,
            "duration_seconds": duration,
            "rss_mb": current_rss_mb(),
            "gpu": gpu_memory_snapshot(include_peak=True),
            "checks": [check.to_dict() for check in checks],
            "fingerprints": {
                "canonical_document": stable_fingerprint(canonical),
                "structure": stable_fingerprint(canonical.get("metadata", {}).get("ielts_structure") or {}),
                "chunks": stable_fingerprint(chunk_data),
            },
        }
        write_json(document_dir / "canonical_document.json", canonical)
        write_json(document_dir / "chunks.json", chunk_data)
        write_json(document_dir / "structure_report.json", structure_report(canonical))
        write_json(document_dir / "extraction_report.json", extraction_report(canonical, recorder.events))
        write_json(document_dir / "events.json", recorder.events)
        write_json(document_dir / "report.json", report)
        if status != "passed":
            render_failure_artifacts(source_path, canonical, document_dir / "failures", checks, processor.config.ocr_dpi)
        return report
    except Exception as exc:
        duration = round(time.perf_counter() - started, 3)
        report = {
            "filename": fixture["filename"],
            "kind": fixture["kind"],
            "status": "failed",
            "duration_seconds": duration,
            "rss_mb": current_rss_mb(),
            "gpu": gpu_memory_snapshot(include_peak=True),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "checks": [],
        }
        write_json(document_dir / "events.json", recorder.events)
        write_json(document_dir / "report.json", report)
        (document_dir / "regression.log").write_text(report["traceback"], encoding="utf-8")
        return report


def evaluate_document(
    canonical: dict[str, Any],
    chunks: list[dict[str, Any]],
    fixture: dict[str, Any],
) -> list[CheckResult]:
    expected = fixture.get("expected") or {}
    checks = [
        equality_check(
            "page_count",
            expected.get("page_count"),
            canonical.get("metadata", {}).get("page_count", len(canonical.get("pages") or [])),
        ),
        CheckResult(
            "chunks_emitted",
            "passed" if chunks else "failed",
            expected="> 0",
            actual=len(chunks),
        ),
    ]
    if fixture["kind"] == "ielts_reading":
        checks.extend(evaluate_reading(canonical, expected.get("reading") or {}))
    elif fixture["kind"] == "ielts_writing_collection":
        checks.extend(evaluate_writing_collection(canonical, expected.get("writing_collection") or {}))
    elif fixture["kind"] == "ielts_writing_task_1_image":
        checks.extend(evaluate_writing_image(canonical, expected.get("writing_task_1") or {}))
    return [check for check in checks if check.expected is not None]


def evaluate_reading(canonical: dict[str, Any], expected: dict[str, Any]) -> list[CheckResult]:
    structure = canonical.get("metadata", {}).get("ielts_structure") or {}
    passages = structure.get("passages") or []
    expected_passages = expected.get("passages") or []
    checks = [equality_check("passage_count", len(expected_passages), len(passages), pages=all_pages(canonical))]

    actual_titles = [passage.get("title") for passage in passages]
    expected_titles = [passage.get("title") for passage in expected_passages]
    checks.append(
        equality_check(
            "passage_titles",
            normalized_titles(expected_titles),
            normalized_titles(actual_titles),
            pages=all_pages(canonical),
        )
    )

    actual_group_map = [
        [question_range(group) for group in passage.get("question_groups") or []]
        for passage in passages
    ]
    expected_group_map = [passage.get("question_groups") or [] for passage in expected_passages]
    checks.append(
        equality_check(
            "question_groups_by_passage",
            expected_group_map,
            actual_group_map,
            pages=all_pages(canonical),
        )
    )

    expected_numbers = expand_number_spec(expected.get("covered_question_numbers"))
    actual_numbers = sorted(
        {
            number
            for passage_groups in actual_group_map
            for start, end in passage_groups
            for number in range(start, end + 1)
        }
    )
    checks.append(
        equality_check(
            "covered_question_numbers",
            expected_numbers,
            actual_numbers,
            pages=all_pages(canonical),
        )
    )

    forbidden = [normalize_label(title) for title in expected.get("forbidden_titles") or []]
    found_forbidden = [title for title in actual_titles if normalize_label(title) in forbidden]
    checks.append(
        CheckResult(
            "forbidden_titles_absent",
            "passed" if not found_forbidden else "failed",
            expected=[],
            actual=found_forbidden,
            pages=all_pages(canonical),
        )
    )

    visual_elements = reading_visual_elements(passages)
    for index, visual_expected in enumerate(expected.get("visuals") or [], 1):
        checks.append(evaluate_visual(visual_elements, visual_expected, f"visual_{index}"))

    diagnostics = structure.get("diagnostics") or {}
    warning_fields = (
        "missing_questions",
        "duplicate_questions",
        "unassigned_questions",
        "overlapping_question_groups",
        "instruction_as_title",
        "suspicious_boundaries",
        "low_confidence_visual_elements",
    )
    warnings = {name: diagnostics.get(name) for name in warning_fields if diagnostics.get(name)}
    checks.append(
        CheckResult(
            "structure_diagnostics",
            "passed" if not warnings else "degraded",
            expected={},
            actual=warnings,
            pages=all_pages(canonical),
        )
    )
    return checks


def evaluate_visual(
    actual_visuals: list[dict[str, Any]],
    expected: dict[str, Any],
    name: str,
) -> CheckResult:
    expected_range = expected.get("question_range")
    expected_type = expected.get("type")
    if expected.get("expected_status") == "unsupported":
        return CheckResult(
            name,
            "unsupported",
            expected={"type": expected_type, "question_range": expected_range},
            actual=None,
            details=expected.get("reason") or "Visual structure is intentionally not supported yet.",
        )
    match = next(
        (
            visual
            for visual in actual_visuals
            if visual.get("type") == expected_type and visual.get("question_range") == expected_range
        ),
        None,
    )
    if match is None:
        return CheckResult(name, "failed", expected=expected, actual=None)

    problems = []
    for field_name, actual_key in (
        ("min_rows", "rows"),
        ("min_columns", "columns"),
        ("min_nodes", "nodes"),
        ("min_edges", "edges"),
        ("min_labels", "labels"),
        ("min_ordered_items", "ordered_items"),
    ):
        minimum = expected.get(field_name)
        if minimum is not None and len(match.get(actual_key) or []) < minimum:
            problems.append(f"{actual_key}<{minimum}")
    problems.extend(str(issue) for issue in match.get("quality_issues") or [])
    return CheckResult(
        name,
        "degraded" if problems else "passed",
        expected=expected,
        actual={
            "type": match.get("type"),
            "question_range": match.get("question_range"),
            "rows": len(match.get("rows") or []),
            "columns": len(match.get("columns") or []),
            "nodes": len(match.get("nodes") or []),
            "edges": len(match.get("edges") or []),
            "labels": len(match.get("labels") or []),
            "connectors": len(match.get("connectors") or []),
            "ordered_items": len(match.get("ordered_items") or []),
            "confidence": match.get("confidence"),
        },
        details=", ".join(problems) if problems else None,
        pages=[int(match["page"])] if match.get("page") else [],
    )


def evaluate_writing_collection(canonical: dict[str, Any], expected: dict[str, Any]) -> list[CheckResult]:
    sections = canonical.get("metadata", {}).get("sections")
    if not isinstance(sections, list):
        return [
            CheckResult(
                "writing_collection_sections",
                "unsupported",
                expected=expected,
                actual=None,
                details="The current canonical schema does not expose mixed Writing task/sample-answer sections.",
                pages=all_pages(canonical),
            )
        ]
    task_sections = [section for section in sections if section.get("type") == "writing_task_1"]
    sample_sections = [section for section in sections if section.get("type") == "sample_answer"]
    checks = [
        equality_check("writing_task_count", expected.get("task_count"), len(task_sections), pages=all_pages(canonical)),
        equality_check(
            "sample_answer_count", expected.get("sample_answer_count"), len(sample_sections), pages=all_pages(canonical)
        ),
    ]
    if expected.get("task_titles") is not None:
        checks.append(
            equality_check(
                "writing_task_titles",
                normalized_titles(expected["task_titles"]),
                normalized_titles([section.get("title") for section in task_sections]),
                pages=all_pages(canonical),
            )
        )
    if expected.get("visual_types") is not None:
        checks.append(
            equality_check(
                "writing_visual_types",
                expected["visual_types"],
                [
                    section.get("visual_type")
                    or (section.get("visual") or {}).get("type")
                    for section in task_sections
                ],
                pages=all_pages(canonical),
            )
        )
    return checks


def evaluate_writing_image(canonical: dict[str, Any], expected: dict[str, Any]) -> list[CheckResult]:
    metadata = canonical.get("metadata", {})
    visual_elements = metadata.get("visual_structure", {}).get("visual_elements") or []
    table = next((element for element in visual_elements if element.get("type") == "table"), None)
    checks = [
        equality_check("document_type", expected.get("document_type"), metadata.get("document_type"), pages=[1]),
        equality_check("task_type", expected.get("task_type"), metadata.get("task_type"), pages=[1]),
    ]
    expected_table = expected.get("table")
    if expected_table is not None:
        if table is None:
            checks.append(CheckResult("writing_table", "failed", expected=expected_table, actual=None, pages=[1]))
        else:
            checks.extend(
                [
                    equality_check("writing_table_columns", expected_table.get("columns"), table.get("columns"), pages=[1]),
                    equality_check("writing_table_rows", expected_table.get("rows"), table.get("rows"), pages=[1]),
                ]
            )
    return checks


def equality_check(
    name: str,
    expected: Any,
    actual: Any,
    pages: list[int] | None = None,
) -> CheckResult:
    return CheckResult(
        name,
        "passed" if expected == actual else "failed",
        expected=expected,
        actual=actual,
        pages=pages or [],
    )


def status_from_checks(checks: Iterable[CheckResult]) -> str:
    statuses = [check.status for check in checks]
    return max(statuses, key=lambda value: STATUS_ORDER[value], default="passed")


def reading_visual_elements(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        group["visual_element"]
        for passage in passages
        for group in passage.get("question_groups") or []
        if isinstance(group.get("visual_element"), dict)
    ]


def question_range(group: dict[str, Any]) -> list[int]:
    return [int(group.get("question_start", 0)), int(group.get("question_end", 0))]


def expand_number_spec(spec: dict[str, Any] | list[int] | None) -> list[int]:
    if isinstance(spec, list):
        return sorted({int(number) for number in spec})
    if isinstance(spec, dict) and spec.get("start") is not None and spec.get("end") is not None:
        return list(range(int(spec["start"]), int(spec["end"]) + 1))
    return []


def normalized_titles(values: Iterable[Any]) -> list[str | None]:
    return [normalize_label(value) if value is not None else None for value in values]


def normalize_label(value: Any) -> str:
    return " ".join(str(value or "").replace("’", "'").split()).strip().casefold()


def all_pages(canonical: dict[str, Any]) -> list[int]:
    return [int(page.get("page_number")) for page in canonical.get("pages") or [] if page.get("page_number")]


def structure_report(canonical: dict[str, Any]) -> dict[str, Any]:
    metadata = canonical.get("metadata", {})
    return {
        "document_type": metadata.get("document_type"),
        "task_type": metadata.get("task_type"),
        "ielts_structure": metadata.get("ielts_structure"),
        "visual_structure": metadata.get("visual_structure"),
        "sections": metadata.get("sections"),
    }


def extraction_report(canonical: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = canonical.get("metadata", {})
    return {
        "filename": canonical.get("filename"),
        "document_id": canonical.get("document_id"),
        "parser_version": canonical.get("parser_version"),
        "timing": metadata.get("timing"),
        "pages": [
            {
                "page_number": page.get("page_number"),
                "processing_route": page.get("processing_route"),
                "quality_score": page.get("quality_score"),
                "metadata": page.get("metadata"),
                "element_count": len(page.get("elements") or []),
                "element_sources": sorted({element.get("source") for element in page.get("elements") or []}),
                "element_types": sorted({element.get("type") for element in page.get("elements") or []}),
            }
            for page in canonical.get("pages") or []
        ],
        "events": events,
    }


def render_failure_artifacts(
    source_path: Path,
    canonical: dict[str, Any],
    failure_dir: Path,
    checks: list[CheckResult],
    dpi: int,
) -> None:
    failure_pages = sorted({page for check in checks if check.status != "passed" for page in check.pages})
    if not failure_pages:
        failure_pages = all_pages(canonical)
    failure_dir.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() == ".pdf":
        render_pdf_failures(source_path, canonical, failure_dir, failure_pages, dpi)
    elif source_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        with Image.open(source_path) as image:
            render_image_failure(image.convert("RGB"), canonical, failure_dir, 1)


def render_pdf_failures(
    source_path: Path,
    canonical: dict[str, Any],
    failure_dir: Path,
    pages: list[int],
    dpi: int,
) -> None:
    import fitz

    page_map = {page["page_number"]: page for page in canonical.get("pages") or []}
    with fitz.open(source_path) as document:
        scale = dpi / 72
        for page_number in pages:
            if page_number < 1 or page_number > len(document):
                continue
            page = document[page_number - 1]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            page_dir = failure_dir / f"page_{page_number}"
            page_dir.mkdir(parents=True, exist_ok=True)
            image.save(page_dir / "page.png")
            draw_debug_overlay(image, page_map.get(page_number) or {}, scale, page_dir)


def render_image_failure(
    image: Image.Image,
    canonical: dict[str, Any],
    failure_dir: Path,
    page_number: int,
) -> None:
    page = next(
        (page for page in canonical.get("pages") or [] if page.get("page_number") == page_number),
        {},
    )
    page_dir = failure_dir / f"page_{page_number}"
    page_dir.mkdir(parents=True, exist_ok=True)
    image.save(page_dir / "page.png")
    draw_debug_overlay(image, page, 1.0, page_dir)


def draw_debug_overlay(image: Image.Image, page: dict[str, Any], native_scale: float, page_dir: Path) -> None:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    metadata = page.get("metadata") or {}
    for region in metadata.get("layout_regions") or []:
        bbox = flat_bbox(region.get("bbox"))
        if bbox:
            draw.rectangle(bbox, outline="red", width=3)
            draw.text((bbox[0] + 3, bbox[1] + 3), str(region.get("type") or "layout"), fill="red")
            crop_bbox = clamp_bbox(bbox, image.size)
            if crop_bbox:
                image.crop(crop_bbox).save(page_dir / f"region_{safe_slug(str(region.get('type') or 'layout'))}_{len(list(page_dir.glob('region_*')))}.png")
    ocr_metadata = metadata.get("ocr_metadata") or {}
    write_json(page_dir / "ocr.json", ocr_metadata)
    for line in ocr_metadata.get("lines") or []:
        bbox = flat_bbox(line.get("bbox"))
        if bbox:
            draw.rectangle(bbox, outline="blue", width=2)
    connector_regions = metadata.get("connector_regions") or []
    write_json(page_dir / "connectors.json", connector_regions)
    for connector_region in connector_regions:
        for connector in connector_region.get("connectors") or []:
            bbox = flat_bbox(connector.get("bbox"))
            if bbox:
                draw.rectangle(bbox, outline="orange", width=3)
            head = connector.get("arrowhead_point")
            if isinstance(head, (list, tuple)) and len(head) >= 2:
                x, y = float(head[0]), float(head[1])
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="magenta", width=3)
    for element in page.get("elements") or []:
        bbox = flat_bbox(element.get("bbox"))
        if bbox:
            scaled = [value * native_scale for value in bbox]
            draw.rectangle(scaled, outline="green", width=2)
    overlay.save(page_dir / "layout_overlay.png")


def flat_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    points = [item for item in value if isinstance(item, (list, tuple)) and len(item) >= 2]
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def clamp_bbox(bbox: list[float], size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    width, height = size
    x0 = max(0, min(width, int(bbox[0])))
    y0 = max(0, min(height, int(bbox[1])))
    x1 = max(0, min(width, int(bbox[2])))
    y1 = max(0, min(height, int(bbox[3])))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def build_summary(
    environment: dict[str, Any],
    manifest: dict[str, Any],
    preflight: dict[str, Any],
    warmup: dict[str, Any],
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = {status: sum(report.get("status") == status for report in reports) for status in STATUS_ORDER}
    status = max((report.get("status", "failed") for report in reports), key=lambda value: STATUS_ORDER[value], default="passed")
    summary = base_summary(environment, manifest)
    summary.update(
        {
            "status": status,
            "phase": "regression",
            "preflight": preflight,
            "model_health": warmup,
            "counts": counts,
            "documents": reports,
        }
    )
    return summary


def base_summary(environment: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": environment.get("git_commit"),
        "fixture_count": len(manifest["documents"]),
    }


def aggregate_timing(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_seconds": round(sum(float(report.get("duration_seconds") or 0.0) for report in reports), 3),
        "documents": [
            {
                "filename": report.get("filename"),
                "status": report.get("status"),
                "duration_seconds": report.get("duration_seconds"),
                "rss_mb": report.get("rss_mb"),
                "gpu": report.get("gpu"),
            }
            for report in reports
        ],
    }


def environment_report() -> dict[str, Any]:
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {},
    }
    for package in PACKAGE_NAMES:
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_mb": round(torch.cuda.get_device_properties(index).total_memory / 1024**2, 2),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        report["torch"] = {"error": str(exc)}
    return report


def current_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 if sys.platform != "darwin" else 1024**2
    return round(float(usage) / divisor, 2)


def reset_gpu_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def gpu_memory_snapshot(include_peak: bool = False) -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        snapshot = {
            "available": True,
            "device": torch.cuda.current_device(),
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
        }
        if include_peak:
            snapshot.update(
                {
                    "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
                    "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
                }
            )
        return snapshot
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def stable_fingerprint(value: Any) -> str:
    cleaned = remove_nondeterministic_fields(copy.deepcopy(value))
    payload = json.dumps(cleaned, ensure_ascii=False, sort_keys=True, default=json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def remove_nondeterministic_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: remove_nondeterministic_fields(child)
            for key, child in value.items()
            if key not in {"timing", "duration_seconds", "elapsed_seconds", "rss_mb", "gpu"}
        }
    if isinstance(value, list):
        return [remove_nondeterministic_fields(child) for child in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    if path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def safe_slug(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in cleaned.split("-") if part) or "document"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def create_archive(output_dir: Path, commit: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = output_dir.parent / f"extraction-regression-debug-{commit[:8]}-{timestamp}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(output_dir.parent))
    return archive


def print_locations(output_dir: Path, archive: Path | None) -> None:
    print(f"REGRESSION_OUTPUT={output_dir}")
    if archive:
        print(f"REGRESSION_ARCHIVE={archive}")


if __name__ == "__main__":
    raise SystemExit(main())
