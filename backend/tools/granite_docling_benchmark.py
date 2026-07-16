from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.document_pipeline.chunking import SemanticChunker
from app.document_pipeline.config import DocumentPipelineConfig
from app.document_pipeline.ielts import IELTSStructureParser, StructuredChunker
from app.document_pipeline.models import DocumentElement, ProcessedDocument, ProcessedPage
from app.document_pipeline.normalization import normalize_text
from app.document_pipeline.visual import WritingTaskTableParser
from tools.extraction_regression import (
    aggregate_timing,
    base_summary,
    create_archive,
    current_rss_mb,
    environment_report,
    evaluate_document,
    gpu_memory_snapshot,
    load_manifest,
    prepare_output_dir,
    reset_gpu_peak_memory,
    render_failure_artifacts,
    safe_slug,
    sha256_file,
    stable_fingerprint,
    status_from_checks,
    verify_fixtures,
    write_json,
)


MODEL_ID = "ibm-granite/granite-docling-258M"
BENCHMARK_SCHEMA_VERSION = "1.0.0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Granite-Docling against an existing extraction regression run."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BACKEND_DIR / "tests" / "fixtures" / "extraction_manifest.json",
    )
    parser.add_argument("--corpus-dir", type=Path, default=REPO_ROOT / "docs")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-path", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    environment = environment_report()
    environment["benchmark"] = {
        "engine": "granite_docling",
        "model_id": MODEL_ID,
        "dtype": select_granite_dtype(args.dtype),
        "production_integrated": False,
        "packages": {
            name: package_version(name)
            for name in ("docling", "docling-core", "transformers")
        },
    }
    write_json(output_dir / "environment.json", environment)

    preflight = verify_fixtures(manifest, args.corpus_dir)
    write_json(output_dir / "preflight.json", preflight)
    if not preflight["ok"]:
        summary = base_summary(environment, manifest)
        summary.update({"status": "failed", "phase": "preflight", "preflight": preflight})
        return finish(args, output_dir, summary, [])

    model_health: dict[str, Any]
    try:
        converter, model_health = create_converter(environment["benchmark"]["dtype"])
    except Exception as exc:
        model_health = {
            "ok": False,
            "model_id": MODEL_ID,
            "dtype": environment["benchmark"]["dtype"],
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "model_health.json", model_health)
        summary = base_summary(environment, manifest)
        summary.update({"status": "failed", "phase": "model_initialization", "model_health": model_health})
        return finish(args, output_dir, summary, [])

    write_json(output_dir / "model_health.json", model_health)
    reports = [
        run_granite_document(
            converter,
            args.corpus_dir / fixture["filename"],
            fixture,
            output_dir,
            environment["benchmark"]["dtype"],
        )
        for fixture in manifest["documents"]
    ]

    model_health["inference_ok"] = any(report.get("conversion_ok") for report in reports)
    model_health["peak_allocated_mb"] = max(
        (float((report.get("gpu") or {}).get("peak_allocated_mb") or 0.0) for report in reports),
        default=0.0,
    )
    model_health["cuda_inference_observed"] = model_health["peak_allocated_mb"] > 0
    model_health["ok"] = bool(
        model_health.get("model_loaded")
        and model_health["inference_ok"]
        and model_health["cuda_inference_observed"]
    )
    write_json(output_dir / "model_health.json", model_health)

    counts = {
        status: sum(report.get("status") == status for report in reports)
        for status in ("passed", "unsupported", "degraded", "failed")
    }
    summary = base_summary(environment, manifest)
    summary.update(
        {
            "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
            "status": worst_status(reports) if model_health["ok"] else "failed",
            "phase": "regression",
            "preflight": preflight,
            "model_health": model_health,
            "counts": counts,
            "documents": reports,
        }
    )
    write_json(output_dir / "regression_summary.json", summary)
    write_json(output_dir / "timing.json", aggregate_timing(reports))

    comparison = build_ab_comparison(args.baseline_dir, output_dir)
    write_json(args.comparison_path, comparison)
    return finish(args, output_dir, summary, reports)


def create_converter(dtype: str) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
    from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    engine_options = TransformersVlmEngineOptions(torch_dtype=dtype)
    vlm_options = VlmConvertOptions.from_preset(
        "granite_docling",
        engine_options=engine_options,
    )
    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
    )
    format_option = PdfFormatOption(
        pipeline_cls=VlmPipeline,
        pipeline_options=pipeline_options,
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: format_option,
            InputFormat.IMAGE: format_option,
        }
    )
    converter.initialize_pipeline(InputFormat.PDF)
    return converter, {
        "ok": False,
        "model_loaded": True,
        "inference_ok": False,
        "model_id": MODEL_ID,
        "runtime": "transformers",
        "device": "cuda",
        "dtype": dtype,
        "initialization_seconds": round(time.perf_counter() - started, 3),
        "gpu": gpu_memory_snapshot(include_peak=True),
    }


def select_granite_dtype(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Granite benchmark requires a CUDA runtime.")
        major, _ = torch.cuda.get_device_capability(0)
        return "bfloat16" if major >= 8 else "float32"
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to select the Granite dtype.") from exc


def run_granite_document(
    converter: Any,
    source_path: Path,
    fixture: dict[str, Any],
    output_root: Path,
    dtype: str,
) -> dict[str, Any]:
    document_dir = output_root / "documents" / safe_slug(source_path.stem)
    document_dir.mkdir(parents=True, exist_ok=True)
    reset_gpu_peak_memory()
    started = time.perf_counter()
    try:
        result = converter.convert(source_path)
        conversion_seconds = round(time.perf_counter() - started, 3)
        raw_document = result.document.export_to_dict()
        markdown = result.document.export_to_markdown(page_break_placeholder="\n\n[Page Break]\n\n")
        doctags = result.document.export_to_doctags()
        processed = processed_document_from_docling(
            result.document,
            source_path,
            fixture,
            dtype=dtype,
        )
        config = DocumentPipelineConfig()
        structured = IELTSStructureParser(config).parse(processed)
        chunks = StructuredChunker(config).chunk(processed, structured)
        if not chunks:
            chunks = SemanticChunker(config).chunk(processed)
        canonical = processed.to_dict()
        chunk_data = [chunk.to_dict() for chunk in chunks]
        checks = evaluate_document(canonical, chunk_data, fixture)
        report = {
            "filename": fixture["filename"],
            "kind": fixture["kind"],
            "status": status_from_checks(checks),
            "conversion_ok": True,
            "duration_seconds": conversion_seconds,
            "rss_mb": current_rss_mb(),
            "gpu": gpu_memory_snapshot(include_peak=True),
            "conversion_status": enum_value(getattr(result, "status", "unknown")),
            "page_timings": page_vlm_timings(result),
            "checks": [check.to_dict() for check in checks],
            "fingerprints": {
                "docling_document": stable_fingerprint(raw_document),
                "canonical_document": stable_fingerprint(canonical),
                "structure": stable_fingerprint(canonical.get("metadata", {}).get("ielts_structure") or {}),
                "chunks": stable_fingerprint(chunk_data),
            },
        }
        write_json(document_dir / "docling_document.json", raw_document)
        (document_dir / "docling.md").write_text(markdown, encoding="utf-8")
        (document_dir / "docling.doctags.txt").write_text(doctags, encoding="utf-8")
        write_json(document_dir / "canonical_document.json", canonical)
        write_json(document_dir / "chunks.json", chunk_data)
        write_json(document_dir / "conversion_timings.json", conversion_timings(result))
        write_json(document_dir / "vlm_responses.json", page_vlm_responses(result))
        write_json(document_dir / "report.json", report)
        if report["status"] != "passed":
            render_failure_artifacts(
                source_path,
                canonical,
                document_dir / "failures",
                checks,
                config.ocr_dpi,
            )
        return report
    except Exception as exc:
        report = {
            "filename": fixture["filename"],
            "kind": fixture["kind"],
            "status": "failed",
            "conversion_ok": False,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "rss_mb": current_rss_mb(),
            "gpu": gpu_memory_snapshot(include_peak=True),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "checks": [],
        }
        write_json(document_dir / "report.json", report)
        (document_dir / "regression.log").write_text(report["traceback"], encoding="utf-8")
        return report


def processed_document_from_docling(
    docling_document: Any,
    source_path: Path,
    fixture: dict[str, Any],
    dtype: str,
) -> ProcessedDocument:
    config = DocumentPipelineConfig()
    page_numbers = sorted(int(number) for number in getattr(docling_document, "pages", {}) or {1: None})
    full_markdown = docling_document.export_to_markdown(page_break_placeholder="\n\n[Page Break]\n\n")
    pages: list[ProcessedPage] = []
    for page_number in page_numbers:
        page_markdown = docling_document.export_to_markdown(page_no=page_number)
        if not page_markdown and len(page_numbers) == 1:
            page_markdown = full_markdown
        parser_text = normalize_docling_markdown(page_markdown)
        elements = []
        if parser_text:
            elements.append(
                DocumentElement(
                    element_id=f"p{page_number}-e1",
                    page=page_number,
                    type="paragraph",
                    raw_text=parser_text,
                    normalized_text=normalize_text(parser_text),
                    source="granite_docling",
                    confidence=1.0,
                    metadata={"model_id": MODEL_ID, "dtype": dtype},
                )
            )
        pages.append(
            ProcessedPage(
                page_number=page_number,
                processing_route="granite_docling_vlm",
                quality_score=1.0 if parser_text else 0.0,
                elements=elements,
                metadata={"model_id": MODEL_ID, "dtype": dtype},
            )
        )

    metadata: dict[str, Any] = {
        "page_count": len(pages),
        "languages": [],
        "benchmark_engine": "granite_docling",
        "benchmark_model": MODEL_ID,
        "benchmark_dtype": dtype,
        "production_integrated": False,
    }
    if fixture.get("kind") == "ielts_writing_task_1_image":
        parsed_visual = WritingTaskTableParser().parse(normalize_text(full_markdown))
        if parsed_visual:
            metadata.update(
                {
                    "document_type": parsed_visual.document_type,
                    "task_type": parsed_visual.task_type,
                    "visual_structure": {
                        "prompt": parsed_visual.prompt,
                        "visual_elements": [parsed_visual.table],
                    },
                }
            )

    return ProcessedDocument(
        document_id=sha256_file(source_path),
        filename=fixture["filename"],
        mime_type=fixture.get("mime_type") or "application/octet-stream",
        parser_version=f"{config.parser_version}+granite-docling-benchmark",
        metadata=metadata,
        pages=pages,
    )


def normalize_docling_markdown(markdown: str) -> str:
    lines = []
    for line in (markdown or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.lstrip().startswith("<!-- image -->"):
            continue
        lines.append(line.removeprefix("###### ").removeprefix("##### ").removeprefix("#### ").removeprefix("### ").removeprefix("## ").removeprefix("# "))
    return "\n".join(lines).strip()


def page_vlm_timings(result: Any) -> list[dict[str, Any]]:
    rows = []
    for index, page in enumerate(getattr(result, "pages", []) or [], 1):
        response = getattr(getattr(page, "predictions", None), "vlm_response", None)
        rows.append(
            {
                "page": index,
                "generation_seconds": getattr(response, "generation_time", None),
                "response_characters": len(getattr(response, "text", "") or ""),
            }
        )
    return rows


def page_vlm_responses(result: Any) -> list[dict[str, Any]]:
    responses = []
    for index, page in enumerate(getattr(result, "pages", []) or [], 1):
        response = getattr(getattr(page, "predictions", None), "vlm_response", None)
        responses.append(
            {
                "page": index,
                "generation_seconds": getattr(response, "generation_time", None),
                "text": getattr(response, "text", None),
            }
        )
    return responses


def conversion_timings(result: Any) -> dict[str, Any]:
    values = {}
    for name, timing in (getattr(result, "timings", {}) or {}).items():
        values[str(name)] = {
            "times": list(getattr(timing, "times", []) or []),
            "scope": enum_value(getattr(timing, "scope", None)),
        }
    return values


def build_ab_comparison(baseline_dir: Path, granite_dir: Path) -> dict[str, Any]:
    baseline = read_summary(baseline_dir)
    granite = read_summary(granite_dir)
    baseline_documents = {item["filename"]: item for item in baseline.get("documents", [])}
    granite_documents = {item["filename"]: item for item in granite.get("documents", [])}
    filenames = sorted(set(baseline_documents) | set(granite_documents))
    documents = []
    for filename in filenames:
        current = baseline_documents.get(filename) or {}
        candidate = granite_documents.get(filename) or {}
        current_checks = {item["name"]: item for item in current.get("checks", [])}
        candidate_checks = {item["name"]: item for item in candidate.get("checks", [])}
        check_names = sorted(set(current_checks) | set(candidate_checks))
        documents.append(
            {
                "filename": filename,
                "current_pipeline": compact_result(current),
                "granite_docling": compact_result(candidate),
                "checks": [
                    {
                        "name": name,
                        "current_pipeline": current_checks.get(name, {}).get("status", "missing"),
                        "granite_docling": candidate_checks.get(name, {}).get("status", "missing"),
                    }
                    for name in check_names
                ],
            }
        )
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "production_integrated": False,
        "current_pipeline_status": baseline.get("status"),
        "granite_docling_status": granite.get("status"),
        "documents": documents,
    }


def compact_result(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status", "missing"),
        "duration_seconds": report.get("duration_seconds"),
        "rss_mb": report.get("rss_mb"),
        "gpu": report.get("gpu"),
    }


def read_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "regression_summary.json" if path.is_dir() else path
    if not summary_path.exists():
        raise FileNotFoundError(f"Regression summary not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def worst_status(reports: list[dict[str, Any]]) -> str:
    order = {"passed": 0, "unsupported": 1, "degraded": 2, "failed": 3}
    return max((report.get("status", "failed") for report in reports), key=lambda item: order[item], default="passed")


def finish(
    args: argparse.Namespace,
    output_dir: Path,
    summary: dict[str, Any],
    reports: list[dict[str, Any]],
) -> int:
    write_json(output_dir / "regression_summary.json", summary)
    if reports:
        write_json(output_dir / "timing.json", aggregate_timing(reports))
    try:
        comparison = build_ab_comparison(args.baseline_dir, output_dir)
        write_json(args.comparison_path, comparison)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    archive_root = (args.archive_root or output_dir).resolve()
    if not args.no_archive:
        create_archive(archive_root, summary.get("git_commit") or "unknown")
    return 0 if summary.get("status") == "passed" else 1


def enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
