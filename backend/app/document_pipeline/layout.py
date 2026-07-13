import tempfile
import threading
from html.parser import HTMLParser
from importlib import metadata
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .config import DocumentPipelineConfig
from .ielts import IELTSDocument, IELTSQuestionGroup
from .models import ProcessedDocument


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            self._current_row.append(" ".join(" ".join(self._current_cell).split()))
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None


class PPStructureProcessor:
    """Optional PP-StructureV3 integration for visual question groups.

    The processor is intentionally best-effort. It only runs for pages that
    already contain table/flowchart question groups and never blocks ingestion
    when PP-StructureV3 is unavailable in the runtime.
    """

    def __init__(self, config: DocumentPipelineConfig) -> None:
        self.config = config
        self._model = None
        self._lock = threading.RLock()

    def enrich_pdf(
        self,
        file_path: Path,
        document: ProcessedDocument,
        structured: IELTSDocument,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "enabled": self.config.enable_pp_structure,
            "status": "skipped",
            "pages": [],
            "visual_groups": [],
        }
        groups = self._visual_groups(structured)
        if not groups:
            report["reason"] = "no_visual_question_groups"
            document.metadata["layout_report"] = report
            return report
        if not self.config.enable_pp_structure:
            report["reason"] = "disabled"
            self._mark_groups(groups, status="disabled", reason="DOCUMENT_ENABLE_PP_STRUCTURE=false")
            document.metadata["layout_report"] = report
            document.metadata["ielts_structure"] = structured.to_dict()
            return report

        try:
            import fitz
        except ImportError as exc:
            report.update({"status": "unavailable", "reason": "pymupdf_not_installed", "error": str(exc)})
            self._mark_groups(groups, status="unavailable", reason="pymupdf_not_installed")
            document.metadata["layout_report"] = report
            document.metadata["ielts_structure"] = structured.to_dict()
            return report

        try:
            with fitz.open(file_path) as pdf:
                page_predictions: dict[int, dict[str, Any]] = {}
                for page_number in sorted({page for group in groups for page in group.page_numbers}):
                    if page_number < 1 or page_number > len(pdf):
                        continue
                    prediction = self._predict_page(pdf[page_number - 1])
                    page_predictions[page_number] = prediction
                    report["pages"].append(
                        {
                            "page": page_number,
                            "status": prediction["status"],
                            "structures_found": len(prediction.get("structures") or []),
                            "error": prediction.get("error"),
                        }
                    )
        except Exception as exc:
            report.update({"status": "error", "error": str(exc)})
            self._mark_groups(groups, status="error", reason=str(exc))
            document.metadata["layout_report"] = report
            document.metadata["ielts_structure"] = structured.to_dict()
            return report

        enriched = 0
        for group in groups:
            merged = self._enrich_group_from_predictions(group, page_predictions)
            report["visual_groups"].append(
                {
                    "question_range": [group.question_start, group.question_end],
                    "question_type": group.question_type,
                    "visual_type": (group.visual_element or {}).get("type"),
                    "layout_status": (group.visual_element or {}).get("layout_status"),
                    "layout_source": (group.visual_element or {}).get("layout_source"),
                }
            )
            if merged:
                enriched += 1

        report["status"] = "ok" if enriched else "no_structured_visuals"
        report["enriched_visual_groups"] = enriched
        document.metadata["layout_report"] = report
        document.metadata["ielts_structure"] = structured.to_dict()
        return report

    def warmup(self) -> dict[str, Any]:
        if not self.config.enable_pp_structure or not self.config.warmup_pp_structure:
            return {"skipped": True}
        try:
            with self._lock:
                self._get_model()
            raw = self._predict_image(self._warmup_image())
            structures = self._extract_structures(raw)
            return {
                "ok": True,
                "engine": "pp_structure_v3",
                "runtime": self._runtime_diagnostics(),
                "model_loaded": True,
                "inference_ok": True,
                "structures_found": len(structures),
                "table_structures_found": sum(1 for structure in structures if structure.get("type") == "table"),
            }
        except Exception as exc:
            return {
                "ok": False,
                "engine": "pp_structure_v3",
                "runtime": self._runtime_diagnostics(),
                "model_loaded": self._model is not None,
                "inference_ok": False,
                "error": str(exc),
            }

    def _visual_groups(self, structured: IELTSDocument) -> list[IELTSQuestionGroup]:
        return [
            group
            for passage in structured.passages
            for group in passage.question_groups
            if group.question_type in {"table_completion", "flowchart_completion"} and group.visual_element
        ]

    def _mark_groups(self, groups: list[IELTSQuestionGroup], status: str, reason: str) -> None:
        for group in groups:
            if group.visual_element is None:
                continue
            group.visual_element["layout_status"] = status
            group.visual_element["layout_reason"] = reason

    def _predict_page(self, page) -> dict[str, Any]:
        image = self._render_page(page)
        try:
            raw = self._predict_image(image)
        except Exception as exc:
            return {"status": "error", "error": str(exc), "structures": []}
        structures = self._extract_structures(raw)
        return {"status": "ok", "structures": structures}

    def _render_page(self, page) -> Image.Image:
        import fitz

        matrix = fitz.Matrix(self.config.pp_structure_dpi / 72, self.config.pp_structure_dpi / 72)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    def _warmup_image(self) -> Image.Image:
        image = Image.new("RGB", (640, 220), "white")
        draw = ImageDraw.Draw(image)
        x_positions = [24, 180, 360, 600]
        y_positions = [32, 88, 144, 196]
        for x in x_positions:
            draw.line((x, y_positions[0], x, y_positions[-1]), fill="black", width=2)
        for y in y_positions:
            draw.line((x_positions[0], y, x_positions[-1], y), fill="black", width=2)
        draw.text((44, 52), "Country", fill="black")
        draw.text((204, 52), "Internet 2019", fill="black")
        draw.text((384, 52), "Internet 2024", fill="black")
        draw.text((44, 108), "A", fill="black")
        draw.text((204, 108), "78", fill="black")
        draw.text((384, 108), "96", fill="black")
        draw.text((44, 164), "B", fill="black")
        draw.text((204, 164), "61", fill="black")
        draw.text((384, 164), "89", fill="black")
        return image

    def _predict_image(self, image: Image.Image) -> Any:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            image_path = Path(handle.name)
        try:
            image.save(image_path)
            with self._lock:
                model = self._get_model()
            if hasattr(model, "predict"):
                return model.predict(str(image_path))
            if callable(model):
                return model(str(image_path))
            raise RuntimeError("PP-StructureV3 model has no predict interface.")
        finally:
            image_path.unlink(missing_ok=True)

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from paddleocr import PPStructureV3
        except ImportError as exc:
            raise RuntimeError("PPStructureV3 is not available in the installed paddleocr package.") from exc

        try:
            self._model = PPStructureV3(device=self.config.pp_structure_device)
        except TypeError:
            self._model = PPStructureV3()
        return self._model

    def _runtime_diagnostics(self) -> dict[str, Any]:
        packages: dict[str, str | None] = {}
        for package_name in ["paddlepaddle", "paddlepaddle-gpu", "paddleocr", "paddlex"]:
            try:
                packages[package_name] = metadata.version(package_name)
            except metadata.PackageNotFoundError:
                packages[package_name] = None
        return {"packages": packages, "device": self.config.pp_structure_device}

    def _extract_structures(self, raw: Any) -> list[dict[str, Any]]:
        structures: list[dict[str, Any]] = []
        for item in self._walk(raw):
            if not isinstance(item, dict):
                continue
            label = str(item.get("type") or item.get("label") or item.get("category") or "").lower()
            html = item.get("html") or item.get("table_html")
            cells = item.get("cells") or item.get("table_cells")
            bbox = item.get("bbox") or item.get("box")
            if html or cells or "table" in label:
                table = self._table_from_item(item)
                if table:
                    structures.append(table)
            elif "flow" in label or "chart" in label or "diagram" in label:
                structures.append(
                    {
                        "type": "flowchart",
                        "nodes": [],
                        "edges": [],
                        "bbox": self._coerce_bbox(bbox),
                        "source": "pp_structure_v3",
                        "confidence": self._coerce_confidence(item),
                        "raw": self._safe_raw(item),
                    }
                )
        return structures

    def _table_from_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        html = item.get("html") or item.get("table_html")
        rows = self._rows_from_html(str(html)) if html else []
        if not rows:
            rows = self._rows_from_cells(item.get("cells") or item.get("table_cells"))
        if not rows:
            return None

        columns = rows[0]
        body_rows = rows[1:] if len(rows) > 1 else []
        return {
            "type": "table",
            "columns": columns,
            "rows": body_rows,
            "bbox": self._coerce_bbox(item.get("bbox") or item.get("box")),
            "source": "pp_structure_v3",
            "confidence": self._coerce_confidence(item),
            "raw": self._safe_raw(item),
        }

    def _rows_from_html(self, html: str) -> list[list[str]]:
        parser = _TableHTMLParser()
        try:
            parser.feed(html)
        except Exception:
            return []
        return parser.rows

    def _rows_from_cells(self, cells: Any) -> list[list[str]]:
        if not isinstance(cells, list):
            return []
        rows: dict[int, dict[int, str]] = {}
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            row_index = self._coerce_index(cell.get("row") or cell.get("row_index"))
            col_index = self._coerce_index(cell.get("col") or cell.get("col_index"))
            if row_index is None or col_index is None:
                continue
            text = str(cell.get("text") or cell.get("content") or "").strip()
            rows.setdefault(row_index, {})[col_index] = text
        return [
            [cols.get(index, "") for index in range(max(cols.keys()) + 1)]
            for _, cols in sorted(rows.items())
            if cols
        ]

    def _enrich_group_from_predictions(
        self,
        group: IELTSQuestionGroup,
        page_predictions: dict[int, dict[str, Any]],
    ) -> bool:
        visual = group.visual_element or {}
        visual_type = visual.get("type")
        for page_number in group.page_numbers:
            prediction = page_predictions.get(page_number)
            if not prediction:
                continue
            for structure in prediction.get("structures") or []:
                if structure.get("type") != visual_type:
                    continue
                if visual_type == "table" and structure.get("columns") and structure.get("rows"):
                    self._merge_table_visual(group, structure, page_number)
                    return True
                if visual_type == "flowchart":
                    self._merge_flowchart_visual(group, structure, page_number)
                    return True

        if group.visual_element is not None:
            group.visual_element.setdefault("layout_status", "no_match")
            group.visual_element.setdefault("layout_source", "pp_structure_v3")
        return False

    def _merge_table_visual(self, group: IELTSQuestionGroup, table: dict[str, Any], page_number: int) -> None:
        visual = group.visual_element or {}
        visual.update(
            {
                "columns": table["columns"],
                "rows": table["rows"],
                "bbox": table.get("bbox") or visual.get("bbox") or [],
                "source": "pp_structure_v3",
                "confidence": max(float(visual.get("confidence") or 0.0), float(table.get("confidence") or 0.0)),
                "layout_status": "enriched",
                "layout_source": "pp_structure_v3",
                "layout_page": page_number,
                "layout_raw": table.get("raw"),
            }
        )
        group.visual_element = visual

    def _merge_flowchart_visual(self, group: IELTSQuestionGroup, flowchart: dict[str, Any], page_number: int) -> None:
        visual = group.visual_element or {}
        visual.update(
            {
                "bbox": flowchart.get("bbox") or visual.get("bbox") or [],
                "source": "pp_structure_v3",
                "confidence": max(float(visual.get("confidence") or 0.0), float(flowchart.get("confidence") or 0.0)),
                "layout_status": "enriched",
                "layout_source": "pp_structure_v3",
                "layout_page": page_number,
                "layout_raw": flowchart.get("raw"),
            }
        )
        group.visual_element = visual

    def _walk(self, value: Any):
        yield value
        if hasattr(value, "res"):
            yield from self._walk(value.res)
            return
        if hasattr(value, "json"):
            try:
                json_value = value.json() if callable(value.json) else value.json
                yield from self._walk(json_value)
            except (TypeError, ValueError):
                pass
            return
        if isinstance(value, dict):
            for child in value.values():
                yield from self._walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from self._walk(child)

    def _coerce_bbox(self, value: Any) -> list[float]:
        if not isinstance(value, (list, tuple)):
            return []
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            return [float(item) for item in value]
        return []

    def _coerce_confidence(self, item: dict[str, Any]) -> float:
        for key in ("confidence", "score", "prob"):
            if key in item:
                try:
                    value = float(item[key])
                    return max(0.0, min(1.0, value / 100 if value > 1 else value))
                except (TypeError, ValueError):
                    pass
        return 0.75

    def _coerce_index(self, value: Any) -> int | None:
        if value is None:
            return 0
        try:
            index = int(float(value))
        except (TypeError, ValueError):
            return None
        return index if index >= 0 else None

    def _safe_raw(self, item: dict[str, Any]) -> dict[str, Any]:
        safe = {}
        for key in ("type", "label", "category", "bbox", "box", "html", "table_html"):
            if key in item:
                safe[key] = item[key]
        return safe
