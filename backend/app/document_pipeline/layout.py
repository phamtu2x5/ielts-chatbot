import tempfile
import threading
from dataclasses import dataclass
from importlib import util
from pathlib import Path
from typing import Any

from PIL import Image

from .config import DocumentPipelineConfig


@dataclass
class LayoutRegion:
    type: str
    confidence: float
    bbox: list[float]
    source: str = "doclayout_yolo"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "source": self.source,
        }


@dataclass
class LayoutResult:
    engine: str
    regions: list[LayoutRegion]
    metadata: dict[str, Any]

    @property
    def ok(self) -> bool:
        return not self.metadata.get("error")

    def region_dicts(self) -> list[dict[str, Any]]:
        return [region.to_dict() for region in self.regions]


class DocLayoutDetector:
    def __init__(self, config: DocumentPipelineConfig) -> None:
        self.config = config
        self._model = None
        self._lock = threading.RLock()

    def detect(self, image: Image.Image) -> LayoutResult:
        if not self.config.layout_enabled:
            return LayoutResult("layout_disabled", [], {"skipped": True})

        with self._lock:
            return self._detect(image)

    def warmup(self) -> dict[str, Any]:
        if not self.config.layout_enabled:
            return {"skipped": True}
        if not self.config.warmup_layout:
            return {"skipped": True, "engine": self.config.layout_engine}

        try:
            self._get_model()
        except Exception as exc:
            return {
                "engine": self.config.layout_engine,
                "ok": False,
                "metadata": {
                    "error": str(exc),
                    "model_repo": self.config.layout_model_repo,
                    "model_path": self.config.layout_model_path or None,
                    "device": self.config.layout_device,
                },
            }
        return {
            "engine": self.config.layout_engine,
            "ok": True,
            "metadata": {
                "model_loaded": True,
                "model_repo": self.config.layout_model_repo,
                "model_path": self.config.layout_model_path or None,
                "device": self.config.layout_device,
            },
        }

    def _detect(self, image: Image.Image) -> LayoutResult:
        if util.find_spec("doclayout_yolo") is None:
            return LayoutResult(
                "doclayout_yolo_unavailable",
                [],
                {"error": "doclayout_yolo_not_installed"},
            )

        image_path = self._save_temp_image(image)
        try:
            model = self._get_model()
            raw_result = model.predict(
                str(image_path),
                imgsz=self.config.layout_image_size,
                conf=self.config.layout_confidence,
                device=self.config.layout_device,
            )
            regions = self._extract_regions(raw_result)
            return LayoutResult(
                "doclayout_yolo",
                regions,
                {
                    "model_repo": self.config.layout_model_repo,
                    "model_path": self.config.layout_model_path or None,
                    "device": self.config.layout_device,
                    "image_size": self.config.layout_image_size,
                    "confidence_threshold": self.config.layout_confidence,
                },
            )
        except Exception as exc:
            return LayoutResult("doclayout_yolo_error", [], {"error": str(exc)})
        finally:
            image_path.unlink(missing_ok=True)

    def _get_model(self):
        if self._model is not None:
            return self._model

        from doclayout_yolo import YOLOv10

        if self.config.layout_model_path:
            self._model = YOLOv10(self.config.layout_model_path)
        else:
            self._model = YOLOv10.from_pretrained(self.config.layout_model_repo)
        return self._model

    def _extract_regions(self, raw_result: Any) -> list[LayoutRegion]:
        regions: list[LayoutRegion] = []
        for result in self._as_list(raw_result):
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy_values = self._as_list(getattr(boxes, "xyxy", None))
            conf_values = self._as_list(getattr(boxes, "conf", None))
            cls_values = self._as_list(getattr(boxes, "cls", None))

            for index, bbox in enumerate(xyxy_values):
                bbox_values = self._as_list(bbox)[:4]
                if len(bbox_values) < 4:
                    continue
                cls_id = self._int_at(cls_values, index)
                confidence = self._float_at(conf_values, index)
                region_type = str(names.get(cls_id, cls_id if cls_id is not None else "unknown"))
                regions.append(
                    LayoutRegion(
                        type=region_type,
                        confidence=max(0.0, min(1.0, confidence)),
                        bbox=[float(value) for value in bbox_values],
                    )
                )
        return regions

    def _save_temp_image(self, image: Image.Image) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            path = Path(handle.name)
        image.convert("RGB").save(path)
        return path

    def _as_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _float_at(self, values: list[Any], index: int) -> float:
        try:
            return float(values[index])
        except (IndexError, TypeError, ValueError):
            return 0.0

    def _int_at(self, values: list[Any], index: int) -> int | None:
        try:
            return int(values[index])
        except (IndexError, TypeError, ValueError):
            return None
