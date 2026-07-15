import tempfile
import threading
from dataclasses import dataclass
from importlib import util
from pathlib import Path
from typing import Any, Dict, Iterable

from PIL import Image

from .config import DocumentPipelineConfig
from .normalization import normalize_text


@dataclass
class OCRResult:
    text: str
    confidence: float
    engine: str
    metadata: Dict[str, Any]


class OCRProcessor:
    def __init__(self, config: DocumentPipelineConfig) -> None:
        self.config = config
        self._rapid_ocr = None
        self._lock = threading.RLock()

    def image_to_text(self, image: Image.Image) -> OCRResult:
        with self._lock:
            return self._image_to_text(image)

    def _image_to_text(self, image: Image.Image) -> OCRResult:
        result = self._image_to_text_with_rapidocr(image)
        if result.text:
            return result

        return OCRResult(
            "",
            0.0,
            "rapidocr_failed",
            {
                "reason": "rapidocr_failed",
                "attempt": self._attempt_summary(result),
            },
        )

    def _attempt_summary(self, result: OCRResult) -> Dict[str, Any]:
        return {
            "engine": result.engine,
            "confidence": result.confidence,
            "has_text": bool(result.text),
            "error": result.metadata.get("error"),
        }

    def warmup(self) -> Dict[str, Any]:
        with self._lock:
            image = Image.new("RGB", (420, 96), "white")
            from PIL import ImageDraw

            draw = ImageDraw.Draw(image)
            draw.text((16, 32), "IELTS OCR warmup", fill="black")

            model = self._image_to_text_with_rapidocr(image)
            return {
                "engine": self.config.ocr_engine,
                "runtime": self.config.ocr_runtime,
                "device": self.config.ocr_device,
                "model": {
                    "engine": model.engine,
                    "confidence": model.confidence,
                    "ok": bool(model.text),
                    "metadata": model.metadata,
                },
                "models_ready": bool(model.text),
            }

    def _rapidocr_is_available(self) -> bool:
        return util.find_spec("rapidocr") is not None

    def _image_to_text_with_rapidocr(self, image: Image.Image) -> OCRResult:
        if not self._rapidocr_is_available():
            return OCRResult("", 0.0, "rapidocr_unavailable", {"reason": "rapidocr_not_installed"})

        try:
            ocr = self._get_rapidocr()
            image_path = self._save_temp_image(image)
            try:
                raw_result = ocr(str(image_path))
            finally:
                image_path.unlink(missing_ok=True)
            texts, scores, boxes = self._extract_rapidocr_result(raw_result)
        except Exception as exc:
            return OCRResult("", 0.0, "rapidocr_error", {"error": str(exc)})

        text = normalize_text("\n".join(texts))
        confidence = sum(scores) / len(scores) if scores else (0.6 if text else 0.0)
        return OCRResult(
            text=text,
            confidence=max(0.0, min(1.0, confidence)),
            engine=(
                f"rapidocr_{self.config.ocr_version.lower()}_"
                f"{self.config.ocr_model_size.lower()}_{self.config.ocr_runtime.lower()}"
            ),
            metadata={
                "word_count": len(texts),
                "lang": self.config.ocr_lang,
                "det_lang": self.config.ocr_det_lang,
                "runtime": self.config.ocr_runtime,
                "device": self.config.ocr_device,
                "ocr_version": self.config.ocr_version,
                "model_size": self.config.ocr_model_size,
                "boxes": boxes,
            },
        )

    def _get_rapidocr(self):
        if self._rapid_ocr is not None:
            return self._rapid_ocr

        from rapidocr import RapidOCR

        self._validate_cuda_runtime()
        params = self._rapidocr_params()
        self._rapid_ocr = RapidOCR(params=params)
        return self._rapid_ocr

    def _rapidocr_params(self) -> Dict[str, Any]:
        from rapidocr.utils.typings import EngineType, LangDet, LangRec, ModelType, OCRVersion

        return {
            "Global.use_cls": False,
            "EngineConfig.torch.use_cuda": True,
            "EngineConfig.torch.cuda_ep_cfg.device_id": self._cuda_device_id(),
            "Det.engine_type": EngineType.TORCH,
            "Det.lang_type": LangDet(self.config.ocr_det_lang),
            "Det.model_type": ModelType(self.config.ocr_model_size),
            "Det.ocr_version": OCRVersion(self.config.ocr_version),
            "Rec.engine_type": EngineType.TORCH,
            "Rec.lang_type": LangRec(self.config.ocr_lang),
            "Rec.model_type": ModelType(self.config.ocr_model_size),
            "Rec.ocr_version": OCRVersion(self.config.ocr_version),
        }

    def _cuda_device_id(self) -> int:
        _, _, device_id = self.config.ocr_device.partition(":")
        return int(device_id) if device_id else 0

    def _validate_cuda_runtime(self) -> None:
        if util.find_spec("torch") is None:
            raise RuntimeError("PyTorch is required for RapidOCR GPU inference.")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for RapidOCR.")
        device_id = self._cuda_device_id()
        if device_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"OCR_DEVICE requests cuda:{device_id}, but only {torch.cuda.device_count()} CUDA device(s) are available."
            )

    def _save_temp_image(self, image: Image.Image) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            path = Path(handle.name)
        image.convert("RGB").save(path)
        return path

    def _extract_rapidocr_result(self, raw_result: Any) -> tuple[list[str], list[float], list[Any]]:
        texts = self._tuple_or_list(getattr(raw_result, "txts", None))
        scores = [self._coerce_score(score) for score in self._tuple_or_list(getattr(raw_result, "scores", None))]
        boxes = self._as_python(getattr(raw_result, "boxes", None))

        if not texts:
            for item in self._walk_rapidocr_result(raw_result):
                if isinstance(item, dict):
                    texts.extend(str(text).strip() for text in self._tuple_or_list(item.get("txts")) if str(text).strip())
                    texts.extend(str(text).strip() for text in self._tuple_or_list(item.get("texts")) if str(text).strip())
                    scores.extend(self._coerce_score(score) for score in self._tuple_or_list(item.get("scores")))

        return [text for text in texts if text], scores, boxes if isinstance(boxes, list) else []

    def _walk_rapidocr_result(self, value: Any) -> Iterable[Any]:
        yield value
        if hasattr(value, "__dict__"):
            yield from self._walk_rapidocr_result(vars(value))
            return
        if isinstance(value, dict):
            for child in value.values():
                yield from self._walk_rapidocr_result(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from self._walk_rapidocr_result(child)

    def _tuple_or_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        if hasattr(value, "tolist"):
            converted = value.tolist()
            return converted if isinstance(converted, list) else [converted]
        return []

    def _as_python(self, value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    def _coerce_score(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if score > 1:
            score /= 100
        return max(0.0, min(1.0, score))
