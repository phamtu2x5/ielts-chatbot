import os
import tempfile
import threading
from dataclasses import dataclass
from importlib import util
from pathlib import Path
from typing import Any, Dict, Iterable

from PIL import Image

from .config import DocumentPipelineConfig
from .normalization import normalize_text


# PaddleOCR on Colab CPU can hit a Paddle oneDNN/PIR runtime error before
# inference starts. Force-disable those CPU paths before the first Paddle
# import by default; advanced users can opt out with PADDLEOCR_DISABLE_ONEDNN=0.
if os.getenv("PADDLEOCR_DISABLE_ONEDNN", "1").lower() not in {"0", "false", "no"}:
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_use_onednn"] = "0"
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["FLAGS_enable_pir_in_executor"] = "0"


@dataclass
class OCRResult:
    text: str
    confidence: float
    engine: str
    metadata: Dict[str, Any]


class OCRProcessor:
    def __init__(self, config: DocumentPipelineConfig) -> None:
        self.config = config
        self._paddle_ocr = None
        self._lock = threading.RLock()

    def image_to_text(self, image: Image.Image) -> OCRResult:
        with self._lock:
            return self._image_to_text(image)

    def _image_to_text(self, image: Image.Image) -> OCRResult:
        result = self._image_to_text_with_paddle(image)
        if result.text:
            return result

        return OCRResult(
            "",
            0.0,
            "paddleocr_failed",
            {
                "reason": "paddleocr_failed",
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

            results = {
                "engine": self.config.ocr_engine,
                "model": None,
            }
            model = self._image_to_text_with_paddle(image)
            results["model"] = {
                "engine": model.engine,
                "confidence": model.confidence,
                "ok": bool(model.text),
                "metadata": model.metadata,
            }
            results["models_ready"] = bool(results["model"]["ok"])
            return results

    def _paddle_is_available(self) -> bool:
        return util.find_spec("paddleocr") is not None

    def _image_to_text_with_paddle(self, image: Image.Image) -> OCRResult:
        if not self._paddle_is_available():
            return OCRResult("", 0.0, "paddleocr_unavailable", {"reason": "paddleocr_not_installed"})

        try:
            ocr = self._get_paddle_ocr()
            image_path = self._save_temp_image(image)
            try:
                raw_result = ocr.predict(str(image_path))
            finally:
                image_path.unlink(missing_ok=True)
            texts, scores = self._extract_paddle_text_scores(raw_result)
        except Exception as exc:
            error = str(exc)
            metadata = {"error": error}
            if "ConvertPirAttribute2RuntimeAttribute" in error or "onednn_instruction" in error:
                metadata["hint"] = (
                    "PaddleOCR failed inside Paddle oneDNN/PIR runtime. "
                    "Restart the backend after disabling FLAGS_use_mkldnn, "
                    "FLAGS_use_onednn, FLAGS_enable_pir_api, and FLAGS_enable_pir_in_executor."
                )
            return OCRResult("", 0.0, "paddleocr_error", metadata)

        text = normalize_text("\n".join(texts))
        confidence = sum(scores) / len(scores) if scores else (0.6 if text else 0.0)
        return OCRResult(
            text=text,
            confidence=max(0.0, min(1.0, confidence)),
            engine="paddleocr_ppocrv6_medium",
            metadata={
                "word_count": len(texts),
                "lang": self.config.paddleocr_lang,
                "det_model": self.config.paddleocr_det_model,
                "rec_model": self.config.paddleocr_rec_model,
            },
        )

    def _get_paddle_ocr(self):
        if self._paddle_ocr is not None:
            return self._paddle_ocr

        self._apply_paddle_runtime_flags()
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            ocr_version="PP-OCRv6",
            lang=self.config.paddleocr_lang,
            text_detection_model_name=self.config.paddleocr_det_model,
            text_recognition_model_name=self.config.paddleocr_rec_model,
            use_doc_orientation_classify=self.config.paddleocr_use_doc_orientation,
            use_doc_unwarping=self.config.paddleocr_use_doc_unwarping,
            use_textline_orientation=self.config.paddleocr_use_textline_orientation,
            device=self.config.paddleocr_device,
        )

        self._paddle_ocr = ocr
        return self._paddle_ocr

    def _apply_paddle_runtime_flags(self) -> None:
        if os.getenv("PADDLEOCR_DISABLE_ONEDNN", "1").lower() in {"0", "false", "no"}:
            return
        flag_values = {
            "FLAGS_use_mkldnn": "0",
            "FLAGS_use_onednn": "0",
            "FLAGS_enable_pir_api": "0",
            "FLAGS_enable_pir_in_executor": "0",
        }
        os.environ.update(flag_values)
        try:
            import paddle

            paddle.set_flags({key: False for key in flag_values})
        except Exception:
            # Some Paddle builds do not expose all flags to Python. Env vars are
            # still left in place for the C++ runtime before predictor creation.
            return

    def _save_temp_image(self, image: Image.Image) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            path = Path(handle.name)
        image.convert("RGB").save(path)
        return path

    def _extract_paddle_text_scores(self, raw_result: Any) -> tuple[list[str], list[float]]:
        texts: list[str] = []
        scores: list[float] = []
        for item in self._walk_paddle_result(raw_result):
            if isinstance(item, dict):
                rec_texts = item.get("rec_texts")
                if rec_texts is None:
                    rec_texts = item.get("texts")
                rec_scores = item.get("rec_scores")
                if rec_scores is None:
                    rec_scores = item.get("scores")
                rec_text_list = self._as_list(rec_texts)
                rec_score_list = self._as_list(rec_scores)
                if rec_text_list is not None:
                    texts.extend(str(text).strip() for text in rec_text_list if str(text).strip())
                    if rec_score_list is not None:
                        scores.extend(self._coerce_score(score) for score in rec_score_list)
                elif isinstance(item.get("text"), str):
                    text = item["text"].strip()
                    if text:
                        texts.append(text)
                        if "score" in item:
                            scores.append(self._coerce_score(item["score"]))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                if isinstance(item[0], str):
                    text = item[0].strip()
                    if text:
                        texts.append(text)
                        scores.append(self._coerce_score(item[1]))
                elif isinstance(item[1], (list, tuple)) and len(item[1]) == 2 and isinstance(item[1][0], str):
                    text = item[1][0].strip()
                    if text:
                        texts.append(text)
                        scores.append(self._coerce_score(item[1][1]))
        return texts, scores

    def _walk_paddle_result(self, value: Any) -> Iterable[Any]:
        yield value
        if hasattr(value, "res"):
            yield from self._walk_paddle_result(value.res)
            return
        if hasattr(value, "json"):
            try:
                json_value = value.json() if callable(value.json) else value.json
                yield from self._walk_paddle_result(json_value)
            except (TypeError, ValueError):
                pass
            return
        if isinstance(value, dict):
            for child in value.values():
                yield from self._walk_paddle_result(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from self._walk_paddle_result(child)

    def _as_list(self, value: Any) -> list[Any] | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return list(value)
        if hasattr(value, "tolist"):
            converted = value.tolist()
            return converted if isinstance(converted, list) else [converted]
        return None

    def _coerce_score(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if score > 1:
            score /= 100
        return max(0.0, min(1.0, score))
