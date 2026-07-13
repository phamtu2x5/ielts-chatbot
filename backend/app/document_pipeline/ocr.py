import os
import tempfile
import threading
from dataclasses import dataclass
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
        self._paddle_small = None
        self._paddle_medium = None
        self._lock = threading.RLock()

    def image_to_text(self, image: Image.Image) -> OCRResult:
        with self._lock:
            return self._image_to_text(image)

    def _image_to_text(self, image: Image.Image) -> OCRResult:
        result = self._image_to_text_with_paddle(image, use_medium=False)
        if result.text and result.confidence >= self.config.paddleocr_min_confidence:
            return result

        fallback_result = self._image_to_text_with_paddle(image, use_medium=True)
        attempts = [self._attempt_summary(result), self._attempt_summary(fallback_result)]
        if fallback_result.text and fallback_result.confidence >= result.confidence:
            fallback_result.metadata["cascade_attempts"] = attempts
            return fallback_result

        if result.text:
            result.metadata["cascade_attempts"] = attempts
            return result

        return OCRResult(
            "",
            0.0,
            "paddleocr_failed",
            {
                "reason": "paddleocr_small_and_medium_failed",
                "cascade_attempts": attempts,
            },
        )

    def _attempt_summary(self, result: OCRResult) -> Dict[str, Any]:
        return {
            "engine": result.engine,
            "confidence": result.confidence,
            "has_text": bool(result.text),
            "error": result.metadata.get("error"),
        }

    def warmup(self, include_medium: bool = True) -> Dict[str, Any]:
        with self._lock:
            image = Image.new("RGB", (420, 96), "white")
            from PIL import ImageDraw

            draw = ImageDraw.Draw(image)
            draw.text((16, 32), "IELTS OCR warmup", fill="black")

            results = {"engine": self.config.ocr_engine, "small": None, "medium": None}
            small = self._image_to_text_with_paddle(image, use_medium=False)
            results["small"] = {
                "engine": small.engine,
                "confidence": small.confidence,
                "ok": bool(small.text),
                "metadata": small.metadata,
            }
            if include_medium:
                medium = self._image_to_text_with_paddle(image, use_medium=True)
                results["medium"] = {
                    "engine": medium.engine,
                    "confidence": medium.confidence,
                    "ok": bool(medium.text),
                    "metadata": medium.metadata,
                }
            results["models_ready"] = bool(results["small"]["ok"]) and (
                not include_medium or bool(results["medium"] and results["medium"]["ok"])
            )
            return results

    def _paddle_is_available(self) -> bool:
        try:
            __import__("paddleocr")
        except ImportError:
            return False
        return True

    def _image_to_text_with_paddle(self, image: Image.Image, use_medium: bool) -> OCRResult:
        if not self._paddle_is_available():
            return OCRResult("", 0.0, "paddleocr_unavailable", {"reason": "paddleocr_not_installed"})

        try:
            ocr = self._get_paddle_ocr(use_medium)
            image_path = self._save_temp_image(image)
            try:
                raw_result = ocr.predict(str(image_path))
            finally:
                image_path.unlink(missing_ok=True)
            texts, scores = self._extract_paddle_text_scores(raw_result)
        except Exception as exc:
            error = str(exc)
            metadata = {"error": error, "tier": "medium" if use_medium else "small"}
            if "ConvertPirAttribute2RuntimeAttribute" in error or "onednn_instruction" in error:
                metadata["hint"] = (
                    "PaddleOCR failed inside Paddle oneDNN/PIR runtime. "
                    "Restart the backend after disabling FLAGS_use_mkldnn, "
                    "FLAGS_use_onednn, FLAGS_enable_pir_api, and FLAGS_enable_pir_in_executor."
                )
            return OCRResult("", 0.0, "paddleocr_error", metadata)

        text = normalize_text("\n".join(texts))
        confidence = sum(scores) / len(scores) if scores else (0.6 if text else 0.0)
        tier = "medium" if use_medium else "small"
        return OCRResult(
            text=text,
            confidence=max(0.0, min(1.0, confidence)),
            engine=f"paddleocr_ppocrv6_{tier}",
            metadata={
                "tier": tier,
                "word_count": len(texts),
                "lang": self.config.paddleocr_lang,
                "det_model": self.config.paddleocr_fallback_det_model
                if use_medium
                else self.config.paddleocr_default_det_model,
                "rec_model": self.config.paddleocr_fallback_rec_model
                if use_medium
                else self.config.paddleocr_default_rec_model,
            },
        )

    def _get_paddle_ocr(self, use_medium: bool):
        if use_medium and self._paddle_medium is not None:
            return self._paddle_medium
        if not use_medium and self._paddle_small is not None:
            return self._paddle_small

        from paddleocr import PaddleOCR

        det_model = self.config.paddleocr_fallback_det_model if use_medium else self.config.paddleocr_default_det_model
        rec_model = self.config.paddleocr_fallback_rec_model if use_medium else self.config.paddleocr_default_rec_model

        ocr = PaddleOCR(
            ocr_version="PP-OCRv6",
            lang=self.config.paddleocr_lang,
            text_detection_model_name=det_model,
            text_recognition_model_name=rec_model,
            use_doc_orientation_classify=self.config.paddleocr_use_doc_orientation,
            use_doc_unwarping=self.config.paddleocr_use_doc_unwarping,
            use_textline_orientation=self.config.paddleocr_use_textline_orientation,
            device=self.config.paddleocr_device,
        )

        if use_medium:
            self._paddle_medium = ocr
        else:
            self._paddle_small = ocr
        return ocr

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
