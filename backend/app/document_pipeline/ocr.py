import shutil
import tempfile
from dataclasses import dataclass
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
        self._paddle_small = None
        self._paddle_medium = None

    def is_available(self) -> bool:
        return self._paddle_is_available() or self._tesseract_is_available()

    def image_to_text(self, image: Image.Image) -> OCRResult:
        if self.config.ocr_engine.lower() == "paddle":
            result = self._image_to_text_with_paddle(image, use_medium=False)
            if result.text and result.confidence >= self.config.paddleocr_min_confidence:
                return result

            fallback_result = self._image_to_text_with_paddle(image, use_medium=True)
            if fallback_result.text and fallback_result.confidence >= result.confidence:
                return fallback_result

            if result.text:
                return result

        if self.config.ocr_fallback_engine.lower() == "tesseract" or self.config.ocr_engine.lower() == "tesseract":
            return self._image_to_text_with_tesseract(image)

        return OCRResult("", 0.0, "unavailable", {"reason": "no_ocr_engine_available"})

    def warmup(self, include_medium: bool = True) -> Dict[str, Any]:
        image = Image.new("RGB", (420, 96), "white")
        try:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(image)
            draw.text((16, 32), "IELTS OCR warmup", fill="black")
        except Exception:
            pass

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
            import paddleocr  # noqa: F401
        except ImportError:
            return False
        return True

    def _tesseract_is_available(self) -> bool:
        return bool(shutil.which("tesseract"))

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
            return OCRResult("", 0.0, "paddleocr_error", {"error": str(exc), "tier": "medium" if use_medium else "small"})

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

        try:
            ocr = PaddleOCR(
                ocr_version="PP-OCRv6",
                lang=self.config.paddleocr_lang,
                text_detection_model_name=det_model,
                text_recognition_model_name=rec_model,
                use_doc_orientation_classify=self.config.paddleocr_use_doc_orientation,
                use_doc_unwarping=self.config.paddleocr_use_doc_unwarping,
                use_textline_orientation=self.config.paddleocr_use_textline_orientation,
                device=self.config.paddleocr_device,
                engine=self.config.paddleocr_engine,
            )
        except TypeError:
            # Compatibility path for older PaddleOCR releases. It will not force PP-OCRv6
            # model names, but keeps the OCR layer usable if the runtime is behind.
            ocr = PaddleOCR(
                lang=self.config.paddleocr_lang,
                use_angle_cls=self.config.paddleocr_use_textline_orientation,
                show_log=False,
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
                rec_texts = item.get("rec_texts") or item.get("texts")
                rec_scores = item.get("rec_scores") or item.get("scores")
                if isinstance(rec_texts, list):
                    texts.extend(str(text).strip() for text in rec_texts if str(text).strip())
                    if isinstance(rec_scores, list):
                        scores.extend(self._coerce_score(score) for score in rec_scores)
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
        if hasattr(value, "json"):
            try:
                json_value = value.json() if callable(value.json) else value.json
                yield from self._walk_paddle_result(json_value)
            except TypeError:
                pass
        if hasattr(value, "res"):
            yield from self._walk_paddle_result(value.res)
        if isinstance(value, dict):
            for child in value.values():
                yield from self._walk_paddle_result(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from self._walk_paddle_result(child)

    def _coerce_score(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if score > 1:
            score /= 100
        return max(0.0, min(1.0, score))

    def _image_to_text_with_tesseract(self, image: Image.Image) -> OCRResult:
        if not self._tesseract_is_available():
            return OCRResult("", 0.0, "tesseract_unavailable", {"reason": "tesseract_not_found"})

        try:
            import pytesseract
        except ImportError:
            return OCRResult("", 0.0, "tesseract_unavailable", {"reason": "pytesseract_not_installed"})

        lang = self.config.ocr_lang
        try:
            data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractError:
            if lang == "eng":
                return OCRResult("", 0.0, "tesseract", {"lang": lang, "fallback": False})
            try:
                data = pytesseract.image_to_data(image, lang="eng", output_type=pytesseract.Output.DICT)
                lang = "eng"
            except pytesseract.TesseractError:
                return OCRResult("", 0.0, "tesseract", {"lang": self.config.ocr_lang, "fallback": True})

        words = []
        confidences = []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            word = (text or "").strip()
            if not word:
                continue
            words.append(word)
            try:
                score = float(conf)
            except (TypeError, ValueError):
                score = -1.0
            if score >= 0:
                confidences.append(score / 100)

        text = normalize_text(" ".join(words))
        confidence = sum(confidences) / len(confidences) if confidences else (0.55 if text else 0.0)
        return OCRResult(
            text=text,
            confidence=max(0.0, min(1.0, confidence)),
            engine="tesseract",
            metadata={"lang": lang, "word_count": len(words)},
        )
