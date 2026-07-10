import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentPipelineConfig:
    parser_version: str = "1.0.0"
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("DOCUMENT_MAX_UPLOAD_MB", "25")))
    max_pdf_pages: int = field(default_factory=lambda: int(os.getenv("DOCUMENT_MAX_PDF_PAGES", "80")))
    native_min_chars: int = field(default_factory=lambda: int(os.getenv("DOCUMENT_NATIVE_MIN_CHARS", "40")))
    native_min_readable_ratio: float = field(
        default_factory=lambda: float(os.getenv("DOCUMENT_NATIVE_MIN_READABLE_RATIO", "0.82"))
    )
    native_max_repeated_line_ratio: float = field(
        default_factory=lambda: float(os.getenv("DOCUMENT_NATIVE_MAX_REPEATED_LINE_RATIO", "0.35"))
    )
    scanned_image_coverage: float = field(
        default_factory=lambda: float(os.getenv("DOCUMENT_SCANNED_IMAGE_COVERAGE", "0.65"))
    )
    ocr_lang: str = field(default_factory=lambda: os.getenv("DOCUMENT_OCR_LANG", os.getenv("PDF_OCR_LANG", "vie+eng")))
    ocr_dpi: int = field(default_factory=lambda: int(os.getenv("DOCUMENT_OCR_DPI", os.getenv("PDF_OCR_DPI", "180"))))
    ocr_engine: str = field(default_factory=lambda: os.getenv("OCR_ENGINE", "paddle"))
    ocr_fallback_engine: str = field(default_factory=lambda: os.getenv("OCR_FALLBACK_ENGINE", "tesseract"))
    paddleocr_device: str = field(default_factory=lambda: os.getenv("PADDLEOCR_DEVICE", "cpu"))
    paddleocr_engine: str = field(default_factory=lambda: os.getenv("PADDLEOCR_ENGINE", "paddle"))
    paddleocr_lang: str = field(default_factory=lambda: os.getenv("PADDLEOCR_LANG", "latin"))
    paddleocr_default_det_model: str = field(
        default_factory=lambda: os.getenv("PADDLEOCR_DEFAULT_DET_MODEL", "PP-OCRv6_small_det")
    )
    paddleocr_default_rec_model: str = field(
        default_factory=lambda: os.getenv("PADDLEOCR_DEFAULT_REC_MODEL", "PP-OCRv6_small_rec")
    )
    paddleocr_fallback_det_model: str = field(
        default_factory=lambda: os.getenv("PADDLEOCR_FALLBACK_DET_MODEL", "PP-OCRv6_medium_det")
    )
    paddleocr_fallback_rec_model: str = field(
        default_factory=lambda: os.getenv("PADDLEOCR_FALLBACK_REC_MODEL", "PP-OCRv6_medium_rec")
    )
    paddleocr_min_confidence: float = field(default_factory=lambda: float(os.getenv("PADDLEOCR_MIN_CONFIDENCE", "0.72")))
    paddleocr_use_doc_orientation: bool = field(
        default_factory=lambda: os.getenv("PADDLEOCR_USE_DOC_ORIENTATION", "false").lower() == "true"
    )
    paddleocr_use_doc_unwarping: bool = field(
        default_factory=lambda: os.getenv("PADDLEOCR_USE_DOC_UNWARPING", "false").lower() == "true"
    )
    paddleocr_use_textline_orientation: bool = field(
        default_factory=lambda: os.getenv("PADDLEOCR_USE_TEXTLINE_ORIENTATION", "false").lower() == "true"
    )
    warmup_ocr: bool = field(default_factory=lambda: os.getenv("WARMUP_OCR", "true").lower() == "true")
    warmup_ocr_medium: bool = field(default_factory=lambda: os.getenv("WARMUP_OCR_MEDIUM", "true").lower() == "true")
    chunk_target_tokens: int = field(default_factory=lambda: int(os.getenv("DOCUMENT_CHUNK_TARGET_TOKENS", "600")))
    chunk_max_tokens: int = field(default_factory=lambda: int(os.getenv("DOCUMENT_CHUNK_MAX_TOKENS", "800")))
    chunk_overlap_tokens: int = field(default_factory=lambda: int(os.getenv("DOCUMENT_CHUNK_OVERLAP_TOKENS", "80")))


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".text"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_DOCX_EXTENSIONS = {".docx"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
