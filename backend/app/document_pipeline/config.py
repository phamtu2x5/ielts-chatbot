import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentPipelineConfig:
    parser_version: str = "1.3.0"
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
    ocr_dpi: int = field(default_factory=lambda: int(os.getenv("DOCUMENT_OCR_DPI", os.getenv("PDF_OCR_DPI", "180"))))
    ocr_engine: str = field(default_factory=lambda: os.getenv("OCR_ENGINE", "paddle"))
    paddleocr_device: str = field(default_factory=lambda: os.getenv("PADDLEOCR_DEVICE", "cpu"))
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
    enable_pp_structure: bool = field(
        default_factory=lambda: os.getenv("DOCUMENT_ENABLE_PP_STRUCTURE", "true").lower() == "true"
    )
    pp_structure_device: str = field(default_factory=lambda: os.getenv("PP_STRUCTURE_DEVICE", os.getenv("PADDLEOCR_DEVICE", "cpu")))
    pp_structure_dpi: int = field(default_factory=lambda: int(os.getenv("PP_STRUCTURE_DPI", "180")))
    warmup_pp_structure: bool = field(
        default_factory=lambda: os.getenv("WARMUP_PP_STRUCTURE", "false").lower() == "true"
    )
    chunk_target_tokens: int = field(default_factory=lambda: int(os.getenv("DOCUMENT_CHUNK_TARGET_TOKENS", "600")))
    chunk_max_tokens: int = field(default_factory=lambda: int(os.getenv("DOCUMENT_CHUNK_MAX_TOKENS", "800")))
    chunk_overlap_tokens: int = field(default_factory=lambda: int(os.getenv("DOCUMENT_CHUNK_OVERLAP_TOKENS", "80")))
    ocr_duplicate_similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("DOCUMENT_OCR_DUPLICATE_SIMILARITY", "0.88"))
    )
    ocr_duplicate_token_overlap_threshold: float = field(
        default_factory=lambda: float(os.getenv("DOCUMENT_OCR_DUPLICATE_TOKEN_OVERLAP", "0.92"))
    )
    ocr_min_new_token_ratio: float = field(
        default_factory=lambda: float(os.getenv("DOCUMENT_OCR_MIN_NEW_TOKEN_RATIO", "0.08"))
    )
    enable_ielts_structure_parser: bool = field(
        default_factory=lambda: os.getenv("DOCUMENT_ENABLE_IELTS_STRUCTURE", "true").lower() == "true"
    )

    def __post_init__(self) -> None:
        if self.max_upload_mb <= 0 or self.max_pdf_pages <= 0:
            raise ValueError("Document size and page limits must be positive.")
        if not 0 < self.chunk_target_tokens <= self.chunk_max_tokens:
            raise ValueError("DOCUMENT_CHUNK_TARGET_TOKENS must be within the chunk max limit.")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_target_tokens:
            raise ValueError("DOCUMENT_CHUNK_OVERLAP_TOKENS must be smaller than the target chunk size.")
        thresholds = (
            self.native_min_readable_ratio,
            self.native_max_repeated_line_ratio,
            self.scanned_image_coverage,
            self.paddleocr_min_confidence,
            self.ocr_duplicate_similarity_threshold,
            self.ocr_duplicate_token_overlap_threshold,
            self.ocr_min_new_token_ratio,
        )
        if any(value < 0 or value > 1 for value in thresholds):
            raise ValueError("Document quality and OCR thresholds must be between 0 and 1.")
        if self.ocr_engine.lower() != "paddle":
            raise ValueError("OCR_ENGINE must be paddle.")
        if self.pp_structure_dpi <= 0:
            raise ValueError("PP_STRUCTURE_DPI must be positive.")


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".text"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_DOCX_EXTENSIONS = {".docx"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
