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
    ocr_engine: str = field(default_factory=lambda: os.getenv("OCR_ENGINE", "rapidocr"))
    ocr_runtime: str = field(default_factory=lambda: os.getenv("OCR_RUNTIME", "torch"))
    ocr_device: str = field(default_factory=lambda: os.getenv("OCR_DEVICE", "cuda:0"))
    ocr_lang: str = field(default_factory=lambda: os.getenv("OCR_LANG", "en"))
    ocr_det_lang: str = field(default_factory=lambda: os.getenv("OCR_DET_LANG", "ch"))
    ocr_version: str = field(default_factory=lambda: os.getenv("OCR_VERSION", "PP-OCRv6"))
    ocr_model_size: str = field(default_factory=lambda: os.getenv("OCR_MODEL_SIZE", "medium"))
    ocr_min_confidence: float = field(default_factory=lambda: float(os.getenv("OCR_MIN_CONFIDENCE", "0.72")))
    layout_enabled: bool = field(default_factory=lambda: os.getenv("LAYOUT_ENABLE", "true").lower() == "true")
    layout_engine: str = field(default_factory=lambda: os.getenv("LAYOUT_ENGINE", "doclayout_yolo"))
    layout_device: str = field(default_factory=lambda: os.getenv("LAYOUT_DEVICE", "cuda:0"))
    layout_model_repo: str = field(
        default_factory=lambda: os.getenv("LAYOUT_MODEL_REPO", "juliozhao/DocLayout-YOLO-DocStructBench")
    )
    layout_model_filename: str = field(
        default_factory=lambda: os.getenv("LAYOUT_MODEL_FILENAME", "doclayout_yolo_docstructbench_imgsz1024.pt")
    )
    layout_model_path: str = field(default_factory=lambda: os.getenv("LAYOUT_MODEL_PATH", ""))
    layout_confidence: float = field(default_factory=lambda: float(os.getenv("LAYOUT_CONFIDENCE", "0.25")))
    layout_image_size: int = field(default_factory=lambda: int(os.getenv("LAYOUT_IMAGE_SIZE", "1024")))
    warmup_layout: bool = field(default_factory=lambda: os.getenv("WARMUP_LAYOUT", "true").lower() == "true")
    warmup_ocr: bool = field(default_factory=lambda: os.getenv("WARMUP_OCR", "true").lower() == "true")
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
            self.ocr_min_confidence,
            self.layout_confidence,
            self.ocr_duplicate_similarity_threshold,
            self.ocr_duplicate_token_overlap_threshold,
            self.ocr_min_new_token_ratio,
        )
        if any(value < 0 or value > 1 for value in thresholds):
            raise ValueError("Document quality and OCR thresholds must be between 0 and 1.")
        if self.ocr_engine.lower() != "rapidocr":
            raise ValueError("OCR_ENGINE must be rapidocr.")
        if self.ocr_runtime.lower() != "torch":
            raise ValueError("OCR_RUNTIME must be torch.")
        ocr_device = self.ocr_device.lower()
        if ocr_device != "cuda" and not (
            ocr_device.startswith("cuda:") and ocr_device.removeprefix("cuda:").isdigit()
        ):
            raise ValueError("OCR_DEVICE must be cuda or cuda:<device_id>.")
        if self.ocr_version not in {"PP-OCRv4", "PP-OCRv5", "PP-OCRv6"}:
            raise ValueError("OCR_VERSION must be PP-OCRv4, PP-OCRv5, or PP-OCRv6 for RapidOCR.")
        if self.ocr_model_size.lower() not in {"mobile", "server", "tiny", "small", "medium"}:
            raise ValueError("OCR_MODEL_SIZE must be mobile, server, tiny, small, or medium for RapidOCR.")
        if self.layout_enabled and self.layout_engine.lower() != "doclayout_yolo":
            raise ValueError("LAYOUT_ENGINE must be doclayout_yolo.")
        if self.layout_image_size <= 0:
            raise ValueError("LAYOUT_IMAGE_SIZE must be positive.")


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".text"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_DOCX_EXTENSIONS = {".docx"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
