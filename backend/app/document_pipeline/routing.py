import mimetypes
from pathlib import Path

from .config import (
    SUPPORTED_DOCX_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
    SUPPORTED_TEXT_EXTENSIONS,
    DocumentPipelineConfig,
)


class FileRouter:
    def __init__(self, config: DocumentPipelineConfig) -> None:
        self.config = config

    def route(self, file_path: Path, filename: str, content_type: str | None) -> str:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_upload_mb:
            raise ValueError(f"Tệp quá lớn. Giới hạn hiện tại là {self.config.max_upload_mb}MB.")

        suffix = Path(filename).suffix.lower()
        guessed_type = content_type or mimetypes.guess_type(filename)[0] or ""

        if suffix in SUPPORTED_TEXT_EXTENSIONS:
            return "text"
        if suffix in SUPPORTED_PDF_EXTENSIONS:
            return "pdf"
        if suffix in SUPPORTED_DOCX_EXTENSIONS:
            return "docx"
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            return "image"

        if not suffix:
            if guessed_type.startswith("text/"):
                return "text"
            if guessed_type == "application/pdf":
                return "pdf"
            if guessed_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return "docx"
            if guessed_type.startswith("image/"):
                return "image"

        raise ValueError("Hiện chỉ hỗ trợ tài liệu dạng TXT/Markdown, PDF, DOCX và ảnh PNG/JPG/JPEG/WebP.")
