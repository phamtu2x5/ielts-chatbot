from collections import Counter
from string import printable
from typing import Iterable

from .config import DocumentPipelineConfig
from .models import PageQuality


def _readable_ratio(text: str) -> float:
    if not text:
        return 0.0
    readable = 0
    for char in text:
        if char.isalnum() or char.isspace() or char in printable or "\u00c0" <= char <= "\u1ef9":
            readable += 1
    return readable / len(text)


def _control_ratio(text: str) -> float:
    if not text:
        return 1.0
    controls = sum(1 for char in text if unicodedata_category(char).startswith("C") and char not in "\n\t")
    return controls / len(text)


def unicodedata_category(char: str) -> str:
    import unicodedata

    return unicodedata.category(char)


def _repeated_line_ratio(lines: Iterable[str]) -> float:
    clean_lines = [line.strip() for line in lines if line.strip()]
    if len(clean_lines) < 4:
        return 0.0
    counts = Counter(clean_lines)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(clean_lines)


def evaluate_native_page_text(
    text: str,
    text_block_count: int,
    image_coverage: float,
    config: DocumentPipelineConfig,
) -> PageQuality:
    reasons = []
    stripped = text.strip()
    readable_ratio = _readable_ratio(stripped)
    control_ratio = _control_ratio(stripped)
    repeated_ratio = _repeated_line_ratio(stripped.splitlines())

    if len(stripped) < config.native_min_chars:
        reasons.append("too_few_chars")
    if readable_ratio < config.native_min_readable_ratio:
        reasons.append("low_readable_ratio")
    if control_ratio > 0.03:
        reasons.append("too_many_control_chars")
    if repeated_ratio > config.native_max_repeated_line_ratio:
        reasons.append("too_many_repeated_lines")
    if image_coverage >= config.scanned_image_coverage and text_block_count <= 1:
        reasons.append("likely_scanned_page")
    if text_block_count == 0:
        reasons.append("no_text_blocks")

    score = 1.0
    score -= min(0.4, max(0, config.native_min_chars - len(stripped)) / max(config.native_min_chars, 1) * 0.4)
    score -= min(0.25, max(0, config.native_min_readable_ratio - readable_ratio))
    score -= min(0.2, control_ratio * 4)
    score -= min(0.2, repeated_ratio)
    score -= 0.25 if "likely_scanned_page" in reasons else 0.0
    score = max(0.0, min(1.0, score))

    usable = not reasons or (score >= 0.72 and "likely_scanned_page" not in reasons and "no_text_blocks" not in reasons)
    requires_layout = text_block_count > 8 or image_coverage > 0.35

    return PageQuality(
        native_text_is_usable=usable,
        score=score,
        reasons=reasons,
        requires_ocr=not usable,
        requires_layout=requires_layout,
        requires_table_analysis=False,
        recommended_dpi=config.ocr_dpi,
    )
