import re
from difflib import SequenceMatcher

from .config import DocumentPipelineConfig
from .models import DocumentElement, ProcessedDocument, ProcessedPage


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def _terms(text: str) -> set[str]:
    return set(TOKEN_RE.findall((text or "").lower()))


def _compact(text: str) -> str:
    return " ".join((text or "").lower().split())[:20_000]


class NativeOCRReconciler:
    def __init__(self, config: DocumentPipelineConfig) -> None:
        self.config = config

    def reconcile(self, document: ProcessedDocument) -> ProcessedDocument:
        pages = [self._reconcile_page(page) for page in document.pages]
        document.pages = pages
        report = document.metadata.setdefault("extraction_report", {})
        report["pages"] = [self._page_report(page) for page in pages]
        return document

    def _reconcile_page(self, page: ProcessedPage) -> ProcessedPage:
        native_elements = [element for element in page.elements if not self._is_ocr_element(element)]
        native_text = "\n".join(element.normalized_text for element in native_elements)
        native_terms = _terms(native_text)
        kept: list[DocumentElement] = []
        alternatives: list[dict] = []
        duplicates_removed = 0
        ocr_elements = 0

        for element in page.elements:
            if not self._is_ocr_element(element):
                element.metadata.setdefault("canonical", True)
                kept.append(element)
                continue

            ocr_elements += 1
            similarity = SequenceMatcher(None, _compact(native_text), _compact(element.normalized_text)).ratio()
            ocr_terms = _terms(element.normalized_text)
            token_overlap = len(ocr_terms & native_terms) / max(len(ocr_terms), 1)
            new_token_ratio = len(ocr_terms - native_terms) / max(len(ocr_terms), 1)
            duplicate = (
                similarity >= self.config.ocr_duplicate_similarity_threshold
                or token_overlap >= self.config.ocr_duplicate_token_overlap_threshold
            ) and new_token_ratio < self.config.ocr_min_new_token_ratio

            alternative = {
                "source": element.source,
                "text": element.normalized_text,
                "confidence": element.confidence,
                "similarity_to_native": round(similarity, 4),
                "token_overlap_with_native": round(token_overlap, 4),
                "new_token_ratio": round(new_token_ratio, 4),
            }

            if duplicate and native_elements:
                alternatives.append(alternative)
                duplicates_removed += 1
                continue

            if native_elements:
                element.type = "ocr_supplement"
            element.metadata.update(
                {
                    "canonical": True,
                    "native_similarity": round(similarity, 4),
                    "native_token_overlap": round(token_overlap, 4),
                    "new_token_ratio": round(new_token_ratio, 4),
                }
            )
            kept.append(element)

        page.elements = kept
        page.processing_route = self._processing_route(
            page.processing_route,
            duplicates_removed,
            ocr_elements,
            bool(native_elements),
        )
        if alternatives:
            page.metadata["alternative_sources"] = alternatives
        page.metadata.update(
            {
                "native_elements": len(native_elements),
                "ocr_elements": ocr_elements,
                "duplicates_removed": duplicates_removed,
            }
        )
        return page

    def _is_ocr_element(self, element: DocumentElement) -> bool:
        return element.type == "ocr_overlay" or "ocr" in element.source.lower()

    def _processing_route(
        self,
        route: str,
        duplicates_removed: int,
        ocr_elements: int,
        has_native: bool,
    ) -> str:
        if not has_native:
            return route
        if duplicates_removed:
            return f"{route}_deduped"
        if ocr_elements:
            return f"{route}_reconciled"
        return route

    def _page_report(self, page: ProcessedPage) -> dict:
        page_metadata = page.metadata
        return {
            "page": page.page_number,
            "processing_route": page.processing_route,
            "quality_score": page.quality_score,
            "native_quality": page_metadata.get("native_quality"),
            "ocr_quality": page_metadata.get("ocr_quality"),
            "ocr_engine": page_metadata.get("ocr_engine"),
            "native_elements": page_metadata.get("native_elements", 0),
            "ocr_elements": page_metadata.get("ocr_elements", 0),
            "duplicates_removed": page_metadata.get("duplicates_removed", 0),
            "alternative_sources": len(page_metadata.get("alternative_sources", [])),
            "element_types": sorted({element.type for element in page.elements}),
        }
