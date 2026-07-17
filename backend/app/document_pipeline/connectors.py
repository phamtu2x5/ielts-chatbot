from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

import numpy as np
from PIL import Image

from .config import DocumentPipelineConfig


@dataclass(frozen=True)
class ConnectorDetectionResult:
    regions: list[dict[str, Any]]
    metadata: dict[str, Any]


class RasterConnectorDetector:
    """Extract lightweight connector geometry from detected figure regions.

    The detector deliberately stops before semantic graph construction. It
    records geometry and an arrowhead candidate; the visual parser later maps
    that evidence to OCR nodes. Ambiguous components remain unresolved.
    """

    def __init__(self, config: DocumentPipelineConfig) -> None:
        self.config = config

    def detect(
        self,
        image: Image.Image,
        layout_regions: list[dict[str, Any]],
        ocr_lines: list[dict[str, Any]],
    ) -> ConnectorDetectionResult:
        if not self.config.connector_enabled:
            return ConnectorDetectionResult([], {"skipped": True, "reason": "disabled"})

        try:
            import cv2
        except ImportError as exc:
            return ConnectorDetectionResult([], {"skipped": True, "reason": str(exc)})

        figures = [
            (region, self._bbox(region.get("bbox")))
            for region in layout_regions
            if str(region.get("type") or "").strip().lower().replace(" ", "_") == "figure"
        ]
        figures = [(region, bbox) for region, bbox in figures if bbox is not None]
        if not figures:
            return ConnectorDetectionResult([], {"skipped": True, "reason": "no_figure_regions"})

        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        _, foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        line_heights = [
            bbox[3] - bbox[1]
            for bbox in (self._bbox(line.get("bbox")) for line in ocr_lines)
            if bbox is not None
        ]
        median_line_height = median(line_heights) if line_heights else 24.0
        self._mask_text(foreground, ocr_lines, median_line_height, cv2)

        detected_regions = []
        for region, bbox in figures:
            connectors = self._components(
                foreground,
                bbox,
                median_line_height,
                cv2,
            )
            detected_regions.append(
                {
                    "bbox": bbox,
                    "layout_confidence": round(float(region.get("confidence") or 0.0), 4),
                    "source": "raster_connector_geometry",
                    "connectors": connectors,
                }
            )
        return ConnectorDetectionResult(
            detected_regions,
            {
                "skipped": False,
                "figure_regions": len(detected_regions),
                "connectors_found": sum(len(region["connectors"]) for region in detected_regions),
            },
        )

    def _components(
        self,
        foreground: np.ndarray,
        bbox: list[float],
        median_line_height: float,
        cv2: Any,
    ) -> list[dict[str, Any]]:
        x0, y0, x1, y1 = self._clip_bbox(bbox, foreground.shape[1], foreground.shape[0])
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        region_area = float(width * height)
        roi = foreground[y0:y1, x0:x1].copy()

        horizontal_size = max(15, round(width * 0.05))
        vertical_size = max(15, round(height * 0.14))
        horizontal = cv2.morphologyEx(
            roi,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1)),
        )
        vertical = cv2.morphologyEx(
            roi,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size)),
        )
        border_mask = cv2.dilate(
            cv2.bitwise_or(horizontal, vertical),
            np.ones((7, 7), dtype=np.uint8),
            iterations=1,
        )
        connector_mask = cv2.subtract(roi, border_mask)
        connector_mask = cv2.morphologyEx(
            connector_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )

        count, labels, stats, _ = cv2.connectedComponentsWithStats(connector_mask)
        connectors = []
        for label in range(1, count):
            left, top, component_width, component_height, area = (
                int(value) for value in stats[label]
            )
            area_ratio = area / region_area
            span_ratio = max(component_width / width, component_height / height)
            if not (
                self.config.connector_min_component_area_ratio <= area_ratio
                <= self.config.connector_max_component_area_ratio
            ):
                continue
            if span_ratio < self.config.connector_min_span_ratio or min(component_width, component_height) < 5:
                continue

            component = labels == label
            geometry = self._component_geometry(component, median_line_height, cv2)
            if geometry is None:
                continue
            connectors.append(
                {
                    "id": f"connector-{len(connectors) + 1}",
                    "bbox": [
                        float(x0 + left),
                        float(y0 + top),
                        float(x0 + left + component_width),
                        float(y0 + top + component_height),
                    ],
                    "endpoints": [
                        [float(round(float(point[0] + x0), 2)), float(round(float(point[1] + y0), 2))]
                        for point in geometry["endpoints"]
                    ],
                    "arrowhead_point": [
                        float(round(float(geometry["arrowhead_point"][0] + x0), 2)),
                        float(round(float(geometry["arrowhead_point"][1] + y0), 2)),
                    ],
                    "direction_confidence": geometry["direction_confidence"],
                    "direction_evidence": geometry["direction_evidence"],
                    "area_ratio": round(area_ratio, 6),
                }
            )
        return connectors[:64]

    def _component_geometry(
        self,
        component: np.ndarray,
        median_line_height: float,
        cv2: Any,
    ) -> dict[str, Any] | None:
        ys, xs = np.where(component)
        if len(xs) < 8:
            return None
        points = np.column_stack((xs, ys)).astype(np.float64)
        center = points.mean(axis=0)
        covariance = np.cov((points - center).T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        projection = (points - center) @ axis
        low, high = np.quantile(projection, [0.03, 0.97])
        endpoints = [center + axis * low, center + axis * high]

        distance = cv2.distanceTransform(component.astype(np.uint8) * 255, cv2.DIST_L2, 5)
        head_y, head_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        head_point = np.array([float(head_x), float(head_y)])
        head_strength = float(distance[head_y, head_x]) / max(1.0, median_line_height * 0.3)
        endpoint_distances = [float(np.linalg.norm(head_point - point)) for point in endpoints]
        separation = abs(endpoint_distances[0] - endpoint_distances[1]) / max(
            1.0,
            endpoint_distances[0] + endpoint_distances[1],
        )
        normalized_head_strength = min(1.0, head_strength)
        confidence = 0.5 * normalized_head_strength + 0.5 * min(1.0, separation * 2.0)
        return {
            "endpoints": endpoints,
            "arrowhead_point": head_point,
            "direction_confidence": round(confidence, 4),
            "direction_evidence": {
                "head_strength": round(normalized_head_strength, 4),
                "endpoint_separation": round(separation, 4),
            },
        }

    def _mask_text(
        self,
        foreground: np.ndarray,
        ocr_lines: list[dict[str, Any]],
        median_line_height: float,
        cv2: Any,
    ) -> None:
        padding = max(2, round(median_line_height * 0.15))
        for line in ocr_lines:
            bbox = self._bbox(line.get("bbox"))
            if bbox is None:
                continue
            x0, y0, x1, y1 = self._clip_bbox(bbox, foreground.shape[1], foreground.shape[0])
            cv2.rectangle(
                foreground,
                (max(0, x0 - padding), max(0, y0 - padding)),
                (min(foreground.shape[1] - 1, x1 + padding), min(foreground.shape[0] - 1, y1 + padding)),
                0,
                -1,
            )

    def _bbox(self, value: Any) -> list[float] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        if len(value) >= 4 and all(isinstance(item, (int, float)) for item in value[:4]):
            x0, y0, x1, y1 = (float(item) for item in value[:4])
            return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        points = [point for point in value if isinstance(point, (list, tuple)) and len(point) >= 2]
        if not points:
            return None
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return [min(xs), min(ys), max(xs), max(ys)]

    def _clip_bbox(self, bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
        x0 = max(0, min(width - 1, int(round(bbox[0]))))
        y0 = max(0, min(height - 1, int(round(bbox[1]))))
        x1 = max(x0 + 1, min(width, int(round(bbox[2]))))
        y1 = max(y0 + 1, min(height, int(round(bbox[3]))))
        return x0, y0, x1, y1
