"""Lightweight OpenCV chart structure detection for ChartOCR-style pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def detect_chart_structure(image_path: Path) -> Dict[str, Any]:
    try:
        import cv2
    except ImportError:
        return {
            "detected_lines": [],
            "detected_components": [],
            "possible_axis_labels": [],
            "error": "missing_opencv",
        }

    img = cv2.imread(str(image_path))
    if img is None:
        return {
            "detected_lines": [],
            "detected_components": [],
            "possible_axis_labels": [],
            "error": "image_read_failed",
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    h, w = gray.shape[:2]

    lines_out: List[Dict[str, Any]] = []
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min(w, h) // 8, maxLineGap=10)
    if lines is not None:
        for line in lines[:40]:
            x1, y1, x2, y2 = [int(v) for v in line[0]]
            orientation = "horizontal" if abs(y2 - y1) < abs(x2 - x1) else "vertical"
            lines_out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "orientation": orientation})

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    components: List[Dict[str, Any]] = []
    for i in range(1, min(num_labels, 80)):
        x, y, bw, bh, area = stats[i]
        if area < 80:
            continue
        aspect = bw / max(bh, 1)
        comp_type = "bar_like" if aspect < 0.8 and bh > h * 0.05 else "component"
        components.append(
            {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "area": int(area), "type": comp_type}
        )

    axis_labels = []
    if lines_out:
        horiz = [l for l in lines_out if l["orientation"] == "horizontal"]
        vert = [l for l in lines_out if l["orientation"] == "vertical"]
        if horiz:
            axis_labels.append("x_axis_candidate")
        if vert:
            axis_labels.append("y_axis_candidate")

    return {
        "detected_lines": lines_out,
        "detected_components": components,
        "possible_axis_labels": axis_labels,
        "error": "",
    }
