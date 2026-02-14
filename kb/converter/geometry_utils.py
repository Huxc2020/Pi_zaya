from __future__ import annotations

import fitz
from typing import Optional, List, Tuple

def _bbox_width(bbox: tuple[float, float, float, float]) -> float:
    return float(bbox[2]) - float(bbox[0])

def _bbox_height(bbox: tuple[float, float, float, float]) -> float:
    return float(bbox[3]) - float(bbox[1])

def _rect_area(rect: fitz.Rect) -> float:
    return float(rect.width) * float(rect.height)

def _rect_intersection_area(a: fitz.Rect, b: fitz.Rect) -> float:
    # Use PyMuPDF's built-in intersection logic which is robust
    # or manual logic if fitz is optional
    if not a.intersects(b):
        return 0.0
    # a.intersect(b) modifies 'a' in place in some versions, or returns new rect?
    # PyMuPDF Rect.intersect() modifies ITSELF and returns empty on failure?
    # Safest is manual
    x0 = max(float(a.x0), float(b.x0))
    y0 = max(float(a.y0), float(b.y0))
    x1 = min(float(a.x1), float(b.x1))
    y1 = min(float(a.y1), float(b.y1))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)

def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def _union_rect(rects: list["fitz.Rect"]) -> Optional["fitz.Rect"]:
    if not rects:
        return None
    x0 = min(float(r.x0) for r in rects)
    y0 = min(float(r.y0) for r in rects)
    x1 = max(float(r.x1) for r in rects)
    y1 = max(float(r.y1) for r in rects)
    return fitz.Rect(x0, y0, x1, y1)
