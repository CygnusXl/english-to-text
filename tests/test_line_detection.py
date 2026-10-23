"""Tests for page preprocessing and line detection on a synthetic page.

Run from the repository root:  python3 -m pytest tests/test_line_detection.py -v

The synthetic page is made with computer-drawn text, so it checks the geometry code
(skew, projection profile, ordering, manual fallback). It does NOT measure
real handwriting performance.
"""

import cv2
import numpy as np

from src.line_detection import (
    crop_line,
    detect_lines,
    manual_line_boxes,
    prepare_page,
)

N_LINES = 5
LINE_TEXTS = [
    "meeting notes monday",
    "buy milk and eggs",
    "finish the report draft",
    "call the dentist at noon",
    "read chapter seven",
]


def make_page(angle: float = 0.0, seed: int = 0) -> np.ndarray:
    """White page, five lines of text, uneven lighting, noise, optional rotation."""
    rng = np.random.default_rng(seed)
    page = np.full((1000, 800, 3), 255, np.uint8)
    for i, text in enumerate(LINE_TEXTS):
        y = 140 + i * 150
        cv2.putText(page, text, (80, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (30, 30, 30), 3, cv2.LINE_AA)

    # Uneven lighting: a smooth horizontal brightness gradient.
    gradient = np.linspace(0.75, 1.0, page.shape[1], dtype=np.float32)[None, :, None]
    page = (page.astype(np.float32) * gradient).clip(0, 255).astype(np.uint8)
    noise = rng.normal(0, 4, page.shape).astype(np.float32)
    page = (page.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)

    if angle:
        h, w = page.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        page = cv2.warpAffine(page, matrix, (w, h), borderValue=(235, 235, 235))
    return page


def test_detects_all_lines_on_straight_page():
    prepared = prepare_page(make_page(angle=0.0))
    boxes = detect_lines(prepared.ink)
    assert len(boxes) == N_LINES


def test_lines_are_in_top_to_bottom_order():
    prepared = prepare_page(make_page(angle=0.0))
    boxes = detect_lines(prepared.ink)
    tops = [b.y0 for b in boxes]
    assert tops == sorted(tops)


def test_deskew_recovers_small_rotation():
    prepared = prepare_page(make_page(angle=3.0))
    # The page was rotated by +3 degrees, so straightening should apply about -3.
    assert abs(prepared.skew_angle + 3.0) <= 1.0
    assert len(detect_lines(prepared.ink)) == N_LINES


def test_crop_is_nonempty_grayscale():
    prepared = prepare_page(make_page())
    box = detect_lines(prepared.ink)[0]
    crop = crop_line(prepared.gray, box)
    assert crop.ndim == 2 and crop.size > 0


def test_blank_page_has_no_lines():
    blank = np.full((600, 500, 3), 255, np.uint8)
    prepared = prepare_page(blank)
    assert detect_lines(prepared.ink) == []


def test_manual_fallback_uses_given_ranges():
    prepared = prepare_page(make_page())
    auto = detect_lines(prepared.ink)
    ranges = [(b.y0, b.y1) for b in auto[:2]]
    manual = manual_line_boxes(prepared.ink, ranges)
    assert len(manual) == 2
    assert manual_line_boxes(prepared.ink, [(10, 10)]) == []
