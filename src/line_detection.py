"""Page preprocessing and text-line detection (classical OpenCV, no neural network).

Pipeline for one page photo or scan:

    BGR image -> grayscale
              -> flatten uneven lighting -> ink mask (Otsu threshold)
              -> estimate skew -> rotate to straighten
              -> horizontal projection profile -> text-line boxes
              -> crop each line

Segmentation is kept separate from recognition so each part can be tested
on its own. Projection-profile detection works for conventional,
left-to-right, top-to-bottom notes with reasonably separated lines. It can
fail on overlapping lines, ruled paper, heavy shadows or curved pages; use
manual_line_boxes() as the fallback in those cases.

Every intermediate result is kept on PreparedPage so the app can show the
process step by step.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LineBox:
    """Axis-aligned box of one text line, in pixel coordinates of the straightened page."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


@dataclass
class PreparedPage:
    gray: np.ndarray  # straightened grayscale page, 0-255, white background
    ink: np.ndarray  # straightened ink mask, 255 where there is ink
    skew_angle: float  # degrees that were applied to straighten the page
    # Intermediate stages, kept so the app can show each step of the process.
    gray_raw: np.ndarray | None = None  # grayscale before straightening
    background: np.ndarray | None = None  # estimated paper brightness (ink erased)
    flat: np.ndarray | None = None  # grayscale divided by background (lighting flattened)
    ink_raw: np.ndarray | None = None  # ink mask before straightening
    skew_angles: np.ndarray | None = None  # angles that were tried
    skew_scores: np.ndarray | None = None  # sharpness score for each tried angle


def resize_to_max_side(img: np.ndarray, max_side: int = 2000) -> np.ndarray:
    """Shrink very large photos so processing is fast and parameters stay consistent."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert a BGR (or already grayscale) image to uint8 grayscale."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def flatten_lighting(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove shadows and uneven lighting.

    A morphological close with a kernel larger than a pen stroke erases the
    ink and leaves an estimate of the paper brightness (the background).
    Dividing the page by it makes the paper uniformly white.
    Returns (background, flattened_gray).
    """
    h, w = gray.shape
    k = max(15, (min(h, w) // 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    flat = cv2.divide(gray, background, scale=255)
    return background, flat


def threshold_ink(flat: np.ndarray) -> np.ndarray:
    """Otsu threshold on a flattened page: 255 on ink, 0 on paper."""
    blurred = cv2.GaussianBlur(flat, (3, 3), 0)
    _, ink = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.medianBlur(ink, 3)  # removes isolated specks


def make_ink_mask(gray: np.ndarray) -> np.ndarray:
    """Return a mask that is 255 on ink and 0 on paper (flatten lighting, then Otsu)."""
    _, flat = flatten_lighting(gray)
    return threshold_ink(flat)


def _rotate(img: np.ndarray, angle: float, border_value: int, interpolation: int) -> np.ndarray:
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=interpolation, borderValue=border_value)


def skew_scores(ink: np.ndarray, max_angle: float = 10.0, step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Score how "line-like" the page is after each trial rotation.

    When text lines are horizontal, the row-wise ink sums form sharp peaks
    (lines) and deep valleys (gaps), so their variance is largest. We try
    small rotations and return the variance for each one.
    """
    longest = max(ink.shape)
    small = ink
    if longest > 800:
        scale = 800 / longest
        small = cv2.resize(ink, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    angles = np.arange(-max_angle, max_angle + 1e-9, step)
    scores = np.empty(len(angles), dtype=np.float64)
    for i, angle in enumerate(angles):
        rotated = _rotate(small, float(angle), 0, cv2.INTER_NEAREST)
        scores[i] = float(np.var(rotated.sum(axis=1, dtype=np.float64)))
    return angles, scores


def estimate_skew_angle(ink: np.ndarray, max_angle: float = 10.0, step: float = 0.5) -> float:
    """Rotation (degrees, OpenCV sign convention) that straightens the text."""
    angles, scores = skew_scores(ink, max_angle, step)
    return float(angles[int(np.argmax(scores))])


def prepare_page(img_bgr: np.ndarray, max_side: int = 2000, deskew: bool = True) -> PreparedPage:
    """Grayscale, flatten lighting, binarise and straighten a page, keeping every stage."""
    gray_raw = to_gray(resize_to_max_side(img_bgr, max_side))
    background, flat = flatten_lighting(gray_raw)
    ink_raw = threshold_ink(flat)

    angle = 0.0
    angles = scores = None
    gray, ink = gray_raw, ink_raw
    if deskew:
        angles, scores = skew_scores(ink_raw)
        angle = float(angles[int(np.argmax(scores))])
        if abs(angle) > 1e-6:
            gray = _rotate(gray_raw, angle, 255, cv2.INTER_CUBIC)
            ink = make_ink_mask(gray)

    return PreparedPage(
        gray=gray,
        ink=ink,
        skew_angle=angle,
        gray_raw=gray_raw,
        background=background,
        flat=flat,
        ink_raw=ink_raw,
        skew_angles=angles,
        skew_scores=scores,
    )


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (end exclusive) of every run of True values."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    return runs


def _box_for_rows(ink: np.ndarray, y0: int, y1: int) -> LineBox | None:
    """Tight horizontal extent of the ink between rows y0 and y1."""
    columns = (ink[y0:y1] > 0).sum(axis=0)
    xs = np.where(columns > 0)[0]
    if xs.size == 0:
        return None
    return LineBox(int(xs[0]), int(y0), int(xs[-1]) + 1, int(y1))


def row_profile(ink: np.ndarray, ink_fraction: float = 0.05) -> tuple[np.ndarray, float]:
    """Smoothed ink count per pixel row, and the threshold above which a row counts as text."""
    h = ink.shape[0]
    profile = (ink > 0).sum(axis=1).astype(np.float64)
    k = max(3, int(h * 0.004) | 1)
    smooth = np.convolve(profile, np.ones(k) / k, mode="same")
    return smooth, float(ink_fraction * smooth.max())


def detect_lines(
    ink: np.ndarray,
    ink_fraction: float = 0.05,
    min_height_fraction: float = 0.012,
    merge_gap_fraction: float = 0.006,
) -> list[LineBox]:
    """Detect text lines with a horizontal projection profile.

    ink_fraction:         a row counts as "text" when its smoothed ink count
                          exceeds this fraction of the busiest row.
    min_height_fraction:  discard runs shorter than this fraction of page height.
    merge_gap_fraction:   join runs separated by a gap smaller than this
                          fraction of page height (keeps i-dots and
                          descenders with their line).

    Returned boxes are sorted top to bottom, which is the reading order for
    conventional notes.
    """
    h, _ = ink.shape
    smooth, threshold = row_profile(ink, ink_fraction)
    if smooth.max() <= 0:
        return []
    active = smooth > threshold

    runs = _runs(active)
    merge_gap = max(1, int(h * merge_gap_fraction))
    merged: list[list[int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    min_height = max(2, int(h * min_height_fraction))
    boxes: list[LineBox] = []
    for start, end in merged:
        if end - start < min_height:
            continue
        box = _box_for_rows(ink, start, end)
        if box is not None:
            boxes.append(box)
    return sorted(boxes, key=lambda b: b.y0)


def manual_line_boxes(ink: np.ndarray, row_ranges: list[tuple[int, int]]) -> list[LineBox]:
    """Fallback: build boxes from y ranges chosen by the user.

    The horizontal extent of each box is found from the ink inside the range.
    Invalid or empty ranges are skipped.
    """
    h = ink.shape[0]
    boxes: list[LineBox] = []
    for y0, y1 in row_ranges:
        y0, y1 = int(max(0, min(y0, y1))), int(min(h, max(y0, y1)))
        if y1 - y0 < 2:
            continue
        box = _box_for_rows(ink, y0, y1)
        if box is not None:
            boxes.append(box)
    return sorted(boxes, key=lambda b: b.y0)


def crop_line(gray: np.ndarray, box: LineBox, pad: int = 6) -> np.ndarray:
    """Crop one line from the straightened grayscale page with a small margin."""
    h, w = gray.shape
    y0, y1 = max(0, box.y0 - pad), min(h, box.y1 + pad)
    x0, x1 = max(0, box.x0 - pad), min(w, box.x1 + pad)
    return gray[y0:y1, x0:x1]


def draw_boxes(gray: np.ndarray, boxes: list[LineBox]) -> np.ndarray:
    """Return an RGB image with numbered line boxes, for display."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for i, box in enumerate(boxes, start=1):
        cv2.rectangle(rgb, (box.x0, box.y0), (box.x1, box.y1), (220, 40, 40), 2)
        cv2.putText(rgb, str(i), (box.x0, max(15, box.y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 40, 40), 2)
    return rgb


def draw_profile_view(
    gray: np.ndarray,
    ink: np.ndarray,
    boxes: list[LineBox],
    ink_fraction: float = 0.05,
    graph_width: int = 260,
) -> np.ndarray:
    """Page with numbered line boxes, side by side with its row-ink profile.

    Both halves share the same vertical scale, so every bar lines up with the
    page rows it was measured from. The red vertical line is the threshold: rows
    whose bar passes it are counted as text.
    """
    page = draw_boxes(gray, boxes)
    h = gray.shape[0]
    smooth, threshold = row_profile(ink, ink_fraction)
    peak = max(float(smooth.max()), 1.0)
    usable = graph_width - 6

    graph = np.full((h, graph_width, 3), 255, np.uint8)
    for box in boxes:  # shade the rows that were accepted as lines
        cv2.rectangle(graph, (0, box.y0), (graph_width, box.y1), (222, 242, 222), -1)
    for y in range(h):
        length = int(smooth[y] / peak * usable)
        if length > 0:
            cv2.line(graph, (0, y), (length, y), (70, 110, 200), 1)
    thr_x = int(threshold / peak * usable)
    cv2.line(graph, (thr_x, 0), (thr_x, h), (220, 40, 40), 2)
    cv2.line(graph, (0, 0), (0, h), (150, 150, 150), 1)
    return cv2.hconcat([page, graph])
