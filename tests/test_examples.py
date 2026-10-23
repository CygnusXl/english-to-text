"""Checks that every example in examples/manifest.csv is usable by the app.

Run: python3 -m pytest tests/test_examples.py -v

For the OpenCV-font demo pages the number of detected lines must equal the
number of ground-truth lines. When real test-set examples are added later,
this same test will tell you which pages the line detector gets wrong.
"""

from pathlib import Path

import cv2
import pandas as pd
import pytest

from src.line_detection import detect_lines, prepare_page

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
MANIFEST = EXAMPLES / "manifest.csv"

rows = pd.read_csv(MANIFEST).fillna("").to_dict("records") if MANIFEST.exists() else []


def test_manifest_has_required_columns():
    assert MANIFEST.exists(), "run scripts/make_demo_pages.py first"
    columns = set(pd.read_csv(MANIFEST).columns)
    assert {"file", "title", "ground_truth", "source"} <= columns


@pytest.mark.parametrize("row", rows, ids=[r["file"] for r in rows])
def test_example_image_exists_and_lines_match_ground_truth(row):
    path = EXAMPLES / row["file"]
    assert path.exists()
    image = cv2.imread(str(path))
    assert image is not None
    expected = len(str(row["ground_truth"]).split("\\n"))
    found = len(detect_lines(prepare_page(image).ink))
    assert found == expected
