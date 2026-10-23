"""Generate the demo pages in examples/ and their manifest.csv.

Run from the repository root:  python3 scripts/make_demo_pages.py

IMPORTANT: these pages are drawn by a script with a computer font. They are NOT real
handwriting and NOT from the OpenHand-Synth test set. They exist so the app has
something to show (and the page-processing code something to be tested on)
before real test samples are exported. Real examples go in the same folder
with the same manifest columns.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "examples"
SOURCE = "Demo page drawn by a script with a computer font (not real handwriting, not from the training data)"

PAGES = [
    {
        "file": "demo_clean.png",
        "title": "Demo 1: clean, straight page",
        "lines": [
            "meeting notes monday",
            "buy milk and eggs",
            "finish the report draft",
            "call the dentist at noon",
            "read chapter seven",
        ],
        "angle": 0.0, "shadow": 1.0, "noise": 2, "ink": 30, "blur": 0,
    },
    {
        "file": "demo_tilted_shadow.png",
        "title": "Demo 2: tilted page with a shadow",
        "lines": [
            "project ideas for friday",
            "draw the model diagram",
            "write the dataset section",
            "test the line detector",
            "email the supervisor",
        ],
        "angle": 4.0, "shadow": 0.7, "noise": 4, "ink": 30, "blur": 0,
    },
    {
        "file": "demo_low_contrast.png",
        "title": "Demo 3: faint ink, noisy, slightly blurred",
        "lines": [
            "shopping list for the week",
            "rice and green tea",
            "new notebook and pens",
            "charge the camera battery",
            "return the library books",
        ],
        "angle": -2.0, "shadow": 0.8, "noise": 7, "ink": 120, "blur": 3,
    },
]


def render(page: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((1000, 800, 3), 255, np.uint8)
    ink = page["ink"]
    for i, text in enumerate(page["lines"]):
        cv2.putText(img, text, (70, 140 + i * 150), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (ink,) * 3, 3, cv2.LINE_AA)

    gradient = np.linspace(page["shadow"], 1.0, img.shape[1], dtype=np.float32)[None, :, None]
    img = (img.astype(np.float32) * gradient).clip(0, 255)
    img = (img + rng.normal(0, page["noise"], img.shape)).clip(0, 255).astype(np.uint8)
    if page["blur"]:
        img = cv2.GaussianBlur(img, (page["blur"] | 1, page["blur"] | 1), 0)
    if page["angle"]:
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), page["angle"], 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderValue=(235, 235, 235))
    return img


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = []
    for seed, page in enumerate(PAGES):
        cv2.imwrite(str(OUT / page["file"]), render(page, seed))
        rows.append(
            {
                "file": page["file"],
                "title": page["title"],
                "ground_truth": "\\n".join(page["lines"]),  # literal \n, decoded by the app
                "source": SOURCE,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "manifest.csv", index=False)
    print(f"Wrote {len(rows)} pages and manifest.csv to {OUT}")


if __name__ == "__main__":
    main()
