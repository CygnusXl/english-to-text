"""Bridge between the app and the trained CRNN-CTC line recognizer.

The recognizer itself (model.py, vocabulary.py, decode.py) is built in
Milestones 3-5. Until a trained checkpoint exists, load_recognizer()
returns None and the app shows the segmentation results only. It never
returns fake text.

Contract for later milestones: src/decode.py must provide

    load_for_inference(checkpoint_path: str, device: str) -> Callable[[list[np.ndarray]], list[str]]

which takes grayscale line crops (uint8, white background) and returns one
decoded string per crop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

CHECKPOINT_PATH = Path("checkpoints/best.pt")

Recognizer = Callable[[list[np.ndarray]], list[str]]


def load_recognizer(checkpoint_path: Path = CHECKPOINT_PATH) -> Recognizer | None:
    """Return a recognizer, or None when no trained model is available yet."""
    if not Path(checkpoint_path).exists():
        return None
    try:
        from src.decode import load_for_inference  # written in Milestone 5
    except ImportError:
        return None

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return None
    return load_for_inference(str(checkpoint_path), device)
