"""Character and word error rates.

CER = (substitutions + deletions + insertions) / number of reference characters
WER = the same computed over words.

Both use the Levenshtein edit distance. Text is compared exactly as given
(case and punctuation count), so normalise it first if you want a looser score.
"""

from __future__ import annotations

from typing import Sequence


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Minimum number of substitutions, deletions and insertions turning ref into hyp."""
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate. Returns 0.0 when both are empty and 1.0 for text against an empty reference."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate (words split on whitespace)."""
    ref_words, hyp_words = reference.split(), hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)
