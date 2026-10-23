"""Hand-checkable tests for CER/WER. Run: python3 -m pytest tests/test_metrics.py -v"""

import pytest

from src.metrics import cer, edit_distance, wer


def test_identical_text_has_zero_error():
    assert cer("hello world", "hello world") == 0.0
    assert wer("hello world", "hello world") == 0.0


def test_one_substitution():
    # "cat" -> "cut": 1 edit over 3 reference characters.
    assert cer("cat", "cut") == pytest.approx(1 / 3)


def test_one_deletion_and_one_insertion():
    assert edit_distance("abc", "ac") == 1  # deletion
    assert edit_distance("ac", "abc") == 1  # insertion


def test_wer_counts_words_not_characters():
    # 1 wrong word out of 4.
    assert wer("the quick brown fox", "the quick brown dog") == pytest.approx(0.25)


def test_empty_reference():
    assert cer("", "") == 0.0
    assert cer("", "abc") == 1.0
    assert wer("", "") == 0.0


def test_error_rate_can_exceed_one():
    # Many insertions against a short reference.
    assert cer("a", "abcdef") == pytest.approx(5.0)
