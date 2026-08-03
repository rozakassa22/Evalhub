"""Unit tests for the deterministic scorers."""
from __future__ import annotations

import pytest

from evalhub.core import scorers


def test_normalize_strips_articles_and_punctuation():
    assert scorers.normalize("The  Paris, France!") == "paris france"


def test_exact_match_is_normalization_insensitive():
    assert scorers.exact_match("Paris.", "the paris") == 1.0
    assert scorers.exact_match("Paris", "London") == 0.0


def test_token_f1_partial_overlap():
    score = scorers.token_f1("the quick brown fox", "the lazy brown dog")
    # 2 shared tokens (brown; "the" is an article and dropped) of 3 each.
    assert 0.0 < score < 1.0
    assert round(score, 4) == round(2 * (1 / 3) * (1 / 3) / (2 / 3), 4)


def test_token_f1_identity():
    assert scorers.token_f1("Tokyo", "tokyo") == 1.0


def test_token_f1_disjoint():
    assert scorers.token_f1("apple", "orange") == 0.0


def test_contains():
    assert scorers.contains("The capital is Paris", "paris") == 1.0
    assert scorers.contains("The capital is Berlin", "paris") == 0.0


def test_length_ratio_clamped():
    assert scorers.length_ratio("one two three four", "one two") == 1.0
    assert scorers.length_ratio("one", "one two three four") == pytest.approx(0.25)


def test_non_empty_needs_no_reference():
    assert scorers.non_empty("something", None) == 1.0
    assert scorers.non_empty("   ", None) == 0.0


def test_missing_reference_raises_for_reference_scorers():
    with pytest.raises(ValueError):
        scorers.token_f1("x", None)


def test_get_scorer_unknown():
    with pytest.raises(KeyError):
        scorers.get_scorer("does_not_exist")
