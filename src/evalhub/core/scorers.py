"""Deterministic, offline scorers.

Each scorer maps a ``(prediction, reference)`` pair to a float in ``[0, 1]``.
They are registered in ``SCORERS`` so the API and CLI can look them up by name.
All scoring is pure and side-effect free, which keeps it fast and testable.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

# A scorer takes prediction and (optional) reference and returns a [0, 1] score.
Scorer = Callable[[str, Optional[str]], float]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ARTICLES = {"a", "an", "the"}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, drop articles, collapse whitespace.

    This is the standard SQuAD-style normalization; it makes string comparison
    robust to trivial formatting differences.
    """
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _ARTICLES]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _ARTICLES]


def _require_reference(name: str, reference: Optional[str]) -> str:
    if reference is None:
        raise ValueError(f"scorer '{name}' requires a reference answer")
    return reference


def exact_match(prediction: str, reference: Optional[str]) -> float:
    """1.0 if the normalized strings are identical, else 0.0."""
    ref = _require_reference("exact_match", reference)
    return 1.0 if normalize(prediction) == normalize(ref) else 0.0


def token_f1(prediction: str, reference: Optional[str]) -> float:
    """Token-overlap F1 between prediction and reference (SQuAD-style)."""
    ref = _require_reference("token_f1", reference)
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(ref)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = _multiset_intersection_size(pred_tokens, ref_tokens)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def contains(prediction: str, reference: Optional[str]) -> float:
    """1.0 if the normalized reference appears within the normalized prediction."""
    ref = _require_reference("contains", reference)
    return 1.0 if normalize(ref) in normalize(prediction) else 0.0


def length_ratio(prediction: str, reference: Optional[str]) -> float:
    """Ratio of prediction length to reference length, clamped to [0, 1].

    A soft signal for verbosity/brevity; requires a reference.
    """
    ref = _require_reference("length_ratio", reference)
    pred_len = len(tokenize(prediction))
    ref_len = len(tokenize(ref))
    if ref_len == 0:
        return 1.0 if pred_len == 0 else 0.0
    return min(pred_len / ref_len, 1.0)


def non_empty(prediction: str, reference: Optional[str]) -> float:
    """1.0 if the prediction contains any content. Needs no reference."""
    return 1.0 if prediction.strip() else 0.0


def _multiset_intersection_size(a: list[str], b: list[str]) -> int:
    """Size of the multiset intersection of two token lists."""
    from collections import Counter

    overlap = Counter(a) & Counter(b)
    return sum(overlap.values())


#: Registry mapping scorer name -> callable.
SCORERS: dict[str, Scorer] = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "contains": contains,
    "length_ratio": length_ratio,
    "non_empty": non_empty,
}

#: Scorers that do not need a reference answer.
REFERENCE_FREE: frozenset[str] = frozenset({"non_empty"})


def get_scorer(name: str) -> Scorer:
    try:
        return SCORERS[name]
    except KeyError:
        raise KeyError(
            f"unknown scorer '{name}'; available: {sorted(SCORERS)}"
        ) from None
