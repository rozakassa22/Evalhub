"""Streaming, single-pass corpus statistics over evaluation samples.

This module is the platform's performance-focused data-processing tool. It is
built to run over datasets far larger than memory:

* Input is consumed as a *generator* — the whole file is never materialized.
* Statistics are accumulated in a *single pass* using running aggregates
  (Welford's algorithm for variance), so memory use is O(vocabulary), not
  O(samples).
* The hot path avoids per-token object churn and reuses compiled regexes.

It computes token-length distributions, vocabulary size, and — when references
are present — the mean token-F1 of predictions against references.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from ..core.models import Sample
from ..core.scorers import token_f1, tokenize


@dataclass
class _RunningStats:
    """Welford's online mean/variance — numerically stable, O(1) memory."""

    count: int = 0
    mean: float = 0.0
    _m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def variance(self) -> float:
        return self._m2 / self.count if self.count > 1 else 0.0

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)

    def as_dict(self) -> dict[str, float]:
        if self.count == 0:
            return {"count": 0, "mean": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": self.count,
            "mean": round(self.mean, 4),
            "stddev": round(self.stddev, 4),
            "min": round(self.minimum, 4),
            "max": round(self.maximum, 4),
        }


@dataclass
class CorpusReport:
    """Aggregate statistics for a corpus of samples."""

    samples: int
    prediction_tokens: dict[str, float]
    reference_tokens: dict[str, float]
    vocabulary_size: int
    scored_pairs: int
    mean_token_f1: Optional[float]

    def as_dict(self) -> dict:
        return {
            "samples": self.samples,
            "prediction_tokens": self.prediction_tokens,
            "reference_tokens": self.reference_tokens,
            "vocabulary_size": self.vocabulary_size,
            "scored_pairs": self.scored_pairs,
            "mean_token_f1": (
                round(self.mean_token_f1, 4)
                if self.mean_token_f1 is not None
                else None
            ),
        }


def corpus_report(samples: Iterable[Sample]) -> CorpusReport:
    """Compute corpus statistics in a single streaming pass.

    Accepts any iterable — a list, or a lazy generator over a huge file. Memory
    footprint is bounded by the vocabulary set, not the number of samples.
    """
    pred_len = _RunningStats()
    ref_len = _RunningStats()
    f1 = _RunningStats()
    vocabulary: set[str] = set()

    for sample in samples:
        pred_tokens = tokenize(sample.prediction)
        pred_len.update(len(pred_tokens))
        vocabulary.update(pred_tokens)

        if sample.reference is not None:
            ref_tokens = tokenize(sample.reference)
            ref_len.update(len(ref_tokens))
            vocabulary.update(ref_tokens)
            f1.update(token_f1(sample.prediction, sample.reference))

    return CorpusReport(
        samples=pred_len.count,
        prediction_tokens=pred_len.as_dict(),
        reference_tokens=ref_len.as_dict(),
        vocabulary_size=len(vocabulary),
        scored_pairs=f1.count,
        mean_token_f1=f1.mean if f1.count else None,
    )


def iter_jsonl_samples(path: str | Path) -> Iterator[Sample]:
    """Yield ``Sample`` objects from a JSONL file, one line at a time.

    The file is streamed, so a multi-gigabyte dataset uses constant memory.
    Blank lines are skipped; malformed lines raise with their line number.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                yield Sample.model_validate(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{p}:{lineno}: invalid sample: {exc}") from exc


@dataclass
class Benchmark:
    """Throughput measurement for a processing run."""

    samples: int
    seconds: float
    report: CorpusReport = field(repr=False)

    @property
    def samples_per_second(self) -> float:
        return self.samples / self.seconds if self.seconds > 0 else math.inf


def benchmark(samples: Iterable[Sample]) -> Benchmark:
    """Materialize ``samples`` once and time a full corpus-report pass."""
    import time

    materialized = list(samples)
    start = time.perf_counter()
    report = corpus_report(materialized)
    elapsed = time.perf_counter() - start
    return Benchmark(samples=len(materialized), seconds=elapsed, report=report)
