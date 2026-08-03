"""Tests for the streaming batch-processing tool."""
from __future__ import annotations

from evalhub.core.models import Sample
from evalhub.processing.fast_metrics import (
    _RunningStats,
    benchmark,
    corpus_report,
    iter_jsonl_samples,
)


def test_running_stats_matches_reference():
    stats = _RunningStats()
    for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
        stats.update(v)
    assert stats.count == 8
    assert stats.mean == 5.0
    assert round(stats.stddev, 4) == 2.0  # population stddev of the set
    assert stats.minimum == 2.0
    assert stats.maximum == 9.0


def test_corpus_report_basic():
    samples = [
        Sample(prediction="paris france", reference="paris"),
        Sample(prediction="tokyo", reference="tokyo"),
    ]
    report = corpus_report(samples)
    assert report.samples == 2
    assert report.scored_pairs == 2
    assert report.mean_token_f1 is not None
    # "paris", "france", "tokyo" -> 3 unique tokens.
    assert report.vocabulary_size == 3


def test_corpus_report_without_references():
    report = corpus_report([Sample(prediction="hello world")])
    assert report.scored_pairs == 0
    assert report.mean_token_f1 is None


def test_iter_jsonl_samples_streams(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        '{"prediction": "paris", "reference": "paris"}\n'
        "\n"  # blank line is skipped
        '{"prediction": "berlin", "reference": "berlin"}\n',
        encoding="utf-8",
    )
    samples = list(iter_jsonl_samples(path))
    assert len(samples) == 2
    assert samples[0].prediction == "paris"


def test_iter_jsonl_samples_reports_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    try:
        list(iter_jsonl_samples(path))
    except ValueError as exc:
        assert ":1:" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_benchmark_reports_throughput():
    samples = [Sample(prediction=f"pred {i}", reference="pred") for i in range(50)]
    result = benchmark(samples)
    assert result.samples == 50
    assert result.samples_per_second > 0
    assert result.report.samples == 50
