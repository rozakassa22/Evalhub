"""Performance-focused batch data-processing utilities."""

from .fast_metrics import CorpusReport, corpus_report, iter_jsonl_samples

__all__ = ["CorpusReport", "corpus_report", "iter_jsonl_samples"]
