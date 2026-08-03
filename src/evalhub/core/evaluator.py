"""The evaluation engine.

Runs a set of scorers (and an optional LLM judge) over every sample in a
dataset, concurrently and with a bounded worker pool, then aggregates the
per-sample scores into summary metrics.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from ..config import Settings
from .judge import Judge, build_judge
from .models import (
    Dataset,
    EvaluationCreate,
    EvaluationResult,
    EvaluationStatus,
    MetricSummary,
    Sample,
    SampleScore,
)
from .scorers import REFERENCE_FREE, get_scorer

logger = logging.getLogger(__name__)


class Evaluator:
    """Scores a dataset. Stateless apart from injected settings."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def validate_request(self, request: EvaluationCreate) -> None:
        """Fail fast on unknown scorer names before any work begins."""
        for name in request.scorers:
            get_scorer(name)  # raises KeyError for unknown names

    async def run(
        self, dataset: Dataset, request: EvaluationCreate
    ) -> EvaluationResult:
        """Execute the evaluation and return a completed result."""
        self.validate_request(request)

        judge: Optional[Judge] = None
        scorer_names = list(request.scorers)
        if request.judge is not None:
            judge = build_judge(request.judge, self._settings)
            scorer_names.append("llm_judge")

        result = EvaluationResult(
            dataset_id=dataset.id,
            status=EvaluationStatus.running,
            scorers=scorer_names,
        )
        start = time.perf_counter()

        semaphore = asyncio.Semaphore(self._settings.max_concurrency)

        async def score_one(sample: Sample) -> SampleScore:
            async with semaphore:
                return await self._score_sample(sample, request, judge)

        try:
            sample_scores = await asyncio.gather(
                *(score_one(s) for s in dataset.samples)
            )
        except Exception as exc:  # noqa: BLE001 - surface any engine failure
            logger.exception("evaluation failed", extra={"dataset_id": dataset.id})
            result.status = EvaluationStatus.failed
            result.error = str(exc)
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result

        result.sample_scores = list(sample_scores)
        result.summary = _aggregate(sample_scores, scorer_names)
        result.status = EvaluationStatus.completed
        result.duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "evaluation completed",
            extra={
                "dataset_id": dataset.id,
                "samples": len(sample_scores),
                "duration_ms": round(result.duration_ms, 2),
            },
        )
        return result

    async def _score_sample(
        self,
        sample: Sample,
        request: EvaluationCreate,
        judge: Optional[Judge],
    ) -> SampleScore:
        scores: dict[str, float] = {}
        reasoning: Optional[str] = None

        for name in request.scorers:
            scorer = get_scorer(name)
            if name not in REFERENCE_FREE and sample.reference is None:
                # A missing reference is a data issue, not an engine crash:
                # record it as a zero rather than aborting the whole run.
                scores[name] = 0.0
                continue
            scores[name] = float(scorer(sample.prediction, sample.reference))

        if judge is not None:
            verdict = await judge.score(sample)
            scores["llm_judge"] = verdict.score
            reasoning = verdict.reasoning

        return SampleScore(
            sample_id=sample.id, scores=scores, judge_reasoning=reasoning
        )


def _aggregate(
    sample_scores: list[SampleScore], scorer_names: list[str]
) -> dict[str, MetricSummary]:
    """Reduce per-sample scores into per-scorer summary statistics."""
    summary: dict[str, MetricSummary] = {}
    for name in scorer_names:
        values = [
            ss.scores[name] for ss in sample_scores if name in ss.scores
        ]
        if not values:
            continue
        summary[name] = MetricSummary(
            count=len(values),
            mean=sum(values) / len(values),
            minimum=min(values),
            maximum=max(values),
        )
    return summary
