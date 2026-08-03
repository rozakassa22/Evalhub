"""Tests for the LLM judge and the evaluation engine."""
from __future__ import annotations

import pytest

from evalhub.config import Settings
from evalhub.core.evaluator import Evaluator
from evalhub.core.judge import HeuristicJudge, JudgeResult, build_judge
from evalhub.core.models import (
    Dataset,
    EvaluationCreate,
    EvaluationStatus,
    JudgeBackend,
    JudgeConfig,
    Sample,
)


@pytest.fixture
def settings() -> Settings:
    return Settings(environment="test", max_concurrency=4, anthropic_api_key=None)


@pytest.fixture
def dataset() -> Dataset:
    return Dataset(
        id="d1",
        name="test",
        samples=[
            Sample(prediction="Paris", reference="Paris"),
            Sample(prediction="Sydney", reference="Canberra"),
            Sample(prediction="Tokyo", reference="Tokyo"),
        ],
    )


async def test_heuristic_judge_scores_in_range():
    judge = HeuristicJudge(JudgeConfig())
    result = await judge.score(Sample(prediction="Paris", reference="Paris"))
    assert isinstance(result, JudgeResult)
    assert 0.0 <= result.score <= 1.0
    assert result.score == pytest.approx(1.0)


async def test_heuristic_judge_without_reference():
    judge = HeuristicJudge(JudgeConfig())
    result = await judge.score(Sample(prediction="anything"))
    assert result.score == 1.0
    result_empty = await judge.score(Sample(prediction="   "))
    assert result_empty.score == 0.0


def test_build_judge_falls_back_without_api_key(settings):
    config = JudgeConfig(backend=JudgeBackend.anthropic)
    judge = build_judge(config, settings)
    assert isinstance(judge, HeuristicJudge)


async def test_evaluator_runs_and_aggregates(settings, dataset):
    evaluator = Evaluator(settings)
    request = EvaluationCreate(
        dataset_id="d1", scorers=["exact_match", "token_f1"]
    )
    result = await evaluator.run(dataset, request)

    assert result.status is EvaluationStatus.completed
    assert len(result.sample_scores) == 3
    # Two of three predictions exactly match their reference.
    assert result.summary["exact_match"].mean == pytest.approx(2 / 3)
    assert result.summary["exact_match"].count == 3
    assert result.duration_ms is not None


async def test_evaluator_with_judge_adds_llm_score(settings, dataset):
    evaluator = Evaluator(settings)
    request = EvaluationCreate(
        dataset_id="d1",
        scorers=["exact_match"],
        judge=JudgeConfig(backend=JudgeBackend.heuristic),
    )
    result = await evaluator.run(dataset, request)
    assert "llm_judge" in result.summary
    assert all("llm_judge" in ss.scores for ss in result.sample_scores)
    assert all(ss.judge_reasoning for ss in result.sample_scores)


async def test_evaluator_handles_missing_reference(settings):
    ds = Dataset(
        id="d2",
        name="noref",
        samples=[Sample(prediction="hello")],  # no reference
    )
    evaluator = Evaluator(settings)
    result = await evaluator.run(
        ds, EvaluationCreate(dataset_id="d2", scorers=["token_f1", "non_empty"])
    )
    assert result.status is EvaluationStatus.completed
    # Reference-requiring scorer records 0.0 rather than crashing.
    assert result.sample_scores[0].scores["token_f1"] == 0.0
    assert result.sample_scores[0].scores["non_empty"] == 1.0


def test_evaluator_validates_unknown_scorer(settings):
    evaluator = Evaluator(settings)
    with pytest.raises(KeyError):
        evaluator.validate_request(
            EvaluationCreate(dataset_id="d1", scorers=["bogus"])
        )
