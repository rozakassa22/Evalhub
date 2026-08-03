"""Evaluation endpoints — run and retrieve evaluations."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ...core.models import EvaluationCreate, EvaluationResult
from ...core.scorers import SCORERS
from ..deps import EvaluatorDep, RepositoryDep

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.get("/scorers", response_model=list[str])
async def list_scorers() -> list[str]:
    """List the names of every registered scorer."""
    return sorted(SCORERS)


@router.post("", response_model=EvaluationResult, status_code=status.HTTP_201_CREATED)
async def run_evaluation(
    body: EvaluationCreate,
    repo: RepositoryDep,
    evaluator: EvaluatorDep,
) -> EvaluationResult:
    """Run an evaluation over a stored dataset and persist the result."""
    dataset = await repo.get_dataset(body.dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"dataset '{body.dataset_id}' not found",
        )

    try:
        evaluator.validate_request(body)
    except KeyError as exc:
        raise HTTPException(
            status_code=422,  # unprocessable content
            detail=str(exc).strip("'\""),
        ) from exc

    result = await evaluator.run(dataset, body)
    await repo.add_evaluation(result)
    return result


@router.get("", response_model=list[EvaluationResult])
async def list_evaluations(repo: RepositoryDep) -> list[EvaluationResult]:
    return await repo.list_evaluations()


@router.get("/{evaluation_id}", response_model=EvaluationResult)
async def get_evaluation(
    evaluation_id: str, repo: RepositoryDep
) -> EvaluationResult:
    result = await repo.get_evaluation(evaluation_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"evaluation '{evaluation_id}' not found",
        )
    return result
