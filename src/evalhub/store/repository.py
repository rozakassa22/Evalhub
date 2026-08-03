"""In-memory repository for datasets and evaluation results.

This is a deliberately small, async-safe store that stands in for a database.
The API depends on the ``Repository`` interface, so swapping in Postgres or
Redis later is a matter of providing another implementation — no route changes.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from ..core.models import Dataset, EvaluationResult, Sample


class Repository:
    """Thread/async-safe in-memory store."""

    def __init__(self) -> None:
        self._datasets: dict[str, Dataset] = {}
        self._evaluations: dict[str, EvaluationResult] = {}
        # Constructed lazily: on Python 3.9 ``asyncio.Lock()`` binds to the
        # running loop at creation time, and the repository may be built from a
        # threadpool worker (a sync FastAPI dependency) with no loop present.
        self._lock_obj: Optional[asyncio.Lock] = None

    @property
    def _lock(self) -> asyncio.Lock:
        if self._lock_obj is None:
            self._lock_obj = asyncio.Lock()
        return self._lock_obj

    # --- Datasets ---------------------------------------------------------
    async def add_dataset(self, dataset: Dataset) -> Dataset:
        async with self._lock:
            self._datasets[dataset.id] = dataset
        return dataset

    async def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        async with self._lock:
            return self._datasets.get(dataset_id)

    async def list_datasets(self) -> list[Dataset]:
        async with self._lock:
            return list(self._datasets.values())

    # --- Evaluations ------------------------------------------------------
    async def add_evaluation(self, result: EvaluationResult) -> EvaluationResult:
        async with self._lock:
            self._evaluations[result.id] = result
        return result

    async def get_evaluation(
        self, evaluation_id: str
    ) -> Optional[EvaluationResult]:
        async with self._lock:
            return self._evaluations.get(evaluation_id)

    async def list_evaluations(self) -> list[EvaluationResult]:
        async with self._lock:
            return list(self._evaluations.values())


_repository: Optional[Repository] = None


def get_repository() -> Repository:
    """Return the process-wide repository singleton, seeded on first use."""
    global _repository
    if _repository is None:
        _repository = Repository()
        _seed(_repository)
    return _repository


def reset_repository() -> None:
    """Drop all state. Used by tests to isolate cases."""
    global _repository
    _repository = None


def _seed(repo: Repository) -> None:
    """Populate a small demo dataset so the API is usable out of the box."""
    demo = Dataset(
        id="demo",
        name="Capitals QA (demo)",
        samples=[
            Sample(
                input="What is the capital of France?",
                prediction="The capital of France is Paris.",
                reference="Paris",
            ),
            Sample(
                input="What is the capital of Japan?",
                prediction="Tokyo",
                reference="Tokyo",
            ),
            Sample(
                input="What is the capital of Australia?",
                prediction="Sydney",
                reference="Canberra",
            ),
        ],
    )
    # Seeding happens before the event loop runs; write directly.
    repo._datasets[demo.id] = demo  # noqa: SLF001 - intentional seed hook
