"""Pydantic schemas shared by the API, the evaluation engine, and the store.

These are the contract of the platform: a *dataset* is a collection of
*samples* (a model prediction plus, usually, a reference answer), and an
*evaluation* scores a dataset with one or more *scorers*.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class Sample(BaseModel):
    """A single evaluation record."""

    id: str = Field(default_factory=_uuid)
    input: Optional[str] = Field(
        default=None, description="The prompt/question given to the model."
    )
    prediction: str = Field(description="The model's output being evaluated.")
    reference: Optional[str] = Field(
        default=None, description="The gold/expected answer, if available."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetCreate(BaseModel):
    """Request body for creating a dataset."""

    name: str = Field(min_length=1, max_length=200)
    samples: list[Sample] = Field(min_length=1)


class Dataset(BaseModel):
    """A stored dataset."""

    id: str = Field(default_factory=_uuid)
    name: str
    samples: list[Sample]
    created_at: datetime = Field(default_factory=_now)

    @property
    def size(self) -> int:
        return len(self.samples)


# --------------------------------------------------------------------------- #
# Evaluations
# --------------------------------------------------------------------------- #
class JudgeBackend(str, Enum):
    heuristic = "heuristic"
    anthropic = "anthropic"


class JudgeConfig(BaseModel):
    """Configuration for the optional LLM-as-judge scorer."""

    backend: JudgeBackend = JudgeBackend.heuristic
    criteria: str = Field(
        default="Rate how well the prediction answers the input and matches the "
        "reference in correctness and completeness.",
        description="Natural-language rubric handed to the judge.",
    )


class EvaluationCreate(BaseModel):
    """Request body for running an evaluation."""

    dataset_id: str
    scorers: list[str] = Field(
        default_factory=lambda: ["exact_match", "token_f1"],
        min_length=1,
        description="Names of registered scorers to apply.",
    )
    judge: Optional[JudgeConfig] = Field(
        default=None,
        description="When set, an 'llm_judge' score is added per sample.",
    )

    @field_validator("scorers")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        # Preserve order, drop duplicates.
        return list(dict.fromkeys(v))


class EvaluationStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class SampleScore(BaseModel):
    """Per-sample scores, one entry per applied scorer."""

    sample_id: str
    scores: dict[str, float]
    judge_reasoning: Optional[str] = None


class MetricSummary(BaseModel):
    """Aggregate statistics for one scorer across the dataset."""

    count: int
    mean: float
    minimum: float
    maximum: float


class EvaluationResult(BaseModel):
    """The full result of an evaluation run."""

    id: str = Field(default_factory=_uuid)
    dataset_id: str
    status: EvaluationStatus = EvaluationStatus.pending
    scorers: list[str] = Field(default_factory=list)
    summary: dict[str, MetricSummary] = Field(default_factory=dict)
    sample_scores: list[SampleScore] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)
    duration_ms: Optional[float] = None
