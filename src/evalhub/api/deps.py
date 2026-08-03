"""FastAPI dependency providers.

Centralizing these here keeps the route handlers declarative and makes it
trivial to override collaborators (settings, repository, evaluator) in tests
via ``app.dependency_overrides``.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from ..config import Settings, get_settings
from ..core.evaluator import Evaluator
from ..store import Repository, get_repository


def settings_dep() -> Settings:
    return get_settings()


def repository_dep() -> Repository:
    return get_repository()


def evaluator_dep(
    settings: Annotated[Settings, Depends(settings_dep)],
) -> Evaluator:
    return Evaluator(settings)


SettingsDep = Annotated[Settings, Depends(settings_dep)]
RepositoryDep = Annotated[Repository, Depends(repository_dep)]
EvaluatorDep = Annotated[Evaluator, Depends(evaluator_dep)]
