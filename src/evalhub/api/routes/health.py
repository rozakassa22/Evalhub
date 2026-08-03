"""Liveness and readiness probes."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ... import __version__
from ..deps import SettingsDep

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep) -> HealthResponse:
    """Liveness probe — the process is up."""
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=__version__,
        environment=settings.environment,
    )


@router.get("/health/ready", response_model=HealthResponse)
async def ready(settings: SettingsDep) -> HealthResponse:
    """Readiness probe — dependencies are wired and the app can serve traffic."""
    return HealthResponse(
        status="ready",
        service=settings.service_name,
        version=__version__,
        environment=settings.environment,
    )
