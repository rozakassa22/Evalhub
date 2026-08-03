"""FastAPI application factory and ASGI entry point.

Run locally with::

    uvicorn evalhub.main:app --reload

or import ``create_app`` to build an isolated instance (used by tests).
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .api.routes import datasets, evaluations, health
from .config import Settings, get_settings
from .logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    logger.info(
        "starting %s",
        settings.service_name,
        extra={"environment": settings.environment, "version": __version__},
    )
    yield
    logger.info("shutting down %s", settings.service_name)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI application."""
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    app = FastAPI(
        title="evalhub",
        version=__version__,
        summary="A scalable LLM evaluation platform.",
        lifespan=_lifespan,
    )
    app.state.settings = settings

    @app.middleware("http")
    async def _timing_and_logging(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(elapsed_ms, 2),
            },
        )
        return response

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Never leak internals to clients; log the detail server-side.
        logger.exception("unhandled error", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500, content={"detail": "internal server error"}
        )

    app.include_router(health.router)
    app.include_router(datasets.router)
    app.include_router(evaluations.router)

    @app.get("/", tags=["meta"])
    async def root() -> dict[str, str]:
        return {
            "service": settings.service_name,
            "version": __version__,
            "docs": "/docs",
        }

    return app


#: Module-level ASGI app for ``uvicorn evalhub.main:app``.
app = create_app()
