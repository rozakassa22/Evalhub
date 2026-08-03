"""Runtime configuration, loaded from the environment with sane defaults.

Everything the service needs to run is expressed here so that deployment is a
matter of setting environment variables (or a ``.env`` file) — no code changes.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Values are read from environment variables prefixed with ``EVALHUB_``
    (e.g. ``EVALHUB_LOG_LEVEL=debug``) or from a local ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="EVALHUB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Service metadata -------------------------------------------------
    service_name: str = "evalhub"
    environment: str = Field(
        default="development",
        description="Deployment environment name (development, staging, production).",
    )
    log_level: str = Field(default="info", description="Root log level.")
    log_json: bool = Field(
        default=False,
        description="Emit logs as JSON lines (recommended in production).",
    )

    # --- Evaluation engine ------------------------------------------------
    max_concurrency: int = Field(
        default=16,
        ge=1,
        le=256,
        description="Maximum number of samples scored concurrently per evaluation.",
    )

    # --- LLM-as-judge -----------------------------------------------------
    judge_backend: str = Field(
        default="heuristic",
        description="Default judge backend: 'heuristic' (offline) or 'anthropic'.",
    )
    anthropic_model: str = Field(
        default="claude-opus-4-8",
        description="Model id used by the Anthropic judge backend.",
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="API key for the Anthropic judge. When unset, the "
        "'anthropic' backend is unavailable and the heuristic judge is used.",
    )
    judge_timeout_seconds: float = Field(default=30.0, gt=0)

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    Cached so that repeated dependency-injection calls don't re-parse the
    environment. Tests can clear the cache via ``get_settings.cache_clear()``.
    """
    return Settings()
