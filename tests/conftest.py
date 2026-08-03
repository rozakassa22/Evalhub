"""Shared pytest fixtures."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from evalhub.config import Settings
from evalhub.main import create_app
from evalhub.store import repository as repo_module


@pytest.fixture(autouse=True)
def _reset_repository():
    """Isolate each test with a fresh, re-seeded repository."""
    repo_module.reset_repository()
    yield
    repo_module.reset_repository()


@pytest.fixture
def settings() -> Settings:
    return Settings(environment="test", log_level="warning", max_concurrency=4)


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as c:
        yield c
