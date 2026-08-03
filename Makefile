.DEFAULT_GOAL := help
.PHONY: help install dev test lint typecheck fmt run docker-build docker-up docker-down clean

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(VENV): ## Create the virtualenv
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV) ## Install the package
	$(PIP) install .

dev: $(VENV) ## Install with dev + llm extras (editable)
	$(PIP) install -e ".[dev,llm]"

test: ## Run the test suite
	$(PY) -m pytest

lint: ## Lint with ruff
	$(PY) -m ruff check src tests

fmt: ## Auto-format with ruff
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

typecheck: ## Static type-check with mypy
	$(PY) -m mypy

run: ## Run the API locally with reload
	$(PY) -m uvicorn evalhub.main:app --reload

docker-build: ## Build the container image
	docker build -t evalhub:latest .

docker-up: ## Start the service via docker compose
	docker compose up --build

docker-down: ## Stop the service
	docker compose down

clean: ## Remove caches and build artifacts
	rm -rf build dist src/*.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
