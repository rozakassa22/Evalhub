# evalhub — a scalable LLM evaluation platform

`evalhub` is a compact but production-shaped platform for **evaluating the
outputs of AI models**. It brings together three pieces that a real evaluation
system needs:

1. **A scalable FastAPI backend** — async, typed, dependency-injected, with
   health probes, structured logging, and request timing.
2. **An AI evaluation workflow** — a pluggable set of scorers plus an
   **LLM-as-judge** that runs offline by default and can call Claude
   (`claude-opus-4-8`) when configured.
3. **A performance-focused data-processing tool** — a single-pass, streaming
   corpus analyzer that handles datasets larger than memory, with a CLI and a
   built-in throughput benchmark.

The whole thing runs and tests **with no API key and no network access** — the
default judge is a deterministic heuristic — so it's easy to try, and easy to
run in CI.

```
┌──────────────┐   HTTP    ┌───────────────────────────────┐
│  client /    │ ───────▶ │  FastAPI app (evalhub.main)    │
│  demo.sh     │          │  ├─ /datasets     (CRUD)       │
└──────────────┘          │  ├─ /evaluations  (run + read) │
                          │  └─ /health       (probes)     │
                          └───────────────┬───────────────┘
                                          │
                     ┌────────────────────┼─────────────────────┐
                     ▼                    ▼                      ▼
            ┌────────────────┐   ┌──────────────────┐   ┌────────────────┐
            │ Evaluator      │   │ Scorers          │   │ Judge          │
            │ (async, bounded│   │ exact_match,     │   │ Heuristic |    │
            │  concurrency)  │   │ token_f1, ...    │   │ Anthropic      │
            └────────────────┘   └──────────────────┘   └────────────────┘

  Batch tool (evalhub-process): streaming JSONL → single-pass corpus stats
```

## Quick start

### Run the tests (no dependencies beyond Python)

```bash
make dev          # create .venv and install with dev + llm extras
make test         # run the full suite
make lint typecheck
```

### Run the API

```bash
make run                      # uvicorn with --reload on :8000
# then, in another shell:
./scripts/demo.sh             # exercises the API end to end
```

Interactive API docs are at <http://localhost:8000/docs>.

### Run in Docker

```bash
docker compose up --build     # builds the image and starts the service
curl localhost:8000/health
```

The image is multi-stage, runs as a non-root user, and ships a container
`HEALTHCHECK`.

## Using the API

Create a dataset and evaluate it:

```bash
# Create a dataset of model predictions + reference answers
curl -X POST localhost:8000/datasets -H 'content-type: application/json' -d '{
  "name": "capitals",
  "samples": [
    {"input": "Capital of France?",    "prediction": "Paris",  "reference": "Paris"},
    {"input": "Capital of Australia?", "prediction": "Sydney", "reference": "Canberra"}
  ]
}'

# Run scorers + the (offline) LLM judge over it
curl -X POST localhost:8000/evaluations -H 'content-type: application/json' -d '{
  "dataset_id": "<id from above>",
  "scorers": ["exact_match", "token_f1", "contains"],
  "judge": {"backend": "heuristic"}
}'
```

Each evaluation returns per-sample scores and per-scorer summary statistics
(mean/min/max), along with the wall-clock duration.

### Scorers

| name           | needs reference | description                                    |
| -------------- | :-------------: | ---------------------------------------------- |
| `exact_match`  | yes             | normalized string equality                     |
| `token_f1`     | yes             | SQuAD-style token-overlap F1                   |
| `contains`     | yes             | reference is a substring of the prediction     |
| `length_ratio` | yes             | prediction/reference length ratio, clamped     |
| `non_empty`    | no              | prediction has any content                     |

### LLM-as-judge

The judge produces a `0–1` quality score plus a short rationale per sample.

- **`heuristic`** (default): deterministic, offline — blends token-F1 with a
  length signal. Great for tests and for a fast baseline.
- **`anthropic`**: calls Claude with adaptive thinking and a structured-output
  JSON schema (so the response is always valid). Enable it by installing the
  `llm` extra and setting the environment:

  ```bash
  pip install "evalhub[llm]"
  export EVALHUB_JUDGE_BACKEND=anthropic
  export EVALHUB_ANTHROPIC_API_KEY=sk-ant-...
  ```

  If the backend is requested but no key is present, the platform degrades
  gracefully to the heuristic judge instead of failing the run.

## The batch-processing tool

`evalhub-process` streams a JSONL file of samples and reports corpus
statistics in a **single pass** with **O(vocabulary)** memory — it never loads
the whole file. Running aggregates use Welford's algorithm for numerically
stable mean/variance.

```bash
# Generate a synthetic dataset and analyze it with a throughput benchmark
evalhub-process gen 200000 --out data/big.jsonl
evalhub-process report data/big.jsonl --benchmark
```

```json
{
  "throughput_samples_per_sec": 812345.6,
  "elapsed_seconds": 0.2463,
  "report": {
    "samples": 200000,
    "prediction_tokens": {"count": 200000, "mean": 4.98, "stddev": 2.87, "min": 1, "max": 10},
    "vocabulary_size": 15,
    "scored_pairs": 200000,
    "mean_token_f1": 0.41
  }
}
```

## Configuration

All settings are environment variables prefixed with `EVALHUB_` (see
[`.env.example`](.env.example)):

| variable                      | default            | purpose                               |
| ----------------------------- | ------------------ | ------------------------------------- |
| `EVALHUB_ENVIRONMENT`         | `development`      | environment name                      |
| `EVALHUB_LOG_LEVEL`           | `info`             | root log level                        |
| `EVALHUB_LOG_JSON`            | `false`            | emit JSON logs (on in the container)  |
| `EVALHUB_MAX_CONCURRENCY`     | `16`               | concurrent samples per evaluation     |
| `EVALHUB_JUDGE_BACKEND`       | `heuristic`        | `heuristic` or `anthropic`            |
| `EVALHUB_ANTHROPIC_MODEL`     | `claude-opus-4-8`  | model id for the Anthropic judge      |
| `EVALHUB_ANTHROPIC_API_KEY`   | _(unset)_          | enables the Anthropic judge           |

## Project layout

```
src/evalhub/
  main.py            # FastAPI app factory + middleware + error handling
  config.py          # env-driven settings (pydantic-settings)
  logging.py         # plain / JSON logging
  api/
    deps.py          # dependency providers (settings, repo, evaluator)
    routes/          # health, datasets, evaluations
  core/
    models.py        # pydantic schemas (the platform's contract)
    scorers.py       # deterministic scorers + registry
    judge.py         # Judge protocol, heuristic + Anthropic backends
    evaluator.py     # async, bounded-concurrency evaluation engine
  processing/
    fast_metrics.py  # streaming single-pass corpus statistics
    cli.py           # evalhub-process console script
  store/
    repository.py    # async-safe in-memory store (swap for a DB later)
tests/               # unit + end-to-end tests (pytest)
Dockerfile           # multi-stage, non-root, healthcheck
docker-compose.yml
.github/workflows/   # CI: lint, type-check, test matrix, docker build
```

## Engineering notes

- **Typed contracts everywhere.** Pydantic v2 models are the single source of
  truth for the API, the engine, and the store.
- **Dependency injection** keeps handlers declarative and makes collaborators
  trivial to override in tests (`app.dependency_overrides`).
- **The store is an interface.** Moving from in-memory to Postgres/Redis means
  providing another `Repository` — no route changes.
- **Graceful degradation.** A configured-but-unavailable LLM judge falls back
  to the offline one rather than failing the request.
- **Fast by construction.** The batch tool is single-pass and streaming; the
  engine bounds concurrency with a semaphore so large datasets don't exhaust
  resources.
