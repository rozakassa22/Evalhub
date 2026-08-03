#!/usr/bin/env bash
# Smoke-test the running API end to end. Start the server first:
#   make run   (or)   docker compose up
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "== health =="
curl -fsS "$BASE_URL/health" | python3 -m json.tool

echo "== available scorers =="
curl -fsS "$BASE_URL/evaluations/scorers" | python3 -m json.tool

echo "== create a dataset =="
DATASET_ID=$(curl -fsS -X POST "$BASE_URL/datasets" \
  -H 'content-type: application/json' \
  -d '{
        "name": "capitals",
        "samples": [
          {"input": "Capital of France?", "prediction": "Paris", "reference": "Paris"},
          {"input": "Capital of Australia?", "prediction": "Sydney", "reference": "Canberra"}
        ]
      }' | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "created dataset: $DATASET_ID"

echo "== run an evaluation (with the offline LLM judge) =="
curl -fsS -X POST "$BASE_URL/evaluations" \
  -H 'content-type: application/json' \
  -d "{
        \"dataset_id\": \"$DATASET_ID\",
        \"scorers\": [\"exact_match\", \"token_f1\", \"contains\"],
        \"judge\": {\"backend\": \"heuristic\"}
      }" | python3 -m json.tool
