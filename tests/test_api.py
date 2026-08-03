"""End-to-end API tests using FastAPI's TestClient."""
from __future__ import annotations


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "evalhub"


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/health/ready").json()["status"] == "ready"
    # Timing middleware sets the header on every response.
    assert "X-Process-Time-ms" in client.get("/health").headers


def test_list_scorers(client):
    resp = client.get("/evaluations/scorers")
    assert resp.status_code == 200
    assert "token_f1" in resp.json()


def test_seeded_demo_dataset_present(client):
    resp = client.get("/datasets/demo")
    assert resp.status_code == 200
    assert resp.json()["name"].startswith("Capitals")


def test_create_and_get_dataset(client):
    body = {
        "name": "my-set",
        "samples": [
            {"prediction": "Paris", "reference": "Paris"},
            {"prediction": "Rome", "reference": "Rome"},
        ],
    }
    created = client.post("/datasets", json=body)
    assert created.status_code == 201
    dataset_id = created.json()["id"]

    fetched = client.get(f"/datasets/{dataset_id}")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "my-set"
    assert len(fetched.json()["samples"]) == 2


def test_create_dataset_requires_samples(client):
    resp = client.post("/datasets", json={"name": "empty", "samples": []})
    assert resp.status_code == 422


def test_get_missing_dataset_404(client):
    assert client.get("/datasets/nope").status_code == 404


def test_run_evaluation_on_demo(client):
    resp = client.post(
        "/evaluations",
        json={"dataset_id": "demo", "scorers": ["exact_match", "token_f1"]},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "completed"
    assert body["summary"]["exact_match"]["count"] == 3
    assert len(body["sample_scores"]) == 3

    # The result is retrievable afterwards.
    eval_id = body["id"]
    again = client.get(f"/evaluations/{eval_id}")
    assert again.status_code == 200
    assert again.json()["id"] == eval_id


def test_run_evaluation_with_judge(client):
    resp = client.post(
        "/evaluations",
        json={
            "dataset_id": "demo",
            "scorers": ["exact_match"],
            "judge": {"backend": "heuristic"},
        },
    )
    assert resp.status_code == 201
    assert "llm_judge" in resp.json()["summary"]


def test_run_evaluation_unknown_dataset_404(client):
    resp = client.post("/evaluations", json={"dataset_id": "missing"})
    assert resp.status_code == 404


def test_run_evaluation_unknown_scorer_422(client):
    resp = client.post(
        "/evaluations", json={"dataset_id": "demo", "scorers": ["bogus"]}
    )
    assert resp.status_code == 422


def test_get_missing_evaluation_404(client):
    assert client.get("/evaluations/nope").status_code == 404
