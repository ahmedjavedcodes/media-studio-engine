from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").status_code == 200


def test_run_and_events() -> None:
    run_id = client.post("/v1/runs", json={"prompt": "a red fox"}).json()["run_id"]
    body = client.get(f"/v1/runs/{run_id}/events").text
    assert "event: progress" in body
    assert "echo: a red fox" in body


def test_unknown_run_is_404() -> None:
    assert client.get("/v1/runs/nope/events").status_code == 404
