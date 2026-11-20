"""The dev shim (scripts/dev_server.py) against the contract in project-status/ARCHITECTURE.md.
Skipped when fastapi is not installed: it is a dev-only extra, not a project dependency."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from media_engine.ports import LocalArtifactStore
from tests.unit.harness.fakes import ScriptedChat, ScriptedImages, image_bytes
from tests.unit.harness.test_entry import ROUTE_SIMPLE, Backend
from tests.unit.harness.test_supervisor import enhanced, verdict

SHIM = Path(__file__).parents[2] / "scripts" / "dev_server.py"
PUBLIC = {"Understanding your request", "Creating your image", "Finishing up"}


@pytest.fixture
def shim(tmp_path):
    spec = importlib.util.spec_from_file_location("dev_server", SHIM)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["dev_server"] = module  # pydantic resolves the shim's annotations through it
    spec.loader.exec_module(module)
    module.ARTIFACTS = LocalArtifactStore(tmp_path / ".artifacts")
    module.POLL_SECONDS = 0.01
    return module


def sse(client: TestClient, run_id: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    with client.stream("GET", f"/runs/{run_id}/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        name = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:])))
    return events


def test_a_run_streams_public_events_then_one_result_with_image_urls(shim):
    shim.GATEWAY = Backend(
        ScriptedChat([ROUTE_SIMPLE, enhanced(), verdict(9)]), ScriptedImages([image_bytes()])
    )
    with TestClient(shim.app) as client:
        posted = client.post(
            "/runs", json={"prompt": "a cozy coffee shop", "plan": "free", "mode": "auto", "n": 1}
        )
        assert posted.status_code == 202
        run_id = posted.json()["run_id"]

        events = sse(client, run_id)
        steps = [data for name, data in events if name == "step"]
        results = [data for name, data in events if name == "result"]

        assert {s["public_label"] for s in steps} <= PUBLIC
        base = {"run_id", "step", "public_label", "phase", "ts"}
        for s in steps:  # only image_ready adds a key: the image id, after the Critic allowed it
            assert set(s) == (base | {"image_id"} if s["phase"] == "image_ready" else base)
        assert len(results) == 1 and events[-1][0] == "result"  # exactly one, and last
        result = results[0]
        assert result["status"] == "done"
        (image,) = result["images"]
        assert image["url"] == f"/images/{image['image_id']}" and image["quality_score"] == 9
        assert "base64" not in json.dumps(result)

        assert client.get(f"/runs/{run_id}").json()["status"] == "done"
        file = client.get(image["url"])
        assert file.status_code == 200 and file.headers["content-type"].startswith("image/")


def test_a_stub_operation_returns_escalate(shim):
    shim.GATEWAY = Backend(ScriptedChat([ROUTE_SIMPLE]), ScriptedImages([]))
    with TestClient(shim.app) as client:
        body = {"prompt": "a comic", "operation": "edit"}
        run_id = client.post("/runs", json=body).json()["run_id"]
        (result,) = [d for name, d in sse(client, run_id) if name == "result"]
        assert result["status"] == "escalate" and result["images"] == []


def test_cancel_sets_the_flag_on_the_backend_side_of_the_store(shim):
    shim.GATEWAY = Backend(
        ScriptedChat([ROUTE_SIMPLE, enhanced(), verdict(9)]), ScriptedImages([image_bytes()])
    )
    with TestClient(shim.app) as client:
        run_id = client.post("/runs", json={"prompt": "a cat"}).json()["run_id"]
        assert client.post(f"/runs/{run_id}/cancel").status_code == 202
        sse(client, run_id)  # let the run end
    import asyncio

    assert asyncio.run(shim.RUN_STORE.cancel_requested(run_id)) is True


def test_unknown_run_and_image_are_404(shim):
    with TestClient(shim.app) as client:
        assert client.get("/runs/nope").status_code == 404
        assert client.post("/runs/nope/cancel").status_code == 404
        assert client.get("/images/img_nope").status_code == 404
        assert client.post("/runs", json={"prompt": "x", "n": 9}).status_code == 422


def test_the_fake_backend_drives_every_outcome_over_the_contract(shim):
    from dev_fake_backend import FakeBackend

    shim.GATEWAY = FakeBackend(delay=0)
    cases = [
        ({"prompt": "a cozy cafe"}, "done"),
        ({"prompt": "a cozy cafe", "plan": "max", "n": 3}, "done"),
        ({"prompt": "reject this"}, "rejected"),
        ({"prompt": "something unsafe"}, "rejected"),  # the Gate blocks, before any image
        ({"prompt": "a borderline battle"}, "done"),  # Gate says review, the reviewer allows
        ({"prompt": "fail this"}, "failed"),
        ({"prompt": "a comic", "operation": "edit"}, "escalate"),  # a stub operation
        ({"prompt": "question me", "mode": "manual"}, "needs_input"),
        ({"prompt": "issues please"}, "done_with_issues"),
    ]
    with TestClient(shim.app) as client:
        for body, want in cases:
            run_id = client.post("/runs", json=body).json()["run_id"]
            (result,) = [d for name, d in sse(client, run_id) if name == "result"]
            assert result["status"] == want, (body, result["status"], result["reason"])
        # done_with_issues: create is linear (v2.2), so the one low-scoring image is delivered
        # with its critique as an issue, and nothing is regenerated
        assert len(result["images"]) == 1 and result["issues"]

        # answering the question re-runs to done
        asked = client.post("/runs", json={"prompt": "question me", "mode": "manual"}).json()
        (first,) = [d for n, d in sse(client, asked["run_id"]) if n == "result"]
        (question,) = first["questions"]
        again = client.post(
            "/runs",
            json={"prompt": "question me", "mode": "manual", "settings": {question: "Paris"}},
        ).json()
        (final,) = [d for n, d in sse(client, again["run_id"]) if n == "result"]
        assert final["status"] == "done" and len(final["images"]) == 1


# -- the developer trace channel (GET /runs/:id/trace, DEBUG_TRACE=1 only) ------------------------


def test_the_trace_endpoint_is_off_unless_debug_trace_is_set(shim):
    from dev_fake_backend import FakeBackend

    shim.GATEWAY = FakeBackend(delay=0)
    shim.DEBUG_TRACE = False
    with TestClient(shim.app) as client:
        run_id = client.post("/runs", json={"prompt": "a cozy cafe"}).json()["run_id"]
        sse(client, run_id)
        assert client.get(f"/runs/{run_id}/trace").status_code == 404


def test_the_trace_endpoint_serves_the_inspector_data_without_changing_the_public_channels(shim):
    from dev_fake_backend import FakeBackend

    shim.GATEWAY = FakeBackend(delay=0)
    shim.DEBUG_TRACE = True
    with TestClient(shim.app) as client:
        run_id = client.post("/runs", json={"prompt": "a cozy cafe"}).json()["run_id"]
        events = sse(client, run_id)
        trace = client.get(f"/runs/{run_id}/trace").json()
        assert client.get("/runs/nope/trace").status_code == 404

    assert [c["role"] for c in trace["calls"]] == [
        "gate",
        "prompt_enhancer",
        "image_creator",
        "critic",
    ]
    assert trace["calls"][0]["seq"] == 1 and trace["status"] == "done"
    assert {
        "decisions",
        "steps",
        "attempts",
        "budget",
        "supervisor_spend",
        "retries_used",
        "critic_reasks_used",
        "orphaned",
    } <= set(trace)
    assert all(s["public_label"] and s["agent"] for s in trace["steps"])
    assert "base64" not in json.dumps(trace)

    # the public channels carry none of it
    public = json.dumps([d for _, d in events])
    secrets = ("prompt_enhancer", "image_creator", "critic", "gate", "provider", "(fake)")
    for secret in (*secrets, "safety_reviewer", "groq", "cloudflare", "gpt-oss", "qwen", "@cf/"):
        assert secret not in public, secret
    # the trace (developer channel) does name the models the fake stood in for
    served = {(c["provider"], c["model"]) for c in trace["calls"]}
    assert ("groq (fake)", "openai/gpt-oss-120b") in served
    assert ("cloudflare (fake)", "@cf/black-forest-labs/flux-1-schnell") in served
