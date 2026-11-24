import base64
from collections.abc import Iterator

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.guardrails.agent import GuardDeps
from app.guardrails.deps import get_guard_deps
from app.main import app
from tests.unit.guard_helpers import FakeJev, FakeVision, jev_answers, make_deps, png

client = TestClient(app)


@pytest.fixture
def deps() -> Iterator[GuardDeps]:
    d = make_deps()
    app.dependency_overrides[get_guard_deps] = lambda: d
    yield d
    app.dependency_overrides.clear()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def test_health() -> None:
    assert client.get("/health").status_code == 200


def test_run_and_events(deps: GuardDeps) -> None:
    run_id = client.post("/v1/runs", json={"prompt": "a red fox"}).json()["run_id"]
    body = client.get(f"/v1/runs/{run_id}/events").text
    assert "event: progress" in body
    assert "echo: a red fox" in body


def test_unknown_run_is_404() -> None:
    assert client.get("/v1/runs/nope/events").status_code == 404


def test_approved_run_stores_raw_prompt_and_original_images(deps: GuardDeps) -> None:
    from app.api.runs import RUNS

    image = b64(png())
    run_id = client.post("/v1/runs", json={"prompt": " N u d e  FOX", "images": [image]}).json()[
        "run_id"
    ]
    assert RUNS[run_id].prompt == " N u d e  FOX"
    assert RUNS[run_id].images == [image]


def test_blocked_prompt_is_422_with_safe_body(deps: GuardDeps) -> None:
    deps.jev = FakeJev(jev_answers(nsfw_explicit=0.93))
    res = client.post("/v1/runs", json={"prompt": "something"})
    assert res.status_code == 422
    assert res.json() == {
        "code": "content_blocked",
        "category": "NSFW_EXPLICIT",
        "blocked_by": "prompt",
        "message": "Your request asks for nudity or sexual content.",
        "tip": "Try a fully clothed version.",
        "warning": False,
        "retry_after": 60,
        "retry_text": "You can try again in 1 minute.",
    }
    assert "run_id" not in res.json()
    # next request is inside the cooldown
    again = client.post("/v1/runs", json={"prompt": "a fox"})
    assert again.status_code == 429
    assert again.json()["code"] == "cooldown"
    assert 0 < again.json()["retry_after"] <= 60
    assert "retry-after" in again.headers


def test_blocked_image_instant_reject_has_strike_text(deps: GuardDeps) -> None:
    deps.vision = FakeVision("unsafe\nS4", 0.97)
    res = client.post("/v1/runs", json={"prompt": "a photo", "images": [b64(png())]})
    body = res.json()
    assert res.status_code == 422
    assert (body["category"], body["blocked_by"], body["warning"]) == ("CSAM", "image", True)
    assert body["strike_count"] == 1
    assert body["strike_text"] == "Strike 1 of 5. At 5 strikes your account will be banned."
    assert body["tip"] is None
    for hidden in ("reason", "scores", "verdicts", "confidence", "fake-vision"):
        assert hidden not in str(body)


def test_banned_user_gets_403(deps: GuardDeps) -> None:
    deps.jev = FakeJev(jev_answers(csam=0.97))
    for _ in range(5):
        client.post("/v1/runs", json={"prompt": "x"}, headers={"X-User-Id": "bad"})
    res = client.post("/v1/runs", json={"prompt": "a fox"}, headers={"X-User-Id": "bad"})
    assert res.status_code == 403 and res.json()["code"] == "account_restricted"
    deps.jev = FakeJev()  # other users are not affected
    assert (
        client.post("/v1/runs", json={"prompt": "a fox"}, headers={"X-User-Id": "ok"}).status_code
        == 200
    )


def test_guardrail_failure_is_503(deps: GuardDeps) -> None:
    deps.jev = FakeJev({})
    res = client.post("/v1/runs", json={"prompt": "a fox"})
    assert res.status_code == 503
    assert res.json() == {
        "code": "guardrail_unavailable",
        "message": "Safety verification timed out. Please try again.",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "a" * 4001},
        {"prompt": ""},
        {"prompt": "x", "images": ["!!!not base64"]},
        {"prompt": "x", "images": [b64(b"not an image")]},
    ],
)
def test_bad_input_is_400(deps: GuardDeps, payload: dict) -> None:  # type: ignore[type-arg]
    res = client.post("/v1/runs", json=payload)
    assert res.status_code == 400 and res.json()["code"] == "bad_input"


def test_data_url_images_are_accepted(deps: GuardDeps) -> None:
    img = "data:image/png;base64," + b64(png())
    assert client.post("/v1/runs", json={"prompt": "x", "images": [img]}).status_code == 200


def test_end_to_end_through_real_providers_with_respx() -> None:
    """Build-mode wiring: Cloudflare Jev + Scout stand-in, both mocked at the HTTP level."""
    from app.core.config import get_settings
    from app.guardrails.deps import get_guard_deps as real

    settings = get_settings()
    settings.cloudflare_account_id, settings.cloudflare_api_token = "acc", "tok"
    real.cache_clear()
    try:
        with respx.mock:
            respx.post(
                "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/typesafe/jev"
            ).mock(
                return_value=httpx.Response(
                    200, json={"success": True, "result": {"answers": jev_answers()}}
                )
            )
            respx.post(
                "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
            ).mock(
                return_value=httpx.Response(
                    200, json={"success": True, "result": {"response": "safe"}}
                )
            )
            res = client.post("/v1/runs", json={"prompt": "a fox", "images": [b64(png())]})
        assert res.status_code == 200
    finally:
        real.cache_clear()
        settings.cloudflare_account_id = settings.cloudflare_api_token = ""
