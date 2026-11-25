"""The HTTP layer for image create, driven through the real app with fakes behind it."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from app.api.deps import get_create_deps, get_run_service, get_token_signer
from app.guardrails.deps import get_guard_deps
from app.main import app
from tests.unit.create_helpers import critic_ok, make_deps, make_service, missing, ready
from tests.unit.guard_helpers import FakeJev, jev_answers
from tests.unit.guard_helpers import make_deps as make_guard_deps


@pytest.fixture
async def api(tmp_path: Path) -> AsyncIterator[tuple[httpx.AsyncClient, object, object]]:
    deps = make_deps(tmp_path, enhancer=[missing(), ready()], critic=[critic_ok(2, best=2)])
    service = make_service(deps)
    guard = make_guard_deps()
    app.dependency_overrides[get_create_deps] = lambda: deps
    app.dependency_overrides[get_run_service] = lambda: service
    app.dependency_overrides[get_token_signer] = lambda: deps.tokens
    app.dependency_overrides[get_guard_deps] = lambda: guard
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        yield client, service, guard
    app.dependency_overrides.clear()


def parse_sse(text: str) -> list[tuple[str, dict]]:  # type: ignore[type-arg]
    out = []
    for block in text.strip().split("\r\n\r\n"):
        lines = [x for x in block.replace("\r", "").split("\n") if x]
        ev = next((x[7:] for x in lines if x.startswith("event: ")), "")
        data = next((x[6:] for x in lines if x.startswith("data: ")), "{}")
        if ev:
            out.append((ev, json.loads(data)))
    return out


async def test_post_events_resume_and_saved_result(api) -> None:  # type: ignore[no-untyped-def]
    client, service, _ = api
    body = {"prompt": "make it nice", "plan": "pro", "settings": {"n": 2, "aspect_ratio": "1:1"}}
    created = (await client.post("/v1/runs", json=body)).json()
    run_id, token = created["run_id"], created["token"]

    first = parse_sse((await client.get(f"/v1/runs/{run_id}/events", params={"t": token})).text)
    assert first[-1][0] == "question"
    q = first[-1][1]["questions"][0]
    assert [o["id"] for o in q["options"]] == ["a", "b", "c", "other"]

    res = await client.post(f"/v1/runs/{run_id}/resume", json={"answers": {"q1": "a"}})
    assert res.status_code == 200
    second = parse_sse((await client.get(f"/v1/runs/{run_id}/events", params={"t": token})).text)
    kinds = [k for k, _ in second]
    assert "image" not in kinds and kinds[-1] == "ready"
    assert any(d.get("progress") == "2/2" for _, d in second)  # progress while images are made
    done = second[-1][1]
    assert len(done["images"]) == 2 and done["recommendation"]

    # the images can be fetched, and the result can be opened again
    url = done["images"][0]["url"]
    img = await client.get(url)
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    saved = (await client.get(f"/v1/runs/{run_id}")).json()
    assert saved["status"] == "done" and len(saved["result"]["images"]) == 2
    text = json.dumps(saved).lower()
    for hidden in ("seed", "provider", "negative_prompt", "lucid", "flux", "phoenix", "score"):
        assert hidden not in text, hidden


async def test_sse_events_carry_no_pipeline_details(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    created = (await client.post("/v1/runs", json={"prompt": "make it nice", "plan": "pro"})).json()
    run_id, token = created["run_id"], created["token"]
    text = (await client.get(f"/v1/runs/{run_id}/events", params={"t": token})).text.lower()
    for secret in (
        "trace",
        "langsmith",
        "enhancer",
        "critic",
        "score",
        "flux",
        "phoenix",
        "cloudflare",
    ):
        assert secret not in text


async def test_resume_checks(api) -> None:  # type: ignore[no-untyped-def]
    client, service, guard = api
    assert (await client.post("/v1/runs/nope/resume", json={"answers": {}})).status_code == 404
    created = (await client.post("/v1/runs", json={"prompt": "make it nice", "plan": "pro"})).json()
    run_id, token = created["run_id"], created["token"]
    await client.get(f"/v1/runs/{run_id}/events", params={"t": token})
    bad = await client.post(f"/v1/runs/{run_id}/resume", json={"answers": {}})
    assert bad.status_code == 400  # every question needs an answer
    other_user = await client.post(
        f"/v1/runs/{run_id}/resume", json={"answers": {"q1": "a"}}, headers={"X-User-Id": "eve"}
    )
    assert other_user.status_code == 404  # not your run
    await client.post(f"/v1/runs/{run_id}/resume", json={"answers": {"q1": "a"}})
    await service.wait(run_id)  # type: ignore[attr-defined]
    again = await client.post(f"/v1/runs/{run_id}/resume", json={"answers": {"q1": "a"}})
    assert again.status_code == 409  # no longer waiting


async def test_typed_answer_goes_through_the_guardrail(api) -> None:  # type: ignore[no-untyped-def]
    client, service, guard = api
    created = (await client.post("/v1/runs", json={"prompt": "make it nice", "plan": "pro"})).json()
    run_id, token = created["run_id"], created["token"]
    await client.get(f"/v1/runs/{run_id}/events", params={"t": token})
    guard.jev = FakeJev(jev_answers(nsfw_explicit=0.95))  # type: ignore[attr-defined]
    res = await client.post(
        f"/v1/runs/{run_id}/resume", json={"answers": {"q1": "other", "q1_text": "something rude"}}
    )
    assert res.status_code == 422 and res.json()["code"] == "content_blocked"
    assert service.runs[run_id].status == "waiting"  # type: ignore[attr-defined]  # still paused


async def test_expired_draft_cannot_resume_and_returns_the_draft(api) -> None:  # type: ignore[no-untyped-def]
    client, service, _ = api
    created = (await client.post("/v1/runs", json={"prompt": "make it nice", "plan": "pro"})).json()
    run_id, token = created["run_id"], created["token"]
    await client.get(f"/v1/runs/{run_id}/events", params={"t": token})
    await service._expire(service.runs[run_id])  # type: ignore[attr-defined]
    res = await client.post(f"/v1/runs/{run_id}/resume", json={"answers": {"q1": "a"}})
    assert res.status_code == 410
    got = (await client.get(f"/v1/runs/{run_id}")).json()
    assert got["status"] == "expired" and got["draft"]["prompt"] == "make it nice"


SHORTLIST = {
    "basic": ["flux-2-klein-4b", "gpt-image-2-5-flare", "recraft-v4-1-flash", "flux-3-klein"],
    "pro": ["seedream-5-flash", "flux-2-pro", "nano-banana-2-1", "recraft-v4-1"],
    "ultra": ["seedream-5-pro", "flux-3", "recraft-v4-1-vector"],
    "max": ["gpt-image-2-5-sunburst", "nano-banana-pro", "recraft-v4-1-pro"],
}


async def test_the_picker_shows_only_the_curated_14_and_locks_by_plan(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    everything = [m for ids in SHORTLIST.values() for m in ids]
    assert len(everything) == 14
    # Free (Basic) users get models up to the Pro tier. Pro and above can use any model
    unlocked_by_plan = {"basic": 8, "pro": 14, "ultra": 14, "max": 14}
    for plan, expected_open in unlocked_by_plan.items():
        res = (await client.get("/v1/models", params={"plan": plan})).json()
        by_id = {m["id"]: m for m in res["models"]}
        assert set(by_id) == set(everything)  # the same 14 on every plan. Nothing else is shown
        assert sum(not m["locked"] for m in res["models"]) == expected_open
        open_tiers = list(SHORTLIST)[:2] if plan == "basic" else list(SHORTLIST)
        want_open = [m for t in open_tiers for m in SHORTLIST[t]]
        assert {m for m, v in by_id.items() if not v["locked"]} == set(want_open)
    basic = (await client.get("/v1/models", params={"plan": "basic"})).json()
    by_id = {m["id"]: m for m in basic["models"]}
    # the old and overlapping models, and the free Cloudflare engines, are not in the picker
    for hidden in (
        "flux-1-schnell",
        "flux-2-klein-9b",
        "phoenix-1",
        "lucid-origin",
        "nano-banana-2",
        "gpt-image-1",
        "gpt-image-2",
        "gpt-image-2-6-flare",
        "recraft-v4",
        "recraft-v4-pro",
        "flux-2-flex",
        "flux-2-max",
        "krea-2-large",
        "muse-image",
        "recraft-v4-1-flash-free",
    ):
        assert hidden not in by_id, hidden
    # only the Cloudflare model is usable today. The rest are "coming soon" (OpenRouter is off)
    assert by_id["flux-2-klein-4b"]["available"] is True
    assert [m["id"] for m in basic["models"] if m["available"]] == ["flux-2-klein-4b"]
    assert all(by_id[m]["available"] is False for m in everything if m != "flux-2-klein-4b")
    avail = [m["available"] for m in basic["models"]]
    assert avail == sorted(avail, reverse=True)  # usable models first, "coming soon" after
    assert by_id["nano-banana-2-1"]["family_label"] == "Google Nano Banana & Gemini"
    assert by_id["recraft-v4-1"]["tag"] == "Posters, ads, brand colors"
    assert "openrouter" not in json.dumps(basic).lower()  # no provider names go out
    assert "cost" not in json.dumps(basic["models"]) and "slug" not in json.dumps(basic["models"])
    plans = (await client.get("/v1/plans")).json()
    assert {p["id"]: p["max_n"] for p in plans["plans"]} == {
        "basic": 2,
        "pro": 4,
        "ultra": 4,
        "max": 4,
    }
    assert {"id": "2K", "available": False} in plans["resolutions"]


async def test_the_auto_pool_prices_every_model_auto_can_pick_even_hidden_ones(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    basic = (await client.get("/v1/models", params={"plan": "basic"})).json()["auto_pool"]
    pro = (await client.get("/v1/models", params={"plan": "pro"})).json()["auto_pool"]
    assert all(
        set(e) == {"max_refs", "credits_per_image", "ref_surcharge", "resolution_multipliers"}
        for e in basic
    )  # prices only. No names
    # OpenRouter is off, so Auto uses the free Cloudflare engines. Free users get models up to
    # the Pro tier, so that includes Phoenix (3 credits), which is not in the picker. The
    # estimate must still count it
    assert max(e["credits_per_image"] for e in basic) == 3
    assert max(e["credits_per_image"] for e in pro) == 3


async def test_files_endpoint_rejects_json_and_unknown(api) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = api
    assert (await client.get("/v1/files/nope/img_1.jpg")).status_code == 404
    assert (await client.get("/v1/files/nope/result.json")).status_code == 404
