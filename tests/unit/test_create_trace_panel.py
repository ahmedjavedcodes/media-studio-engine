"""The dev trace panel: a local record of every step, and the token-protected /trace endpoint."""

import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from app.agents.image_grand.workers.safety import Safety, check_image_detail
from app.api.deps import get_create_deps, get_run_service, get_token_signer
from app.contracts.request import GenerationRequest, ImageSettings
from app.core.plans import Plan
from app.core.tracing import Tracer
from app.gateway.providers.cloudflare import CloudflareClient
from app.gateway.providers.cloudflare_create import CloudflareLLM, add_usage
from app.guardrails.deps import get_guard_deps
from app.main import app
from tests.unit.create_helpers import (
    SeqVision,
    critic_ok,
    jpeg,
    make_deps,
    make_service,
    missing,
    ready,
)
from tests.unit.guard_helpers import make_deps as make_guard_deps


def req(prompt: str = "iced coffee ad", **kw: Any) -> GenerationRequest:
    plan = kw.pop("plan", Plan.PRO)
    return GenerationRequest(prompt=prompt, plan=plan, settings=ImageSettings(**kw))


async def run(service: Any, run_id: str, r: GenerationRequest, **kw: Any) -> list[Any]:
    service.start(run_id, "u1", r, **kw)
    return [e async for e in service.stream(run_id)]


def by_name(snap: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for sp in snap["spans"]:
        out.setdefault(sp["name"], []).append(sp)
    return out


GUARD = {
    "started_at": time.time() - 0.4,
    "ended_at": time.time() - 0.1,
    "outputs": {
        "decision": "APPROVE",
        "category": "SAFE",
        "confidence": 0.02,
        "timings_ms": {"total": 300},
        "checks": [{"source": "jev", "model": "jev-1", "latency_ms": 280}],
        "model": "jev-1",
    },
}


# ---- the local record ----------------------------------------------------------------------------


async def test_a_run_records_every_step_with_time_model_provider_and_cost(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(2)], trace_ui=True)
    service = make_service(deps)
    await run(service, "t1", req(n=2), guard=GUARD)
    await service.wait("t1")
    trace = deps.tracer.trace_for("t1")
    assert trace is not None  # kept after the run, for the panel
    snap = trace.snapshot("done")
    spans = by_name(snap)

    for step in (
        "guardrail",
        "code_checks",
        "router",
        "enhancer",
        "code_guard",
        "generate",
        "critic",
        "settle",
    ):
        assert step in spans, step
    assert len(spans["generate_image"]) == 2 and len(spans["safety"]) == 2
    # a step knows how long it took and when it began, and the guardrail came first
    assert all(sp["duration_ms"] >= 0 and isinstance(sp["start_ms"], int) for sp in snap["spans"])
    assert snap["spans"][0]["name"] == "guardrail" and snap["spans"][0]["start_ms"] <= 0
    # the tree: generate -> generate_image -> safety
    gen_id = spans["generate"][0]["id"]
    assert {s["parent"] for s in spans["generate_image"]} == {gen_id}
    image_ids = {s["id"] for s in spans["generate_image"]}
    assert {s["parent"] for s in spans["safety"]} == image_ids
    # model and provider on the steps that use one
    assert (
        spans["router"][0]["model"] == "phoenix-1"
        and spans["router"][0]["provider"] == "cloudflare"
    )
    assert spans["enhancer"][0]["model"] == "fake-enhancer"
    assert spans["critic"][0]["model"] == "fake-critic"
    img = spans["generate_image"][0]["outputs"]
    assert img["model"] == "phoenix-1" and img["provider"] == "cloudflare"
    assert img["slug"] == "@cf/leonardo/phoenix-1.0" and img["cost_usd"] == 0.001
    assert img["cost_estimate_usd"] > 0 and img["size"] and img["latency_ms"] >= 0
    assert img["options"]["resolution"] is None and img["seed_sent"] is True
    assert spans["safety"][0]["outputs"]["verdict"] == "safe"
    assert spans["safety"][0]["model"] == "fake-safety"


async def test_the_totals_add_up(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(2)], trace_ui=True)
    service = make_service(deps)
    await run(service, "t2", req(n=2))
    await service.wait("t2")
    totals = deps.tracer.trace_for("t2").snapshot("done")["totals"]  # type: ignore[union-attr]
    assert totals["images_requested"] == 2 and totals["images_delivered"] == 2
    assert totals["images_usd"] == pytest.approx(0.002) and not totals["images_cost_estimated"]
    # enhancer 10 + critic 10 + two safety checks 2 each
    assert totals["neurons"] == pytest.approx(24.0)
    assert totals["tokens"] == {"prompt": 200, "completion": 100}
    assert totals["llm_usd_estimate"] == pytest.approx(24.0 * 0.011 / 1000, rel=1e-3)
    assert totals["total_usd"] == pytest.approx(totals["images_usd"] + totals["llm_usd_estimate"])
    assert totals["credits_held"] == 6 and totals["credits_charged"] == 6  # phoenix: 3 x 2
    assert totals["duration_ms"] >= 0


async def test_a_cost_the_provider_did_not_report_is_marked_as_an_estimate(tmp_path: Path) -> None:
    from app.contracts.ports import GeneratedImage
    from tests.unit.create_helpers import FakeProvider

    class NoCost(FakeProvider):
        async def generate(self, **kw: Any) -> GeneratedImage:
            return GeneratedImage(data=jpeg(), cost_usd=None)

    deps = make_deps(tmp_path, provider=NoCost(), critic=[critic_ok(1)], trace_ui=True)
    service = make_service(deps)
    await run(service, "t3", req(n=1))
    await service.wait("t3")
    totals = deps.tracer.trace_for("t3").snapshot("done")["totals"]  # type: ignore[union-attr]
    assert totals["images_cost_estimated"] and totals["images_usd"] > 0


async def test_the_code_guard_shows_what_it_changed(tmp_path: Path) -> None:
    weak = ready("a subtle, serene iced coffee on a table")
    deps = make_deps(tmp_path, enhancer=[weak], critic=[critic_ok(1)], trace_ui=True)
    service = make_service(deps)
    await run(service, "t4", req(n=1))
    await service.wait("t4")
    guard_span = by_name(deps.tracer.trace_for("t4").snapshot("done"))["code_guard"][0]  # type: ignore[union-attr]
    out = guard_span["outputs"]
    assert out["llm_prompt"] == "a subtle, serene iced coffee on a table"
    assert "subtle" not in out["final_prompt"] and "serene" not in out["final_prompt"]
    assert out["prompt_changed"] is True
    assert out["max_words"] == 60 and out["size"].count("x") == 1


async def test_a_question_adds_an_interrupt_span_and_a_second_enhancer_span(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, enhancer=[missing(), ready()], critic=[critic_ok(1)], trace_ui=True)
    service = make_service(deps)
    await run(service, "t5", req("make it nice", n=1))
    live = by_name(deps.tracer.trace_for("t5").snapshot("waiting"))  # type: ignore[union-attr]
    assert live["interrupt"][0]["status"] == "running"  # the user has not answered yet
    await service.resume("t5", {"q1": "A landscape"})
    [e async for e in service.stream("t5")]
    await service.wait("t5")
    done = by_name(deps.tracer.trace_for("t5").snapshot("done"))  # type: ignore[union-attr]
    assert len(done["enhancer"]) == 2 and done["interrupt"][0]["status"] == "ok"
    assert done["interrupt"][0]["outputs"]["answers"] == {"q1": "A landscape"}
    assert done["enhancer"][0]["outputs"]["status"] == "missing"


async def test_a_running_run_shows_open_spans(tmp_path: Path) -> None:
    tracer = Tracer(client=None, project="x", enabled=False)
    trace = tracer.start_run("r", metadata={"run_id": "r", "user_id": "u"}, inputs={"prompt": "p"})
    span = trace.start("enhancer", inputs={"prompt": "p"})
    snap = trace.snapshot("running")
    assert snap["spans"][0]["status"] == "running" and snap["status"] == "running"
    span.end()
    assert trace.snapshot("done")["spans"][0]["status"] == "ok"


async def test_a_failed_step_is_marked_as_an_error(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, vision=SeqVision(["unsafe\nS12"]), trace_ui=True)
    service = make_service(deps)
    events = await run(service, "t6", req(n=1))
    await service.wait("t6")
    assert events[-1].type == "error"
    snap = deps.tracer.trace_for("t6").snapshot("failed")  # type: ignore[union-attr]
    assert by_name(snap)["safety"][0]["outputs"]["verdict"] == "unsafe"
    assert snap["status"] == "failed"


async def test_the_trace_holds_no_image_bytes(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(1)], trace_ui=True)
    service = make_service(deps)
    await run(service, "t7", req(n=1))
    await service.wait("t7")
    text = json.dumps(deps.tracer.trace_for("t7").snapshot("done"))  # type: ignore[union-attr]
    assert "data:image" not in text and "base64" not in text and "\\xff\\xd8" not in text


# ---- usage from the models -----------------------------------------------------------------------


def test_usage_adds_up_over_calls() -> None:
    sink: dict[str, Any] = {}
    add_usage(
        sink, {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "neurons": 1.5, "x": 1}}
    )
    add_usage(sink, {"usage": {"prompt_tokens": 20, "neurons": 2.0}})
    add_usage(sink, {"usage": "nope"})
    add_usage(None, {"usage": {"neurons": 9}})  # no sink: nothing happens
    assert sink == {"prompt_tokens": 30, "completion_tokens": 5, "neurons": 3.5}


@respx.mock
async def test_the_cloudflare_llm_reports_its_usage() -> None:
    from app.agents.image_grand.workers.brief import EnhancerOutput

    url = "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/@cf/openai/gpt-oss-20b"
    answer = ready().model_dump_json()
    respx.post(url).mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "result": {
                    "choices": [{"message": {"content": answer}}],
                    "usage": {"prompt_tokens": 94, "completion_tokens": 26, "neurons": 6.8},
                },
            },
        )
    )
    async with httpx.AsyncClient() as http:
        llm = CloudflareLLM(
            CloudflareClient(http, account_id="acc", api_token="t"), "@cf/openai/gpt-oss-20b"
        )
        usage: dict[str, Any] = {}
        await llm.complete_json(
            system="s",
            user="u",
            schema=EnhancerOutput,
            max_tokens=50,
            temperature=0.2,
            timeout=5,
            usage=usage,
        )
    assert usage == {"prompt_tokens": 94, "completion_tokens": 26, "neurons": 6.8}


async def test_a_safety_check_reports_its_time_model_and_neurons() -> None:
    result = await check_image_detail(SeqVision(["safe"]), jpeg(), "p", timeout=5)
    assert result.verdict is Safety.SAFE and result.model == "fake-safety"
    assert result.neurons == 2.0 and result.ended_at >= result.started_at
    assert result.p_unsafe is not None and result.p_unsafe < 0.1
    failed = await check_image_detail(SeqVision([RuntimeError("x")]), jpeg(), "p", timeout=5)
    assert failed.verdict is Safety.FAILED and failed.error == "RuntimeError"


# ---- the endpoint --------------------------------------------------------------------------------


@pytest.fixture
async def api(tmp_path: Path) -> AsyncIterator[Any]:
    async def build(**kw: Any) -> tuple[httpx.AsyncClient, Any, Any]:
        deps = make_deps(tmp_path, critic=[critic_ok(1)], **kw)
        service = make_service(deps)
        app.dependency_overrides[get_create_deps] = lambda: deps
        app.dependency_overrides[get_run_service] = lambda: service
        app.dependency_overrides[get_token_signer] = lambda: deps.tokens
        app.dependency_overrides[get_guard_deps] = lambda: make_guard_deps()
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
        return client, deps, service

    opened: list[httpx.AsyncClient] = []

    async def factory(**kw: Any) -> tuple[httpx.AsyncClient, Any, Any]:
        c, d, s = await build(**kw)
        opened.append(c)
        return c, d, s

    yield factory
    for c in opened:
        await c.aclose()
    app.dependency_overrides.clear()


async def start(c: httpx.AsyncClient, body: dict[str, Any] | None = None) -> tuple[str, str]:
    created = (
        await c.post("/v1/runs", json=body or {"prompt": "iced coffee ad", "plan": "pro"})
    ).json()
    return created["run_id"], created["token"]


async def test_the_trace_endpoint_is_off_by_default(api) -> None:  # type: ignore[no-untyped-def]
    c, _, _ = await api()  # TRACE_UI is false
    run_id, token = await start(c)
    await c.get(f"/v1/runs/{run_id}/events", params={"t": token})
    res = await c.get(f"/v1/runs/{run_id}/trace", params={"t": token})
    assert res.status_code == 404  # a valid token is not enough
    assert (await c.get("/v1/plans")).json()["trace_ui"] is False


async def test_the_trace_endpoint_returns_the_whole_run_when_on(api) -> None:  # type: ignore[no-untyped-def]
    c, _, _ = await api(trace_ui=True)
    run_id, token = await start(c)
    await c.get(f"/v1/runs/{run_id}/events", params={"t": token})
    res = await c.get(f"/v1/runs/{run_id}/trace", params={"t": token})
    assert res.status_code == 200 and res.headers["cache-control"] == "no-store"
    snap = res.json()
    names = [s["name"] for s in snap["spans"]]
    assert names[0] == "guardrail"  # the real guardrail ran first, and is in the trace
    assert {
        "code_checks",
        "router",
        "enhancer",
        "code_guard",
        "generate",
        "generate_image",
        "safety",
        "critic",
        "settle",
    } <= set(names)
    guard = snap["spans"][0]["outputs"]
    assert guard["decision"] == "APPROVE" and guard["checks"] and guard["timings_ms"]
    assert snap["input"]["prompt"] == "iced coffee ad" and snap["input"]["plan"] == "pro"
    assert snap["totals"]["images_delivered"] == 1 and snap["status"] == "done"
    assert (await c.get("/v1/plans")).json()["trace_ui"] is True


@pytest.mark.parametrize("bad", ["", "nonsense", "9999999999.deadbeef"])
async def test_the_trace_needs_a_valid_token(api, bad: str) -> None:  # type: ignore[no-untyped-def]
    c, _, _ = await api(trace_ui=True)
    run_id, token = await start(c)
    other_id, other_token = await start(c)
    await c.get(f"/v1/runs/{run_id}/events", params={"t": token})
    assert (await c.get(f"/v1/runs/{run_id}/trace", params={"t": bad})).status_code == 404
    assert (await c.get(f"/v1/runs/{run_id}/trace", params={"t": other_token})).status_code == 404
    assert other_id != run_id


async def test_the_trace_is_never_on_in_prod(api) -> None:  # type: ignore[no-untyped-def]
    c, _, _ = await api(trace_ui=True, app_env="prod")
    run_id, token = await start(c)
    await c.get(f"/v1/runs/{run_id}/events", params={"t": token})
    assert (await c.get(f"/v1/runs/{run_id}/trace", params={"t": token})).status_code == 404
    assert (await c.get("/v1/plans")).json()["trace_ui"] is False


async def test_the_sse_stream_still_carries_no_trace_data(api) -> None:  # type: ignore[no-untyped-def]
    c, _, _ = await api(trace_ui=True)
    run_id, token = await start(c)
    text = (await c.get(f"/v1/runs/{run_id}/events", params={"t": token})).text.lower()
    for word in ("duration_ms", "neurons", "cost_usd", "slug", "span", "latency", "guardrail"):
        assert word not in text, word


async def test_a_failed_runs_trace_survives_its_deleted_files(api, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    c, deps, service = await api(trace_ui=True, vision=SeqVision(["unsafe\nS12"]))
    run_id, token = await start(c)
    events = (await c.get(f"/v1/runs/{run_id}/events", params={"t": token})).text
    assert "event: error" in events
    await service.wait(run_id)
    assert not (tmp_path / run_id).exists()  # the images are gone
    assert (tmp_path / "traces" / f"{run_id}.json").exists()  # the trace is not
    deps.tracer._finished.clear()  # as after a restart: only the saved copy is left
    service.runs.clear()
    res = await c.get(f"/v1/runs/{run_id}/trace", params={"t": token})
    assert res.status_code == 200 and res.json()["status"] == "failed"
    assert by_name(res.json())["safety"][0]["outputs"]["verdict"] == "unsafe"


async def test_an_expired_run_keeps_its_trace(api, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    c, deps, service = await api(trace_ui=True, enhancer=[missing()])
    run_id, token = await start(c, {"prompt": "make it nice", "plan": "pro"})
    await c.get(f"/v1/runs/{run_id}/events", params={"t": token})
    await service._expire(service.runs[run_id])
    saved = json.loads((tmp_path / "traces" / f"{run_id}.json").read_text())
    assert saved["status"] == "expired"
    assert any(s["name"] == "interrupt" for s in saved["spans"])


async def test_nothing_is_saved_when_the_panel_is_off(api, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    c, _, service = await api()
    run_id, token = await start(c)
    await c.get(f"/v1/runs/{run_id}/events", params={"t": token})
    await service.wait(run_id)
    assert not (tmp_path / "traces").exists()


async def test_the_trace_is_readable_while_the_run_is_still_going(api) -> None:  # type: ignore[no-untyped-def]
    import asyncio

    c, deps, service = await api(trace_ui=True)

    class SlowCritic:
        model = "slow"
        gate = asyncio.Event()

        async def complete_json(self, **kw: Any) -> Any:
            await self.gate.wait()
            return critic_ok(1)

    critic = SlowCritic()
    deps.critic = critic  # type: ignore[assignment]
    run_id, token = await start(c)
    await asyncio.sleep(0.3)  # the run is waiting for the critic
    mid = (await c.get(f"/v1/runs/{run_id}/trace", params={"t": token})).json()
    assert mid["status"] == "running"
    open_spans = [s["name"] for s in mid["spans"] if s["status"] == "running"]
    assert "critic" in open_spans  # the step in progress
    critic.gate.set()
    await service.wait(run_id)
    final = (await c.get(f"/v1/runs/{run_id}/trace", params={"t": token})).json()
    assert final["status"] == "done" and not [s for s in final["spans"] if s["status"] == "running"]
