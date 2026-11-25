"""Regression tests for the review of the txt2img bundle (C1, H1-H3, M1-M6, L1-L8)."""

import asyncio
import base64
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from app.agents.image_grand.workers import image_gen
from app.api.deps import get_create_deps, get_run_service, get_token_signer
from app.api.run_service import RunNotWaiting
from app.contracts.request import GenerationRequest, ImageSettings
from app.core.config import prompts_dir
from app.core.errors import InvalidModelOutput, ProviderCallFailed, SpendCapReached
from app.core.plans import Plan
from app.core.safe_text import clean_ui_text
from app.core.user_errors import INTERNAL_MESSAGE, PROVIDER_DOWN_MESSAGE, to_user_error
from app.gateway.budget import SpendGuard
from app.gateway.compiler import guard_prompt, strip_weak_words, user_weak_words
from app.gateway.providers.openrouter_images import OpenRouterImages
from app.gateway.registry import load_registry
from app.gateway.router import classify_prompt
from app.guardrails.deps import get_guard_deps
from app.local.output_store import OutputStore
from app.main import app
from tests.unit.create_helpers import (
    FakeProvider,
    SeqVision,
    critic_ok,
    jpeg,
    make_deps,
    make_service,
    missing,
    ready,
)
from tests.unit.guard_helpers import FakeJev, jev_answers
from tests.unit.guard_helpers import make_deps as make_guard_deps


def req(prompt: str = "iced coffee ad", **kw: Any) -> GenerationRequest:
    plan = kw.pop("plan", Plan.BASIC)
    return GenerationRequest(prompt=prompt, plan=plan, settings=ImageSettings(**kw))


async def run(service: Any, run_id: str, r: GenerationRequest, user: str = "u1") -> list[Any]:
    service.start(run_id, user, r)
    return [e async for e in service.stream(run_id)]


# ---- C1: the file store and endpoint cannot leave the output folder -------------------------


def test_store_cannot_read_a_secret_in_the_parent_folder(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("CLOUDFLARE_API_TOKEN=secret")
    (tmp_path / "outputs").mkdir()
    store = OutputStore(tmp_path / "outputs")
    assert (tmp_path / ".env").read_text() == "CLOUDFLARE_API_TOKEN=secret"  # it is really there
    for run_id, name in [
        ("..", ".env"),
        (".", ".env"),
        ("..", "env"),
        ("a/..", ".env"),
        ("x", "../.env"),
    ]:
        assert store.read_bytes(run_id, name) is None, (run_id, name)
    with pytest.raises(ValueError):
        store.write_bytes("..", "x.jpg", b"x")


async def test_files_endpoint_cannot_reach_the_env_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("CLOUDFLARE_API_TOKEN=secret")
    deps = make_deps(tmp_path / "outputs")
    app.dependency_overrides[get_create_deps] = lambda: deps
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            for path in [
                "/v1/files/%2E%2E/.env",
                "/v1/files/../.env",
                "/v1/files/%2e%2e/.env",
                "/v1/files/..%2F/.env",
                "/v1/files/x/..%2F..%2F.env",
            ]:
                res = await c.get(path)
                assert res.status_code == 404, path
                assert "secret" not in res.text
    finally:
        app.dependency_overrides.clear()


async def test_only_released_images_can_be_fetched(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(1)])
    deps.store.write_bytes("run1", "img_1.jpg", jpeg())
    assert deps.store.read_released("run1", "img_1.jpg") is None  # saved, not released
    service = make_service(deps)
    await run(service, "run2", req(n=1))
    assert deps.store.read_released("run2", "img_1.jpg") is not None  # in the saved result
    assert deps.store.read_released("run2", "doc_1.json") is None  # never served


# ---- H1: nothing is shown or fetchable before the safety check ------------------------------


class BlockingCritic:
    model = "blocking-critic"

    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self.go = asyncio.Event()

    async def complete_json(self, **kw: Any) -> Any:
        self.reached.set()
        await self.go.wait()
        return critic_ok(1)


async def test_images_are_not_fetchable_until_the_critic_has_passed_them(tmp_path: Path) -> None:
    deps = make_deps(tmp_path)
    critic = BlockingCritic()
    deps.critic = critic  # type: ignore[assignment]
    service = make_service(deps)
    service.start("h1", "u1", req(n=1))
    await critic.reached.wait()
    assert deps.store.read_bytes("h1", "img_1.jpg") is not None  # it exists on disk...
    assert deps.store.read_released("h1", "img_1.jpg") is None  # ...but it is not released
    seen = [e.type for e in service.runs["h1"].events]
    assert "ready" not in seen and "image" not in seen
    critic.go.set()
    events = [e async for e in service.stream("h1")]
    assert events[-1].type == "ready"
    assert deps.store.read_released("h1", "img_1.jpg") is not None


async def test_a_failed_run_deletes_every_file(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, vision=SeqVision(["unsafe\nS12"]))  # every image is unsafe
    service = make_service(deps)
    events = await run(service, "h2", req(n=2))
    assert events[-1].type == "error"
    assert not (tmp_path / "h2").exists()  # nothing of the run is left to fetch
    assert deps.store.read_released("h2", "img_1.jpg") is None
    assert await deps.credits.balance("u1") == 100


# ---- H2: only user-safe text reaches the client ---------------------------------------------

LEAKS = (
    "cloudflare",
    "provider returned",
    "wrong format",
    "500",
    "openrouter",
    "thumbnail",
    "unsupported image",
    "api.",
    "gpt-oss",
    "llama",
    "traceback",
)
ALLOWED_CODES = {
    "plan_limit",
    "out_of_credits",
    "content_refused",
    "output_blocked",
    "provider_down",
    "draft_expired",
    "internal_error",
    "not_available",
    "bad_input",
}


def assert_clean(event: Any) -> None:
    assert event.type == "error" and event.code in ALLOWED_CODES, event
    text = f"{event.code} {event.message}".lower()
    for leak in LEAKS:
        assert leak not in text, (leak, text)


@pytest.mark.parametrize(
    "where",
    ["enhancer_provider", "enhancer_format", "safety", "all_images", "settle_crash"],
)
async def test_failures_never_leak_internal_text(tmp_path: Path, where: str) -> None:
    kw: dict[str, Any] = {}
    if where == "enhancer_provider":
        kw["enhancer"] = [
            ProviderCallFailed("provider returned 500 from api.cloudflare.com", retryable=True)
        ]
    elif where == "enhancer_format":
        kw["enhancer"] = [InvalidModelOutput("the model answered in the wrong format")]
    elif where == "safety":
        kw["vision"] = SeqVision(
            [ProviderCallFailed("cloudflare reported failure", retryable=False)]
        )
    provider = FakeProvider()
    if where == "all_images":

        async def down(**_: Any) -> Any:
            raise ProviderCallFailed("openrouter key is not set", retryable=False)

        provider.generate = down  # type: ignore[method-assign]
    deps = make_deps(tmp_path, provider=provider, **kw)
    if where == "settle_crash":

        def crash(*a: Any, **k: Any) -> None:
            if a[1] == "result.json":
                raise OSError("disk path /secret/location unsupported image type: GIF")

        deps.store.write_json = crash  # type: ignore[method-assign]
    service = make_service(deps)
    events = await run(service, "e1", req(n=1))
    assert_clean(events[-1])
    assert await deps.credits.balance("u1") == 100


def test_to_user_error_trusts_only_our_own_messages() -> None:
    assert to_user_error(ProviderCallFailed("provider returned 500", retryable=True)) == {
        "code": "provider_down",
        "message": PROVIDER_DOWN_MESSAGE,
    }
    assert to_user_error(InvalidModelOutput("garbled"))["message"] == INTERNAL_MESSAGE
    assert to_user_error(RuntimeError("x"))["code"] == "internal_error"
    from app.core.errors import PlanLimit

    assert to_user_error(PlanLimit("Up to 2 images."))["message"] == "Up to 2 images."


# ---- H3: resume and expiry cannot race ------------------------------------------------------


async def paused(tmp_path: Path, **kw: Any) -> tuple[Any, Any]:
    deps = make_deps(tmp_path, enhancer=[missing(), ready()], critic=[critic_ok(1)], **kw)
    service = make_service(deps)
    first = await run(service, "p1", req("make it nice", n=1))
    assert first[-1].type == "question"
    return deps, service


async def test_two_concurrent_resumes_run_the_graph_once(tmp_path: Path) -> None:
    deps, service = await paused(tmp_path)
    results = await asyncio.gather(
        service.resume("p1", {"q1": "A"}), service.resume("p1", {"q1": "A"}), return_exceptions=True
    )
    assert sum(isinstance(r, RunNotWaiting) for r in results) == 1
    events = [e async for e in service.stream("p1")]
    assert events[-1].type == "ready"
    assert len(deps.enhancer.calls) == 2  # type: ignore[attr-defined]  # asked once, resumed once
    assert len(deps.gateway._providers["cloudflare"].calls) == 1  # type: ignore[attr-defined]
    assert await deps.credits.balance("u1") == 99


async def test_resume_after_expiry_is_refused_and_nothing_is_generated(tmp_path: Path) -> None:
    deps, service = await paused(tmp_path)
    await service._expire(service.runs["p1"])
    with pytest.raises(RunNotWaiting) as err:
        await service.resume("p1", {"q1": "A"})
    assert err.value.status == "expired"
    assert deps.gateway._providers["cloudflare"].calls == []  # type: ignore[attr-defined]
    assert await deps.credits.balance("u1") == 100  # the hold was released


async def test_expiry_after_a_resume_started_does_nothing(tmp_path: Path) -> None:
    deps, service = await paused(tmp_path)
    await service.resume("p1", {"q1": "A"})
    await service._expire(service.runs["p1"])  # the timer firing late
    events = [e async for e in service.stream("p1")]
    assert events[-1].type == "ready" and service.runs["p1"].status == "done"
    assert await deps.credits.balance("u1") == 99  # the run kept its hold and was charged


async def test_expiry_during_the_guardrail_call_returns_410_and_charges_nothing(
    tmp_path: Path,
) -> None:
    deps = make_deps(tmp_path, enhancer=[missing(), ready()], critic=[critic_ok(1)])
    service = make_service(deps)
    guard = make_guard_deps()
    app.dependency_overrides[get_create_deps] = lambda: deps
    app.dependency_overrides[get_run_service] = lambda: service
    app.dependency_overrides[get_token_signer] = lambda: deps.tokens
    app.dependency_overrides[get_guard_deps] = lambda: guard
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            created = (await c.post("/v1/runs", json={"prompt": "make it nice"})).json()
            rid, token = created["run_id"], created["token"]
            await c.get(f"/v1/runs/{rid}/events", params={"t": token})
            guard.jev = FakeJev(jev_answers(), delay=0.3)  # type: ignore[attr-defined]
            body = {"answers": {"q1": "other", "q1_text": "a red fox"}}
            pending = asyncio.create_task(c.post(f"/v1/runs/{rid}/resume", json=body))
            await asyncio.sleep(0.1)  # the guardrail call is running
            await service._expire(service.runs[rid])
            res = await pending
            assert res.status_code == 410
            assert await deps.credits.balance("u1") == 100
            assert deps.gateway._providers["cloudflare"].calls == []  # type: ignore[attr-defined]
    finally:
        app.dependency_overrides.clear()


# ---- M1: no result cache ---------------------------------------------------------------------


async def test_pressing_create_again_gives_new_seeds(tmp_path: Path) -> None:
    provider = FakeProvider()
    deps = make_deps(tmp_path, provider=provider, critic=[critic_ok(1)])
    service = make_service(deps)
    await run(service, "m1a", req(model="phoenix-1", plan=Plan.PRO, n=1))
    await run(service, "m1b", req(model="phoenix-1", plan=Plan.PRO, n=1))
    seeds = [c["seed"] for c in provider.calls]
    assert len(seeds) == 2 and seeds[0] != seeds[1]


# ---- M2: closing the stream ------------------------------------------------------------------


async def test_the_paused_phase_is_closed_before_resume_continues(tmp_path: Path) -> None:
    deps, service = await paused(tmp_path)
    record = service.runs["p1"]
    assert record.task is not None
    await service.resume("p1", {"q1": "A"})
    assert [e async for e in service.stream("p1")][-1].type == "ready"
    assert record.task.done()


# ---- M3: the spend cap holds with parallel calls --------------------------------------------

OR = "https://openrouter.ai/api/v1/images"


def or_reply(cost: float | None = 1.0) -> httpx.Response:
    b64 = base64.b64encode(b"PNG").decode()
    body: dict[str, Any] = {"data": [{"b64_json": b64, "media_type": "image/png"}]}
    if cost is not None:
        body["usage"] = {"cost": cost}
    return httpx.Response(200, json=body)


def or_args(estimate: float) -> dict[str, Any]:
    return dict(
        slug="s",
        family="gemini",
        prompt="x",
        negative_prompt="",
        width=1,
        height=1,
        aspect_ratio="1:1",
        seed=None,
        timeout=5,
        estimated_cost_usd=estimate,
    )


@respx.mock
async def test_four_parallel_calls_cannot_pass_a_cap_of_one_and_a_half_calls() -> None:
    route = respx.post(OR).mock(return_value=or_reply(1.0))
    guard = SpendGuard(1.5)
    async with httpx.AsyncClient() as http:
        p = OpenRouterImages(
            http, api_key="k", base_url="https://openrouter.ai/api/v1", guard=guard
        )
        results = await asyncio.gather(
            *(p.generate(**or_args(1.0)) for _ in range(4)), return_exceptions=True
        )
    assert route.call_count == 1
    assert sum(isinstance(r, SpendCapReached) for r in results) == 3
    assert guard.spent_usd == pytest.approx(1.0) and guard.reserved_usd == pytest.approx(0)


@respx.mock
async def test_a_missing_billed_cost_keeps_the_estimate_so_the_cap_still_trips() -> None:
    respx.post(OR).mock(return_value=or_reply(None))
    guard = SpendGuard(1.5)
    async with httpx.AsyncClient() as http:
        p = OpenRouterImages(
            http, api_key="k", base_url="https://openrouter.ai/api/v1", guard=guard
        )
        await p.generate(**or_args(1.0))
        assert guard.spent_usd == pytest.approx(1.0)  # the estimate, not zero
        with pytest.raises(SpendCapReached):
            await p.generate(**or_args(1.0))


@respx.mock
async def test_a_failed_call_releases_its_reservation() -> None:
    respx.post(OR).mock(return_value=httpx.Response(400))
    guard = SpendGuard(5)
    async with httpx.AsyncClient() as http:
        p = OpenRouterImages(
            http, api_key="k", base_url="https://openrouter.ai/api/v1", guard=guard
        )
        with pytest.raises(ProviderCallFailed):
            await p.generate(**or_args(1.0))
    assert guard.reserved_usd == 0 and guard.spent_usd == 0


# ---- M4: a failure after the work is done costs the user nothing ----------------------------


async def test_a_crash_while_saving_the_result_charges_nothing_and_leaves_no_files(
    tmp_path: Path,
) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(2)])
    real = deps.store.write_json

    def flaky(run_id: str, name: str, data: object) -> None:
        if name == "result.json":
            raise OSError("disk full")
        real(run_id, name, data)

    deps.store.write_json = flaky  # type: ignore[method-assign]
    service = make_service(deps)
    events = await run(service, "m4", req(n=2))
    assert events[-1].type == "error" and events[-1].code == "internal_error"
    assert await deps.credits.balance("u1") == 100
    assert not (tmp_path / "m4").exists()


# ---- M5: weak words the user typed stay ------------------------------------------------------


def test_minimal_stays_when_the_user_asked_for_it() -> None:
    card = load_registry().get("flux-2-klein-4b")
    assert card is not None
    out = guard_prompt(
        card,
        "minimal flat logo, serene calm",
        aspect_ratio="1:1",
        user_prompt="a minimal design logo",
    )
    assert "minimal" in out.prompt and "serene" not in out.prompt  # serene was the LLM's addition
    assert user_weak_words("A Minimal design") == {"minimal"}
    assert strip_weak_words("a minimal logo") == "a logo"  # nobody asked for it: stripped
    assert strip_weak_words("a minimal logo", frozenset({"minimal"})) == "a minimal logo"


# ---- M6: one broken variation does not kill the run -----------------------------------------


async def test_a_thumbnail_error_in_one_variation_still_delivers_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = image_gen.make_thumbnail
    calls = {"n": 0}

    def flaky(data: bytes, size: int = 256) -> bytes:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("cannot write thumbnail")
        return real(data, size)

    monkeypatch.setattr(image_gen, "make_thumbnail", flaky)
    deps = make_deps(tmp_path, critic=[critic_ok(2)])
    service = make_service(deps)
    events = await run(service, "m6", req(n=2))
    assert events[-1].type == "ready" and len(events[-1].images) == 1
    assert await deps.credits.balance("u1") == 99


# ---- L1: router false positives --------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("can you make a cat sitting on a sofa", "general"),
        ("a 3d model of a car", "photo"),
        ("people watch the sunset on a beach", "people"),
        ("a wooden box of apples", "general"),
        ("a sign of spring in a meadow", "general"),
    ],
)
def test_common_words_do_not_misroute(prompt: str, expected: str) -> None:
    assert classify_prompt(prompt) == expected


# ---- L2: the hold is the real model's cost ---------------------------------------------------


async def test_the_hold_uses_the_chosen_models_tier_not_a_worst_case(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, credits=1, critic=[critic_ok(1)])  # one credit only
    service = make_service(deps)
    # a Pro plan user picks a Basic model: 1 credit is enough, the Pro worst case (3) would block
    events = await run(service, "l2", req(plan=Plan.PRO, model="flux-2-klein-4b", n=1))
    assert events[-1].type == "ready"
    assert await deps.credits.balance("u1") == 0


async def test_credits_are_held_before_the_enhancer_runs(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, credits=0)
    service = make_service(deps)
    events = await run(service, "l2b", req(n=1))
    assert events[-1].code == "out_of_credits"
    assert deps.enhancer.calls == []  # type: ignore[attr-defined]  # no LLM cost without credits


# ---- L4: ownership ---------------------------------------------------------------------------


async def test_other_users_cannot_read_a_run(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(1)])
    service = make_service(deps)
    app.dependency_overrides[get_create_deps] = lambda: deps
    app.dependency_overrides[get_run_service] = lambda: service
    app.dependency_overrides[get_token_signer] = lambda: deps.tokens
    try:
        await run(service, "own1", req(n=1), user="alice")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            mine = {"X-User-Id": "alice"}
            assert (await c.get("/v1/runs/own1", headers=mine)).status_code == 200
            assert (await c.get("/v1/runs/own1", headers={"X-User-Id": "eve"})).status_code == 404
    finally:
        app.dependency_overrides.clear()


# ---- L6: memory ------------------------------------------------------------------------------


async def test_old_finished_runs_are_forgotten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.api import run_service

    monkeypatch.setattr(run_service, "MAX_KEPT_RUNS", 3)
    deps = make_deps(tmp_path, critic=[critic_ok(1)])
    service = make_service(deps)
    for i in range(5):
        await run(service, f"old{i}", req(prompt=f"a fox {i}", n=1))
        await service.wait(f"old{i}")
    assert len(service.runs) <= 4


async def test_an_expired_run_drops_its_trace(tmp_path: Path) -> None:
    from tests.unit.create_helpers import RecordingClient

    client = RecordingClient()
    deps, service = await paused(tmp_path, trace=client)
    await service._expire(service.runs["p1"])
    assert "p1" not in deps.tracer._runs


# ---- L7: the enhancer prompt no longer contradicts itself ------------------------------------


def test_enhancer_prompt_never_asks_about_a_settings_clash() -> None:
    text = (prompts_dir() / "create" / "txt2img" / "enhancer.md").read_text()
    assert "do NOT ask" in text and "settings win" in text
    assert "or which of two clashing" not in text
    spec = (
        Path(__file__).parents[2]
        / "execution-steps"
        / "Image Create Agent"
        / "TXT2IMG-AGENT-SPEC.md"
    ).read_text()
    assert "The prompt clashes with the settings (for example" not in spec


# ---- L8: model text shown to users is cleaned ------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A product on a table", "A product on a table"),
        ("Visit https://evil.example/x now", None),
        ("see evil.com", None),
        ("Made with FLUX Phoenix", None),
        ("Powered by Cloudflare", None),
        ("**Bold** [link](x)", "Bold link (x)"),
        ("   ", None),
    ],
)
def test_clean_ui_text(text: str, expected: str | None) -> None:
    assert clean_ui_text(text) == expected


def test_clean_ui_text_caps_the_length() -> None:
    assert len(clean_ui_text("word " * 100, 40) or "") <= 40


def test_questions_with_a_model_name_are_dropped_and_options_cleaned() -> None:
    from app.agents.image_grand.workers.brief import EnhancerOutput, Option, Question
    from app.agents.image_grand.workers.prompt_enhancer import check_output

    def q(text: str, labels: list[str]) -> Question:
        opts = [Option(id=x, label=lab) for x, lab in zip("abc", labels, strict=False)]
        return Question(
            id="q1", text=text, options=[*opts, Option(id="other", label="Something else")]
        )

    bad = check_output(
        EnhancerOutput(status="missing", questions=[q("Use FLUX?", ["a", "b", "c"])]),
        prompt="p",
        allow_questions=True,
    )
    assert bad.status == "ready"  # the only question was dropped, so the run goes on
    good = check_output(
        EnhancerOutput(
            status="missing", questions=[q("What?", ["A cat", "See http://x.io", "A dog"])]
        ),
        prompt="p",
        allow_questions=True,
    )
    assert [o.label for o in good.questions[0].options] == ["A cat", "A dog", "Something else"]


async def test_the_critic_highlight_is_cleaned(tmp_path: Path) -> None:
    out = critic_ok(2, best=2).model_copy(update={"highlight": "made by FLUX at http://x.io"})
    deps = make_deps(tmp_path, critic=[out])
    service = make_service(deps)
    events = await run(service, "l8", req(plan=Plan.BASIC, n=2))
    assert events[-1].recommendation == "Image 2 matches your prompt best."
