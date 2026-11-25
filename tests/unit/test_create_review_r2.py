"""Regression tests for round 2 of the review (R2 to R8, R10 is frontend)."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from app.agents.image_grand.workers.brief import EnhancerOutput, Option, Question
from app.agents.image_grand.workers.prompt_enhancer import check_output
from app.api.deps import get_create_deps
from app.api.run_service import RunNotWaiting, RunService
from app.contracts.request import GenerationRequest, ImageSettings
from app.core.errors import ProviderCallFailed
from app.core.plans import Plan
from app.core.safe_text import clean_ui_text
from app.core.tracing import Span
from app.gateway.budget import SpendGuard
from app.gateway.providers.openrouter_images import OpenRouterImages
from app.main import app
from tests.unit.create_helpers import critic_ok, make_deps, make_service, missing, ready
from tests.unit.test_create_review_fixes import or_args, or_reply

OR = "https://openrouter.ai/api/v1/images"


def req(prompt: str = "iced coffee ad", **kw: Any) -> GenerationRequest:
    plan = kw.pop("plan", Plan.BASIC)
    return GenerationRequest(prompt=prompt, plan=plan, settings=ImageSettings(**kw))


async def run(service: Any, run_id: str, r: GenerationRequest) -> list[Any]:
    service.start(run_id, "u1", r)
    return [e async for e in service.stream(run_id)]


async def paused(tmp_path: Path) -> tuple[Any, RunService]:
    deps = make_deps(tmp_path, enhancer=[missing(), ready()], critic=[critic_ok(1)])
    service = make_service(deps)
    assert (await run(service, "p1", req("make it nice", n=1)))[-1].type == "question"
    return deps, service


# ---- R2: everyday words are not hidden when the user wrote them -----------------------------


def test_a_llama_in_the_andes_is_allowed_when_the_user_wrote_it() -> None:
    prompt = "a llama in the Andes at a food critic's table, a phoenix rising"
    assert (
        clean_ui_text("Llama close-up, phoenix glow", user_text=prompt)
        == "Llama close-up, phoenix glow"
    )
    assert clean_ui_text("Llama close-up", user_text="a cat") is None  # the model brought it in
    assert clean_ui_text("Made with FLUX", user_text="a flux capacitor") == "Made with FLUX"


def test_provider_and_pipeline_names_are_hidden_even_if_the_user_wrote_them() -> None:
    for name in ("cloudflare", "openrouter", "langsmith", "enhancer", "guardrail", "jev", "sdxl"):
        assert clean_ui_text(f"about {name}", user_text=f"draw {name}") is None


def test_a_question_about_a_llama_is_kept_when_the_prompt_has_a_llama() -> None:
    def q(text: str) -> Question:
        opts = [Option(id=i, label=f"{i} llama") for i in "abc"]
        return Question(id="q1", text=text, options=[*opts, Option(id="other", label="x")])

    out = EnhancerOutput(status="missing", questions=[q("Which llama?")])
    kept = check_output(out, prompt="a llama in the Andes", allow_questions=True)
    assert kept.status == "missing" and kept.questions[0].text == "Which llama?"
    dropped = check_output(out, prompt="make it nice", allow_questions=True)
    assert dropped.status == "ready"  # no user llama: the model's llama question is dropped


async def test_the_critic_highlight_keeps_a_word_the_user_wrote(tmp_path: Path) -> None:
    out = critic_ok(2, best=2).model_copy(update={"highlight": "sharp llama fur"})
    deps = make_deps(tmp_path, critic=[out])
    events = await run(make_service(deps), "r2", req("a llama in the Andes", n=2))
    assert events[-1].recommendation == "Image 2 matches your prompt best: sharp llama fur."


# ---- R3: a timeout may still be billed --------------------------------------------------------


@respx.mock
@pytest.mark.parametrize("failure", ["timeout", "5xx", "transport"])
async def test_a_timeout_or_5xx_keeps_the_estimate_against_the_cap(failure: str) -> None:
    if failure == "timeout":
        respx.post(OR).mock(side_effect=httpx.ReadTimeout("slow"))
    elif failure == "5xx":
        respx.post(OR).mock(return_value=httpx.Response(503))
    else:
        respx.post(OR).mock(side_effect=httpx.ReadError("dropped"))
    guard = SpendGuard(5)
    async with httpx.AsyncClient() as http:
        p = OpenRouterImages(
            http, api_key="k", base_url="https://openrouter.ai/api/v1", guard=guard
        )
        with pytest.raises(ProviderCallFailed):
            await p.generate(**or_args(1.0))
    assert guard.spent_usd == pytest.approx(1.0) and guard.reserved_usd == 0


@respx.mock
async def test_a_refused_request_is_not_counted() -> None:
    respx.post(OR).mock(return_value=httpx.Response(400))
    guard = SpendGuard(5)
    async with httpx.AsyncClient() as http:
        p = OpenRouterImages(
            http, api_key="k", base_url="https://openrouter.ai/api/v1", guard=guard
        )
        with pytest.raises(ProviderCallFailed):
            await p.generate(**or_args(1.0))
    assert guard.spent_usd == 0 and guard.reserved_usd == 0
    respx.post(OR).mock(return_value=or_reply(0.5))


# ---- R4: a crash after a question leaves nothing resumable ----------------------------------


async def test_a_crash_after_the_interrupt_cannot_be_resumed_for_free(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, enhancer=[missing(), ready()], critic=[critic_ok(1)])
    service = make_service(deps)
    real = service._on_interrupt

    async def interrupt_then_crash(record: Any, interrupts: Any) -> None:
        await real(record, interrupts)
        raise RuntimeError("boom while draining")

    service._on_interrupt = interrupt_then_crash  # type: ignore[method-assign]
    await run(service, "c1", req("make it nice", n=1))  # the stream ends at the question
    await service.wait("c1")  # then the crash is handled
    record = service.runs["c1"]
    assert record.events[-1].type == "error" and record.events[-1].code == "internal_error"
    assert record.status == "failed"
    assert record.timer is not None and record.timer.cancelled()
    with pytest.raises(RunNotWaiting):
        await service.resume("c1", {"q1": "A"})
    assert deps.gateway._providers["cloudflare"].calls == []  # type: ignore[attr-defined]
    assert await deps.credits.balance("u1") == 100


async def test_a_crash_noticed_during_a_resume_stops_the_resume(tmp_path: Path) -> None:
    deps, service = await paused(tmp_path)
    record = service.runs["p1"]
    record.crashed = True  # the paused phase crashed while resume was starting
    with pytest.raises(RunNotWaiting) as err:
        await service.resume("p1", {"q1": "A"})
    assert err.value.status == "failed" and record.status == "failed"
    assert record.events[-1].code == "internal_error"
    assert deps.gateway._providers["cloudflare"].calls == []  # type: ignore[attr-defined]


# ---- R5: no stale question during a resume --------------------------------------------------


async def test_a_reader_connecting_during_resuming_never_sees_the_old_question(
    tmp_path: Path,
) -> None:
    deps, service = await paused(tmp_path)
    record = service.runs["p1"]
    old = record.task
    gate = asyncio.Event()

    async def slow_close() -> None:  # the paused phase takes a moment to close
        await gate.wait()

    record.task = asyncio.create_task(slow_close())
    resume = asyncio.create_task(service.resume("p1", {"q1": "A"}))
    await asyncio.sleep(0.05)
    assert record.status == "resuming" and record.events == []  # cleared in the same step
    reader = asyncio.create_task(_collect(service, "p1"))
    await asyncio.sleep(0.05)
    assert not reader.done()  # it waits for the new phase instead of replaying the question
    gate.set()
    await resume
    events = await reader
    assert "question" not in [e.type for e in events] and events[-1].type == "ready"
    assert old is not None


async def _collect(service: RunService, run_id: str) -> list[Any]:
    return [e async for e in service.stream(run_id)]


# ---- R6: an error right after the hold releases it ------------------------------------------


async def test_an_error_after_the_hold_releases_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = Span.log

    def boom(self: Span, **outputs: Any) -> None:
        if "held" in outputs:
            raise RuntimeError("something added after the hold")
        real(self, **outputs)

    monkeypatch.setattr(Span, "log", boom)
    deps = make_deps(tmp_path)
    events = await run(make_service(deps), "r6", req(n=1))
    assert events[-1].type == "error"
    assert await deps.credits.balance("u1") == 100  # the hold did not leak


# ---- R7: a phase with no result is a failure --------------------------------------------------


class StatusOnlyGraph:
    """A graph that ends after status events only: no `ready`, no `error`."""

    async def astream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
        yield ((), "custom", {"type": "status", "run_id": "s1", "status": "thinking"})

    async def aget_state(self, *a: Any, **k: Any) -> None:
        return None


async def test_a_phase_that_ends_without_ready_is_failed(tmp_path: Path) -> None:
    deps = make_deps(tmp_path)
    service = RunService(StatusOnlyGraph(), deps)  # type: ignore[arg-type]
    events = await run(service, "s1", req(n=1))
    assert events[-1].type == "error" and events[-1].code == "internal_error"
    assert service.runs["s1"].status == "failed"


# ---- R8: a bad result.json is a 404, never a 500 ---------------------------------------------


@pytest.mark.parametrize("content", [b"{not json", b'{"documents": [{"x": 1}]}', b"[1, 2]", b""])
async def test_a_corrupt_result_file_releases_nothing(tmp_path: Path, content: bytes) -> None:
    deps = make_deps(tmp_path)
    deps.store.write_bytes("bad1", "img_1.jpg", b"x")
    deps.store.write_bytes("bad1", "result.json", content)
    assert deps.store.released("bad1") == set()
    app.dependency_overrides[get_create_deps] = lambda: deps
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            assert (await c.get("/v1/files/bad1/img_1.jpg")).status_code == 404
    finally:
        app.dependency_overrides.clear()
