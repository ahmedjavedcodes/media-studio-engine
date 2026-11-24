"""Regression tests for round 3 of the review (A1 to A5)."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from app.api.run_service import RunNotWaiting, RunService
from app.contracts.request import GenerationRequest, ImageSettings
from app.core.errors import ProviderCallFailed, SpendCapReached
from app.core.plans import Plan
from app.core.safe_text import clean_ui_text
from app.gateway.image_gateway import BREAKER_PAUSE_S, ImageGateway
from app.gateway.registry import ModelCard, ProviderEntry
from tests.unit.create_helpers import (
    FakeProvider,
    critic_ok,
    make_deps,
    make_service,
    missing,
    ready,
)


def req(prompt: str = "iced coffee ad", **kw: Any) -> GenerationRequest:
    plan = kw.pop("plan", Plan.BASIC)
    return GenerationRequest(prompt=prompt, plan=plan, settings=ImageSettings(**kw))


async def run(service: Any, run_id: str, r: GenerationRequest) -> list[Any]:
    service.start(run_id, "u1", r)
    return [e async for e in service.stream(run_id)]


# ---- A1 + A5: the real interleaving of a crash and a resume ----------------------------------


async def test_a_crash_while_resume_waits_pushes_one_error_and_generates_nothing(
    tmp_path: Path,
) -> None:
    deps = make_deps(tmp_path, enhancer=[missing(), ready()], critic=[critic_ok(1)])
    service = make_service(deps)
    gate = asyncio.Event()
    real = service._on_interrupt

    async def interrupt_wait_then_crash(record: Any, interrupts: Any) -> None:
        await real(record, interrupts)  # the real code: the question, `waiting`, the timer
        await gate.wait()  # the driver is still running while resume() arrives
        raise RuntimeError("boom while draining")

    service._on_interrupt = interrupt_wait_then_crash  # type: ignore[method-assign]
    await run(service, "i1", req("make it nice", n=1))  # ends at the question
    record = service.runs["i1"]
    assert record.status == "waiting"

    resume = asyncio.create_task(service.resume("i1", {"q1": "A"}))
    await asyncio.sleep(0.05)
    assert record.status == "resuming"  # resume got in first and is waiting for the old task
    gate.set()  # now the old driver crashes while resume is waiting for it
    with pytest.raises(RunNotWaiting) as err:
        await resume
    assert err.value.status == "failed"
    errors = [e for e in record.events if e.type == "error"]
    assert len(errors) == 1 and errors[0].code == "internal_error"  # one error, not two
    assert record.status == "failed"
    assert deps.gateway._providers["cloudflare"].calls == []  # type: ignore[attr-defined]
    assert await deps.credits.balance("u1") == 100  # the hold came back
    assert not (tmp_path / "i1").exists()


class HoldThenStatusOnly:
    """A graph that makes a real hold, reports it, then ends with status events only."""

    def __init__(self, deps: Any) -> None:
        self._deps = deps

    async def astream(self, *a: Any, **k: Any) -> AsyncIterator[Any]:
        hold = await self._deps.credits.hold("u1", "h1", 5)
        self._deps.store.write_bytes("h1", "img_1.jpg", b"x")
        yield ((), "updates", {"router": {"hold_id": hold}})
        yield ((), "custom", {"type": "status", "run_id": "h1", "status": "thinking"})

    async def aget_state(self, *a: Any, **k: Any) -> None:
        return None


async def test_a_phase_with_no_result_returns_the_hold_and_purges_the_folder(
    tmp_path: Path,
) -> None:
    deps = make_deps(tmp_path)
    service = RunService(HoldThenStatusOnly(deps), deps)  # type: ignore[arg-type]
    events = await run(service, "h1", req(n=1))
    assert events[-1].type == "error" and service.runs["h1"].status == "failed"
    assert await deps.credits.balance("u1") == 100  # the 5 held credits came back
    assert not (tmp_path / "h1").exists()  # and the files are gone


# ---- A2: a cancelled resume does not get stuck in `resuming` -------------------------------


async def test_a_cancelled_resume_ends_the_run_cleanly(tmp_path: Path) -> None:
    from tests.unit.create_helpers import RecordingClient

    client = RecordingClient()
    deps = make_deps(tmp_path, enhancer=[missing(), ready()], critic=[critic_ok(1)], trace=client)
    service = make_service(deps)
    await run(service, "a2", req("make it nice", n=1))
    record = service.runs["a2"]
    stuck = asyncio.create_task(asyncio.sleep(30))  # stands in for the paused phase
    record.task = stuck
    stuck.cancel()
    with pytest.raises(asyncio.CancelledError):
        await service.resume("a2", {"q1": "A"})
    assert record.status == "failed"  # not stuck in `resuming`
    assert await deps.credits.balance("u1") == 100
    assert "a2" not in deps.tracer._runs  # the trace was closed and dropped


# ---- A3: company names are hidden too ---------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["Black Forest Labs", "ByteDance", "Google", "Workers AI", "Meta", "Alibaba", "Tencent"],
)
def test_company_names_are_always_hidden(name: str) -> None:
    assert clean_ui_text(f"Powered by {name}") is None
    assert clean_ui_text(f"Powered by {name}", user_text=f"draw {name}") is None


# ---- A4: spend cap fallback and the circuit breaker -----------------------------------------


def card(*providers: tuple[str, float]) -> ModelCard:
    return ModelCard(
        id="m",
        name="M",
        family="flux",
        status="live",
        providers=[ProviderEntry(name=n, slug=n, cost=c) for n, c in providers],
    )


def failing(exc: Exception) -> FakeProvider:
    p = FakeProvider()

    async def boom(**_: Any) -> Any:
        raise exc

    p.generate = boom  # type: ignore[method-assign]
    return p


async def gen(gw: ImageGateway, c: ModelCard) -> Any:
    return await gw.generate(
        c, prompt="p", negative_prompt="", width=8, height=8, aspect_ratio="1:1", seed=1
    )


async def test_the_spend_cap_falls_back_to_a_free_provider() -> None:
    paid, free = failing(SpendCapReached()), FakeProvider()
    gw = ImageGateway(
        {"openrouter": paid, "cloudflare": free}, ["openrouter", "cloudflare"], paid={"openrouter"}
    )
    res = await gen(gw, card(("openrouter", 0.01), ("cloudflare", 0.02)))
    assert res.provider == "cloudflare" and res.tried == ["openrouter", "cloudflare"]


async def test_the_spend_cap_never_moves_to_another_paid_provider() -> None:
    a, b = failing(SpendCapReached()), FakeProvider()
    gw = ImageGateway(
        {"openrouter": a, "fal": b}, ["openrouter", "fal"], paid={"openrouter", "fal"}
    )
    with pytest.raises(SpendCapReached):
        await gen(gw, card(("openrouter", 0.01), ("fal", 0.02)))
    assert b.calls == []


async def test_the_spend_cap_with_no_free_provider_raises() -> None:
    gw = ImageGateway(
        {"openrouter": failing(SpendCapReached())}, ["openrouter"], paid={"openrouter"}
    )
    with pytest.raises(SpendCapReached):
        await gen(gw, card(("openrouter", 0.01)))


async def test_three_failures_in_a_row_pause_a_paid_provider_for_a_minute() -> None:
    now = [1000.0]
    paid, free = failing(ProviderCallFailed("503", retryable=True)), FakeProvider()
    gw = ImageGateway(
        {"openrouter": paid, "cloudflare": free},
        ["openrouter", "cloudflare"],
        paid={"openrouter"},
        clock=lambda: now[0],
    )
    c = card(("openrouter", 0.01), ("cloudflare", 0.02))
    calls = {"n": 0}
    inner = paid.generate

    async def counted(**kw: Any) -> Any:
        calls["n"] += 1
        return await inner(**kw)

    paid.generate = counted  # type: ignore[method-assign]
    for _ in range(3):
        assert (await gen(gw, c)).provider == "cloudflare"
    assert calls["n"] == 3
    for _ in range(3):  # the breaker is open: OpenRouter is not even tried
        res = await gen(gw, c)
        assert res.tried == ["cloudflare"]
    assert calls["n"] == 3
    now[0] += BREAKER_PAUSE_S + 1  # a minute later it is tried again
    await gen(gw, c)
    assert calls["n"] == 4


async def test_a_success_resets_the_failure_count_and_free_providers_are_never_paused() -> None:
    now = [0.0]
    flaky = FakeProvider()
    seq = iter([True, True, False, True, True])

    inner = flaky.generate

    async def sometimes(**kw: Any) -> Any:
        if next(seq):
            raise ProviderCallFailed("503", retryable=True)
        return await inner(**kw)

    flaky.generate = sometimes  # type: ignore[method-assign]
    gw = ImageGateway(
        {"openrouter": flaky, "cloudflare": FakeProvider()},
        ["openrouter", "cloudflare"],
        paid={"openrouter"},
        clock=lambda: now[0],
    )
    c = card(("openrouter", 0.01), ("cloudflare", 0.02))
    results = [(await gen(gw, c)).provider for _ in range(5)]
    assert results == ["cloudflare", "cloudflare", "openrouter", "cloudflare", "cloudflare"]
    assert gw._open_until == {}  # never 3 in a row

    down = failing(ProviderCallFailed("503", retryable=True))
    only_free = ImageGateway({"cloudflare": down}, ["cloudflare"], paid=set())
    for _ in range(5):
        with pytest.raises(ProviderCallFailed):
            await gen(only_free, card(("cloudflare", 0.01)))
    assert only_free._open_until == {}  # a free provider is never switched off
