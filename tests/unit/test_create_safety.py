"""B2: every image is safety-checked on its own, before anything is released."""

import asyncio
import base64
from pathlib import Path
from typing import Any

import pytest

from app.agents.image_grand.workers.safety import Safety, check_image
from app.contracts.request import GenerationRequest, ImageSettings
from app.core.errors import ProviderCallFailed
from app.core.plans import Plan
from tests.unit.create_helpers import (
    FakeLLM,
    SeqVision,
    critic_ok,
    jpeg,
    make_deps,
    make_service,
)


def req(prompt: str = "a red fox", **kw: Any) -> GenerationRequest:
    plan = kw.pop("plan", Plan.BASIC)
    return GenerationRequest(prompt=prompt, plan=plan, settings=ImageSettings(**kw))


async def run(service: Any, run_id: str, r: GenerationRequest) -> list[Any]:
    service.start(run_id, "u1", r)
    return [e async for e in service.stream(run_id)]


# ---- the check itself ------------------------------------------------------------------------


async def test_safe_and_unsafe_answers() -> None:
    assert await check_image(SeqVision(["safe"]), jpeg(), "p", timeout=5) is Safety.SAFE
    assert await check_image(SeqVision(["unsafe\nS12"]), jpeg(), "p", timeout=5) is Safety.UNSAFE


@pytest.mark.parametrize(
    "answer",
    [ProviderCallFailed("timeout", retryable=False), RuntimeError("x"), "maybe", ""],
)
async def test_a_check_that_fails_is_never_a_pass(answer: Any) -> None:
    assert await check_image(SeqVision([answer]), jpeg(), "p", timeout=5) is Safety.FAILED


async def test_a_timeout_is_a_failed_check() -> None:
    class Slow:
        model = "slow"

        async def classify(self, **_: Any) -> Any:
            await asyncio.sleep(5)

    assert await check_image(Slow(), jpeg(), "p", timeout=0.05) is Safety.FAILED  # type: ignore[arg-type]


async def test_a_retryable_error_is_retried_once() -> None:
    v = SeqVision([ProviderCallFailed("503", retryable=True), "safe"])
    assert await check_image(v, jpeg(), "p", timeout=5) is Safety.SAFE
    assert len(v.calls) == 2


async def test_the_check_sees_the_thumbnail_and_the_final_prompt() -> None:
    v = SeqVision(["safe"])
    await check_image(v, b"\xff\xd8thumb", "a red fox", timeout=5)
    call = v.calls[0]
    assert call["text"] == "a red fox"
    assert call["image_urls"] == [
        "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8thumb").decode()
    ]


# ---- in the run --------------------------------------------------------------------------------


async def test_every_generated_image_is_checked_once(tmp_path: Path) -> None:
    vision = SeqVision(["safe"])
    deps = make_deps(tmp_path, critic=[critic_ok(4)], vision=vision)
    events = await run(make_service(deps), "s1", req(n=2, plan=Plan.BASIC))
    assert events[-1].type == "ready" and len(vision.calls) == 2


async def test_an_unsafe_image_is_deleted_never_sent_and_never_charged(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(2)], vision=SeqVision(["unsafe\nS12", "safe"]))
    events = await run(make_service(deps), "s2", req(n=2, plan=Plan.BASIC))
    ready = events[-1]
    assert ready.type == "ready" and len(ready.images) == 1
    on_disk = sorted(p.name for p in (tmp_path / "s2").glob("img_*"))
    assert on_disk == [ready.images[0].url.split("/")[-1].split("?")[0]]
    assert await deps.credits.balance("u1") == 99


async def test_a_safety_timeout_drops_that_image_only(tmp_path: Path) -> None:
    vision = SeqVision([ProviderCallFailed("timed out", retryable=False), "safe"])
    deps = make_deps(tmp_path, critic=[critic_ok(2)], vision=vision)
    events = await run(make_service(deps), "s3", req(n=2, plan=Plan.BASIC))
    assert events[-1].type == "ready" and len(events[-1].images) == 1
    assert await deps.credits.balance("u1") == 99


async def test_if_every_check_fails_nothing_is_delivered_and_nothing_charged(
    tmp_path: Path,
) -> None:
    vision = SeqVision([ProviderCallFailed("down", retryable=False)])
    deps = make_deps(tmp_path, vision=vision)
    events = await run(make_service(deps), "s4", req(n=2, plan=Plan.BASIC))
    assert events[-1].type == "error" and events[-1].code == "provider_down"
    assert await deps.credits.balance("u1") == 100
    assert not (tmp_path / "s4").exists()


async def test_all_unsafe_is_output_blocked_not_provider_down(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, vision=SeqVision(["unsafe\nS4"]))
    events = await run(make_service(deps), "s5", req(n=2, plan=Plan.BASIC))
    assert events[-1].code == "output_blocked"
    assert await deps.credits.balance("u1") == 100


async def test_critic_failure_still_delivers_in_generation_order_without_a_line(
    tmp_path: Path,
) -> None:
    boom = ProviderCallFailed("down", retryable=False)
    deps = make_deps(tmp_path, critic=[boom])  # the critic fails twice
    events = await run(make_service(deps), "s6", req(n=3, plan=Plan.PRO))
    ready = events[-1]
    assert ready.type == "ready"
    assert [i.id for i in ready.images] == ["img_1", "img_2", "img_3"]  # not ranked
    assert ready.recommendation is None
    assert await deps.credits.balance("u1") == 100 - 9  # charged for the 3 delivered images
    assert len(deps.critic.calls if hasattr(deps.critic, "calls") else []) == 2  # one retry


async def test_the_critic_is_not_asked_about_safety(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(1)])
    await run(make_service(deps), "s7", req(n=1))
    critic: FakeLLM = deps.critic  # type: ignore[assignment]
    assert (
        "safe"
        not in critic.calls[0]["schema"].model_fields["images"].annotation.__args__[0].model_fields
    )


class BlockingVision:
    model = "blocking"

    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self.go = asyncio.Event()

    async def classify(self, **_: Any) -> Any:
        from app.contracts.ports import VisionRaw

        self.reached.set()
        await self.go.wait()
        return VisionRaw("safe", [("safe", -0.01), ("unsafe", -5.0)])


async def test_nothing_unchecked_is_ever_released(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(1)])
    vision = BlockingVision()
    deps.vision = vision  # type: ignore[assignment]
    service = make_service(deps)
    service.start("s8", "u1", req(n=1))
    await vision.reached.wait()
    assert deps.store.read_bytes("s8", "img_1.jpg") is not None  # made, still being checked
    assert deps.store.read_released("s8", "img_1.jpg") is None  # but not released
    assert all(e.type not in ("ready",) for e in service.runs["s8"].events)
    vision.go.set()
    events = [e async for e in service.stream("s8")]
    assert events[-1].type == "ready"
    assert deps.store.read_released("s8", "img_1.jpg") is not None
