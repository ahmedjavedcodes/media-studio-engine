"""M5: Image Creator. No LLM; formats a request, calls the model gateway, stores the image,
returns the id. Never puts bytes in the envelope. All image calls are faked (ScriptedImages)."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import pytest

from media_engine.agents.base import AgentConfig
from media_engine.contracts import AuthContext, Goal, Status
from media_engine.gateway.errors import (
    AllProvidersFailed,
    BadRequestError,
    ContentRefused,
    DeadlineExceeded,
    ProviderError,
)
from media_engine.gateway.llm_gateway import chain_resolver
from media_engine.gateway.model_gateway import ModelGateway
from media_engine.ports import ListEventSink, LocalArtifactStore
from media_engine.schemas import GenerationResult
from media_engine.services.create.loader import load_create_config
from media_engine.tools.image_gen_model import ImageGenModel
from tests.unit.harness.fakes import ScriptedImages, image_bytes
from tests.unit.waiting import wait_for_event

AUTH = AuthContext(plan="free", execution_mode="auto", model_demand="balanced")
GOAL = Goal(task="generate", instructions="Create the image.")
CFG = load_create_config()
STOP = CFG.policy.stopping_condition("generate", "free", "one stored image")
PROMPT = "A cozy coffee shop interior at dusk, warm lamps, wooden tables."


def creator(backend: ScriptedImages, artifacts: Any) -> ImageGenModel:
    config = AgentConfig.from_policy("image_creator", CFG.agent("image_creator"), None)
    images = ModelGateway(backend, chain_resolver(CFG.agents))
    return ImageGenModel(config, ListEventSink(), images=images, artifacts=artifacts)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / ".artifacts")


async def run(backend: ScriptedImages, artifacts: Any, **overrides: Any):  # type: ignore[no-untyped-def]
    inputs: dict[str, Any] = {"prompt": PROMPT, "width": 1024, "height": 1024, "seed": 42}
    inputs.update(overrides)
    return await creator(backend, artifacts).run(GOAL, STOP, AUTH, inputs)


# -- happy path ---------------------------------------------------------------------------------


async def test_generates_stores_and_returns_only_the_id(store):
    png = image_bytes(1024, 1024)
    backend = ScriptedImages([png])

    env = await run(backend, store)

    assert env.state.status is Status.DONE
    result = GenerationResult.model_validate(env.output)
    assert await store.get(result.image_id) == png  # the bytes live in the store...
    # ...and the envelope holds only the GenerationResult scalars, no encoding of them.
    assert set(env.output) == set(GenerationResult.model_fields)
    scalars = {k: v for k, v in env.output.items() if k != "thumbnails"}
    assert all(v is None or isinstance(v, str | int | float | bool) for v in scalars.values())
    assert all(isinstance(size, int) for size in env.output["thumbnails"])  # sizes, not bytes
    assert base64.b64encode(png).decode()[:64] not in env.model_dump_json()
    assert (result.provider, result.model, result.seed) == (
        "cloudflare",
        "@cf/black-forest-labs/flux-1-schnell",
        42,
    )
    assert (result.width, result.height, result.mime) == (1024, 1024, "image/png")
    assert result.seed_honoured is False  # flux-1-schnell takes no seed (verified 2026-11-19)
    assert env.state.spend.neurons == pytest.approx(57.6)
    assert env.state.spend.llm_calls == 0  # no LLM, ever


async def test_request_carries_prompt_size_seed_tier_chain_and_deadline(store):
    backend = ScriptedImages([image_bytes()])

    await run(backend, store, width=1024, height=576, seed=7)

    task, request, deadline_s = backend.requests[0]
    assert task == "image_create"  # agents.yaml image_creator -> chain
    assert (request.prompt, request.width, request.height, request.seed) == (PROMPT, 1024, 576, 7)
    assert deadline_s == STOP.deadline_ms / 1000


async def test_reports_the_size_that_came_back_not_the_one_asked_for(store):
    # flux-1-schnell is sent no width/height; the supervisor must see the real size.
    backend = ScriptedImages([image_bytes(1024, 1024, fmt="JPEG")])

    env = await run(backend, store, width=1024, height=576)

    assert (env.output["width"], env.output["height"]) == (1024, 1024)
    assert env.output["mime"] == "image/jpeg"


def test_has_no_llm_and_refuses_a_prompt_or_output_model(store):
    agent = creator(ScriptedImages([]), store)
    assert agent._gateway is None
    with pytest.raises(ValueError):
        ImageGenModel(
            AgentConfig.from_policy("critic", CFG.agent("critic"), GenerationResult),
            ListEventSink(),
            images=ModelGateway(ScriptedImages([]), chain_resolver(CFG.agents)),
            artifacts=store,
        )


# -- cancellation is the supervisor's job -------------------------------------------------------


async def test_an_is_cancelled_input_is_ignored_the_supervisor_guard_owns_cancel(store):
    backend = ScriptedImages([image_bytes()])

    env = await run(backend, store, is_cancelled=lambda: True)  # the old hook: gone

    assert env.state.status is Status.DONE and len(backend.requests) == 1


# -- failures are typed for the supervisor --------------------------------------------------------


def _pe(**kw: Any) -> ProviderError:
    return ProviderError("cloudflare", "m", "x", **kw)


@pytest.mark.parametrize(
    ("error", "failure"),
    [
        (ContentRefused(_pe(status=400, code=3030), "image_create"), "content_refused"),
        (AllProvidersFailed("image_create", []), "no_provider"),
        (DeadlineExceeded("image_create", 60.0, []), "deadline"),
        (BadRequestError(_pe(status=400)), "bad_request"),
    ],
)
async def test_gateway_failures_become_typed_failed_envelopes(store, error, failure):
    env = await run(ScriptedImages([error]), store)

    assert env.state.status is Status.FAILED
    assert env.output == {"failure": failure}
    assert await store.orphaned() == []  # nothing was stored


async def test_a_refusal_is_failed_not_rejected(store):
    # REJECTED is the Critic's terminal safety verdict. A provider filter (with known false
    # positives) is a decision for the supervisor, so it must not look terminal here.
    env = await run(ScriptedImages([ContentRefused(_pe(status=400, code=3030), "t")]), store)
    assert env.state.status is Status.FAILED


async def test_bytes_that_are_not_an_image_are_refused_but_still_charged(store, tmp_path):
    env = await run(ScriptedImages([b"<html>error page</html>"]), store)

    assert env.state.status is Status.FAILED
    assert env.output == {"failure": "bad_image"}
    assert env.state.spend.neurons == pytest.approx(57.6)  # the call happened and was billed
    assert not any((tmp_path / ".artifacts").glob("img_*"))


class BrokenStore(LocalArtifactStore):
    async def put(self, data: bytes, mime: str) -> str:
        raise OSError("disk full")


async def test_storage_failure_is_reported_with_the_spend(tmp_path):
    env = await run(ScriptedImages([image_bytes()]), BrokenStore(tmp_path))

    assert env.state.status is Status.FAILED
    assert env.output == {"failure": "storage"}
    assert env.state.spend.neurons == pytest.approx(57.6)


@pytest.mark.parametrize(
    "bad",
    [
        {"prompt": ""},
        {"width": 0},
        {"seed": -1},
        {"seed": None},
        {"tier": "platinum"},
    ],
)
async def test_invalid_inputs_fail_before_any_call(store, bad):
    backend = ScriptedImages([image_bytes()])

    env = await run(backend, store, **bad)

    assert env.state.status is Status.FAILED
    assert env.output == {"failure": "invalid_input"}
    assert backend.requests == []


async def test_tier_override_picks_that_tiers_chain(store):
    backend = ScriptedImages([image_bytes()])
    await run(backend, store, tier="max")  # raise_tier (M7) passes the next tier up
    assert backend.requests[0][0] == CFG.agent("image_creator").chain_for("max")


# -- review fixes -------------------------------------------------------------------------------


async def test_a_billed_refusal_is_charged_to_the_step(store):
    refused = ContentRefused(_pe(status=200, refused=True, cost_usd=0.002), "t", neurons=9.5)

    env = await run(ScriptedImages([refused]), store)

    assert env.output == {"failure": "content_refused"}
    assert env.state.spend.neurons == pytest.approx(9.5)
    assert env.state.spend.cost_usd == pytest.approx(0.002)


# -- review fix 4: a cancel during the store must not lose the image --------------------------


class SlowStore(LocalArtifactStore):
    """`put` takes a moment, so a cancel can land in the middle of it."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.started = asyncio.Event()

    async def put(self, data: bytes, mime: str) -> str:
        self.started.set()
        await asyncio.sleep(0.05)
        return await super().put(data, mime)


async def test_a_cancel_during_the_store_still_finishes_the_put_and_orphans_the_image(tmp_path):
    slow = SlowStore(tmp_path / ".artifacts")
    task = asyncio.ensure_future(run(ScriptedImages([image_bytes(1024, 1024)]), slow))
    await wait_for_event(slow.started)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):  # the cancel is still delivered to the caller
        await task

    orphans = [json.loads(row) for row in (slow.root / "orphaned.jsonl").read_text().splitlines()]
    assert [o["reason"] for o in orphans] == ["cancelled"]  # not leaked: it will be swept
    assert await slow.get(orphans[0]["image_id"])  # and the put really finished


# -- step 5: the stored type comes from the bytes; the provider's claim is passed through ------


class ClaimingImages(ScriptedImages):
    """A backend whose provider CLAIMS a media type for each image (like OpenRouter's
    data[].media_type), which may disagree with the bytes."""

    def __init__(self, replies: Any, claims: list[str | None]) -> None:
        super().__init__(replies)
        self.claims = claims

    async def image(self, task: str, request: Any, **kwargs: Any) -> Any:
        import dataclasses

        result = await super().image(task, request, **kwargs)
        output = result.output.model_copy(update={"media_types": self.claims})
        return dataclasses.replace(result, output=output)


async def test_the_stored_type_comes_from_the_bytes_and_a_wrong_claim_is_logged(store):
    from structlog.testing import capture_logs

    backend = ClaimingImages([image_bytes()], claims=["image/jpeg"])  # the bytes are PNG

    with capture_logs() as logs:
        env = await creator(backend, store).run(
            GOAL, STOP, AUTH, {"prompt": "a cat", "width": 1024, "height": 1024, "seed": 1}
        )

    assert env.state.status is Status.DONE and env.output["mime"] == "image/png"
    assert (store.root / f"{env.output['image_id']}.png").exists()  # never .jpg
    mismatch = [e for e in logs if e["event"] == "creator.media_type_mismatch"]
    assert mismatch and mismatch[0]["claimed"] == "image/jpeg"


async def test_a_matching_claim_is_not_logged(store):
    from structlog.testing import capture_logs

    backend = ClaimingImages([image_bytes()], claims=["image/png"])

    with capture_logs() as logs:
        await creator(backend, store).run(
            GOAL, STOP, AUTH, {"prompt": "a cat", "width": 1024, "height": 1024, "seed": 1}
        )

    assert not [e for e in logs if e["event"] == "creator.media_type_mismatch"]


# -- step 5: thumbnails are made once, at store time, and go with their image -------------------


async def test_thumbnails_are_stored_with_the_image(store):
    from media_engine.core.images import inspect_image

    env = await run(ScriptedImages([image_bytes(1024, 1024)]), store)

    result = GenerationResult.model_validate(env.output)
    assert result.thumbnails == [512, 256]
    for size in (512, 256):
        info = inspect_image(await store.get_thumbnail(result.image_id, size))
        assert max(info.width, info.height) == size and info.mime == "image/jpeg"
    assert await store.get(result.image_id) == image_bytes(1024, 1024)  # never a thumbnail


async def test_an_orphaned_image_takes_its_thumbnails_with_it(store):
    env = await run(ScriptedImages([image_bytes(1024, 1024)]), store)
    image_id = env.output["image_id"]

    await store.mark_orphaned(image_id, "unchecked")
    removed = await store.sweep_orphaned()

    assert removed == 3  # the image and both thumbnails
    assert list(store.root.glob(f"{image_id}*")) == []


async def test_a_failed_thumbnail_never_fails_the_image(store):
    class NoThumbs(LocalArtifactStore):
        async def put_thumbnail(self, image_id, size, data, mime):  # type: ignore[no-untyped-def]
            raise OSError("disk full")

    broken = NoThumbs(store.root)
    env = await run(ScriptedImages([image_bytes(1024, 1024)]), broken)

    assert env.state.status is Status.DONE and env.output["thumbnails"] == []
