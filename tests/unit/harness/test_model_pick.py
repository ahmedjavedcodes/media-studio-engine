"""Batch (b): the user's model pick and the resolution picker, end to end through
`create_images()`. The pick is checked against the plan's catalogue BEFORE any model call, then
handed to the gateway, which tries it first and falls back down the chain."""

from __future__ import annotations

from pathlib import Path

import pytest

from media_engine.agents.image_generation.create.agent import PICK_NOT_USED
from media_engine.contracts import AuthContext, Status
from media_engine.gateway.types import ModelRef
from media_engine.ports import LocalArtifactStore
from media_engine.services.catalog import catalog
from media_engine.services.create.entry import create_images
from tests.unit.harness.fakes import ScriptedChat, ScriptedImages, batch_reply, image_bytes
from tests.unit.harness.test_entry import ROUTE_SIMPLE, Backend
from tests.unit.harness.test_linear import enhanced, verdict

GPT_LOW = "openai/gpt-image-2@low"


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / ".artifacts")


def plan(name: str) -> AuthContext:
    return AuthContext(plan=name, execution_mode="auto", model_demand="balanced")  # type: ignore[arg-type]


async def run(store, images, **kwargs):  # type: ignore[no-untyped-def]
    llm = ScriptedChat([ROUTE_SIMPLE, enhanced(), verdict(9)])
    env = await create_images(
        "a cozy cafe", gateway=Backend(llm, images), artifacts=store, **kwargs
    )
    return env, llm


# -- the pick reaches the gateway ---------------------------------------------------------------


async def test_a_pick_is_sent_to_the_gateway_as_the_first_model_with_its_quality(store):
    images = ScriptedImages([image_bytes()], provider="openrouter", model="openai/gpt-image-2")

    env, _ = await run(store, images, model=GPT_LOW)

    assert env.state.status is Status.DONE
    assert images.firsts == [ModelRef("openrouter", "openai/gpt-image-2")]
    assert images.requests[0][1].quality == "low"
    assert PICK_NOT_USED not in env.output["issues"]


async def test_auto_sends_no_pick(store):
    images = ScriptedImages([image_bytes()])

    await run(store, images)

    assert images.firsts == [None] and images.requests[0][1].quality is None


async def test_a_pick_another_model_stood_in_for_is_reported_as_an_issue(store):
    images = ScriptedImages([image_bytes()])  # the fake answers as Cloudflare flux-1-schnell

    env, _ = await run(store, images, model="black-forest-labs/flux.2-klein-4b")

    assert env.state.status is Status.DONE_WITH_ISSUES
    assert PICK_NOT_USED in env.output["issues"]


# -- a pick the plan may not make is refused before any call ------------------------------------


@pytest.mark.parametrize(
    ("model", "who", "why"),
    [
        ("black-forest-labs/flux.2-pro", "free", "needs the pro plan"),
        ("openai/gpt-image-2@high", "max", "needs the ultra plan"),
        ("openai/gpt-image-2@high", "ultra", "batch only"),
        ("google/gemini-nano-banana-2.1", "ultra", "not in the catalogue"),  # offered: false
        ("meta/muse-image", "free", "not in the catalogue"),  # available: false
        ("made/up", "free", "not in the catalogue"),
    ],
)
async def test_a_pick_the_plan_may_not_make_is_invalid_before_any_call(store, model, who, why):
    images = ScriptedImages([])

    env, llm = await run(store, images, model=model, auth=plan(who))

    assert env.state.status is Status.FAILED
    assert env.output == {"failure": "invalid_input", "billable": False}
    assert why in (env.state.reason or "")
    assert llm.requests == [] and images.requests == []  # not even the Gate ran


def test_every_unlocked_interactive_option_is_a_valid_pick():
    from media_engine.services.catalog import resolve_pick

    for who in ("free", "pro", "max", "ultra"):
        for option in catalog(who).models:  # type: ignore[arg-type]
            if option.locked or option.batch_only:
                continue
            assert resolve_pick(option.id, who).slug == option.slug  # type: ignore[arg-type]


# -- resolution -----------------------------------------------------------------------------


@pytest.mark.parametrize(("res", "size"), [("1K", (1024, 576)), ("2K", (2048, 1152))])
async def test_the_resolution_scales_the_preset_size(store, res, size):
    images = ScriptedImages([image_bytes()])

    await run(store, images, settings={"aspect_ratio": "16:9", "resolution": res})

    request = images.requests[0][1]
    assert (request.width, request.height) == size  # the gateway fits it per model on the call


async def test_4k_is_ultra_only_and_checked_before_any_call(store):
    images = ScriptedImages([])

    env, llm = await run(store, images, settings={"resolution": "4K"}, auth=plan("max"))

    assert env.state.status is Status.FAILED and "resolution must be one of 1K, 2K" in (
        env.state.reason or ""
    )
    assert llm.requests == []


async def test_4k_on_ultra_is_sent(store):
    images = ScriptedImages([image_bytes()])

    await run(
        store, images, settings={"aspect_ratio": "1:1", "resolution": "4K"}, auth=plan("ultra")
    )

    assert (images.requests[0][1].width, images.requests[0][1].height) == (4096, 4096)


@pytest.mark.parametrize(
    ("who", "allowed"), [("free", ["1K", "2K"]), ("ultra", ["1K", "2K", "4K"])]
)
def test_the_catalogue_lists_the_plans_resolutions(who, allowed):
    assert catalog(who).resolutions == allowed  # type: ignore[arg-type]


# -- review nits: the pick's provider, run-level notes ------------------------------------------


async def test_the_picks_provider_comes_from_the_models_card(store, monkeypatch):
    from media_engine.gateway import image_registry as registry_module

    real = registry_module.image_registry()
    spec = real.get("openai/gpt-image-2")
    assert spec is not None and spec.provider == "openrouter"  # the default today
    moved = spec.model_copy(update={"provider": "elsewhere"})
    monkeypatch.setattr(
        "media_engine.tools.image_gen_model.image_registry",
        lambda: type("R", (), {"get": staticmethod(lambda slug: moved)})(),
    )
    images = ScriptedImages([image_bytes()])

    await run(store, images, model=GPT_LOW)

    assert images.firsts == [ModelRef("elsewhere", "openai/gpt-image-2")]  # not assumed OpenRouter


async def test_a_run_with_several_images_reports_a_substituted_pick_once(store):
    llm = ScriptedChat([ROUTE_SIMPLE, enhanced(), batch_reply([verdict(9), verdict(8)])])
    images = ScriptedImages([image_bytes(), image_bytes()])  # Cloudflare answers both

    env = await create_images(
        "a cozy cafe",
        n=2,
        model="black-forest-labs/flux.2-klein-4b",
        gateway=Backend(llm, images),
        artifacts=store,
    )

    notes = [i for i in env.output["issues"] if "model you chose" in i]
    assert notes == [PICK_NOT_USED]  # once, not "image 1: ..." and "image 2: ..."


@pytest.mark.parametrize(
    ("res", "returned", "noted"),
    [
        ("2K", 1024, True),  # asked for 2048, the model made 1024
        ("2K", 2048, False),  # the model delivered what was asked
        ("1K", 1024, False),  # 1K is the default: nothing to explain
    ],
)
async def test_a_resolution_the_model_could_not_deliver_is_reported_once(
    store, res, returned, noted
):
    from media_engine.agents.image_generation.create.agent import RESOLUTION_LIMITED

    images = ScriptedImages([image_bytes(returned, returned)])

    env, _ = await run(store, images, settings={"aspect_ratio": "1:1", "resolution": res})

    assert (RESOLUTION_LIMITED in env.output["issues"]) is noted
