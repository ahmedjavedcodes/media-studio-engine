"""Create is LINEAR (IMAGE-CREATE-OPTIMIZE v2.2): one attempt, no regeneration loop, no lever
call, no edge back to an earlier step. A low score, a creator failure or a content refusal never
buys another image. The one re-run left is the Critic RE-ASK: the SAME images checked again
inside the critique step, with its own small cap, never generating.

Replaces test_retries.py (M7). The tests that covered behaviour create still has were moved
here; the ones that tested the retry loop, the levers and the retry reserve were deleted with
that code. Everything runs on fakes."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from media_engine.agents.image_grand.supervisor import ImageGrandAgent, create_image_grand_agent
from media_engine.contracts import Status
from media_engine.gateway.errors import (
    AllProvidersFailed,
    ContentRefused,
    DeadlineExceeded,
    ProviderError,
)
from media_engine.gateway.llm_gateway import chain_resolver
from media_engine.harness.run_state import RunState
from media_engine.ports import ListEventSink, LocalArtifactStore
from tests.unit.harness.fakes import (
    AgentRecordingLLMGateway,
    RaisingSink,
    ScriptedChat,
    ScriptedImages,
    image_bytes,
)
from tests.unit.harness.test_supervisor import (
    AGENT_NAMES,
    CFG,
    RecordingStore,
    create,
    enhanced,
    verdict,
)

REFUSED = ContentRefused(
    ProviderError("cloudflare", "m", "NSFW", status=400, code=3030), "image_create"
)
ENHANCER = "prompt_enhancer"
CRITIC = "critic"


def red() -> bytes:
    return image_bytes(color="red")


def blue() -> bytes:
    return image_bytes(color="blue")


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / ".artifacts")


def build(
    llm: ScriptedChat,
    images: ScriptedImages,
    store: LocalArtifactStore,
    *,
    run_store: Any = None,
    events: Any = None,
    cfg: Any = CFG,
    allow_paid: bool = False,
    trace: Any = None,
) -> ImageGrandAgent:
    seeds = itertools.count(100)
    return create_image_grand_agent(
        cfg,
        llm=AgentRecordingLLMGateway(llm, chain_resolver(cfg.agents)),
        images=images,
        artifacts=store,
        events=events if events is not None else ListEventSink(),
        run_store=run_store,
        paid_providers=("openrouter",),
        allow_paid=allow_paid,
        seed_source=lambda: next(seeds),
        trace=trace,
    )


def final_state(run_store: RecordingStore) -> RunState:
    return RunState.model_validate_json(run_store.snapshots[-1])


def orphan_reasons(store: LocalArtifactStore) -> dict[str, str]:
    path = store.root / "orphaned.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["image_id"]: row["reason"] for row in rows}


def calls_by_agent(llm: ScriptedChat) -> Counter[str]:
    """LLM calls per AGENT, by name (AgentRecordingLLMGateway), never by prompt text."""
    return Counter(llm.agents)


# -- one attempt, whatever happens ---------------------------------------------------------------


async def test_a_low_score_is_delivered_with_its_critique_and_never_regenerates(store):
    llm = ScriptedChat([enhanced(), verdict(3, critique="too dark")])  # a 3rd call would raise
    images = ScriptedImages([red()])  # a 2nd image call would raise

    env = await create(build(llm, images, store), "ultra")

    assert env.state.status is Status.DONE_WITH_ISSUES
    assert len(env.output["image_ids"]) == 1 and env.output["issues"] == ["too dark"]
    assert len(images.requests) == 1
    assert calls_by_agent(llm) == {ENHANCER: 1, CRITIC: 1}


async def test_the_supervisor_never_makes_an_llm_call(store):
    run_store = RecordingStore()
    llm = ScriptedChat([enhanced(), verdict(2)])

    await create(build(llm, ScriptedImages([red()]), store, run_store=run_store), "max")

    state = final_state(run_store)
    assert state.supervisor_spend.llm_calls == 0
    assert "image_grand_agent" not in calls_by_agent(llm)
    assert state.retries_used == 0 and [a.lever for a in state.attempts] == [None]


@pytest.mark.parametrize(
    "failure",
    [DeadlineExceeded("image_create", 60.0, []), b"this is not an image"],
    ids=["deadline", "bad_image"],
)
async def test_a_creator_deadline_or_bad_image_fails_without_a_second_image(store, failure):
    images = ScriptedImages([failure, blue()])

    env = await create(build(ScriptedChat([enhanced()]), images, store), "pro")

    assert env.state.status is Status.FAILED
    assert len(images.requests) == 1  # no reseed: create never buys another image


async def test_a_content_refusal_fails_without_re_enhancing(store):
    llm = ScriptedChat([enhanced()])  # a re-enhance would raise
    images = ScriptedImages([REFUSED, red()])

    env = await create(build(llm, images, store), "free")

    assert env.state.status is Status.FAILED
    assert "content_refused" in (env.state.reason or "")
    assert len(images.requests) == 1 and calls_by_agent(llm) == {ENHANCER: 1}


@pytest.mark.parametrize("plan", ["free", "pro"])
async def test_no_provider_for_the_creator_fails_at_once(store, plan):
    images = ScriptedImages([AllProvidersFailed("image_create", [])])

    env = await create(build(ScriptedChat([enhanced()]), images, store), plan)

    assert env.state.status is Status.FAILED
    assert len(images.requests) == 1


# -- the Critic re-ask: the same images, its own cap, no new image -------------------------------


async def test_a_critic_failure_checks_the_same_image_again_without_regenerating(store):
    llm = ScriptedChat([enhanced(), "garbage", "more garbage", verdict(9)])
    images = ScriptedImages([red()])
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store), "free")

    assert env.state.status is Status.DONE
    assert len(images.requests) == 1  # no second image: the same image, checked again
    assert calls_by_agent(llm) == {ENHANCER: 1, CRITIC: 3}
    assert final_state(run_store).critic_reasks_used == 1
    assert await store.orphaned() == []


async def test_critic_re_asks_have_their_own_small_cap(store):
    cap = CFG.policy.max_critic_reasks
    bad = ["junk", "junk"]  # one failed critique: the reply and its one repair
    llm = ScriptedChat([enhanced(), *bad * (cap + 1)])
    images = ScriptedImages([red()])

    env = await create(build(llm, images, store), "ultra")

    assert env.state.status is Status.FAILED and len(images.requests) == 1
    assert calls_by_agent(llm)[CRITIC] == 2 * (cap + 1)  # asked, re-asked `cap` times, stops


async def test_a_critic_failure_after_the_cap_leaves_the_image_unchecked_and_undelivered(store):
    bad = ["garbage", "more garbage"]
    llm = ScriptedChat([enhanced(), *bad * (CFG.policy.max_critic_reasks + 1)])

    env = await create(build(llm, ScriptedImages([red()]), store), "free")

    assert env.state.status is Status.FAILED
    assert env.output["image_ids"] == []
    assert list(orphan_reasons(store).values()) == ["unchecked"]


async def test_no_provider_for_the_critic_is_not_re_asked(store):
    llm = ScriptedChat([enhanced(), AllProvidersFailed("vision", [])])
    images = ScriptedImages([red()])

    env = await create(build(llm, images, store), "pro")

    assert env.state.status is Status.FAILED
    assert llm.replies == [] and len(images.requests) == 1
    assert list(orphan_reasons(store).values()) == ["unchecked"]


# -- block is terminal --------------------------------------------------------------------------


@pytest.mark.parametrize("score", [0, 10])
async def test_a_block_is_rejected_with_no_further_call(store, score):
    block = verdict(score, safety_verdict="block", safety_reason="graphic violence")
    llm = ScriptedChat([enhanced(), block])  # any further call would raise
    images = ScriptedImages([red(), blue()])

    env = await create(build(llm, images, store), "ultra")

    assert env.state.status is Status.REJECTED
    assert env.output["image_ids"] == []
    assert len(images.requests) == 1
    assert calls_by_agent(llm) == {ENHANCER: 1, CRITIC: 1}


# -- budget -------------------------------------------------------------------------------------


def cfg_free_neurons(total: float) -> Any:
    budgets = dict(CFG.policy.budgets)
    budgets["free"] = budgets["free"].model_copy(update={"neurons_total": total})
    return CFG.model_copy(update={"policy": CFG.policy.model_copy(update={"budgets": budgets})})


async def test_with_nothing_delivered_a_budget_refusal_is_failed_before_the_image(store):
    est = CFG.policy.step_estimates
    usable = est["generate"].neurons + est["critique"].neurons - 1
    cfg = cfg_free_neurons(usable / (1 - CFG.policy.budgets["free"].reserve_fraction))
    images = ScriptedImages([red()])

    env = await create(build(ScriptedChat([enhanced()]), images, store, cfg=cfg), "free")

    assert env.state.status is Status.FAILED and images.requests == []


def cfg_pro_usd_cap(cap: float) -> Any:
    budgets = dict(CFG.policy.budgets)
    budgets["pro"] = budgets["pro"].model_copy(update={"cost_cap_usd": cap})
    return CFG.model_copy(update={"policy": CFG.policy.model_copy(update={"budgets": budgets})})


async def test_a_paid_image_that_would_break_the_dollar_cap_is_refused_before_the_call(store):
    est = CFG.policy.step_estimates
    assert est["generate"].cost_usd > 0 and est["critique"].cost_usd > 0
    one_image = est["generate"].cost_usd + est["critique"].cost_usd
    cfg = cfg_pro_usd_cap(one_image * 0.5)  # steps may use 70% of the cap: less than one image
    images = ScriptedImages([red()], provider="openrouter", cost_usd=est["generate"].cost_usd)

    env = await create(
        build(ScriptedChat([enhanced()]), images, store, cfg=cfg, allow_paid=True), "pro"
    )

    assert env.state.status is Status.FAILED and "budget" in (env.state.reason or "")
    assert images.requests == []  # refused BEFORE the paid call


async def test_a_paid_fallback_image_within_the_cap_runs_and_its_cost_is_charged(store):
    est = CFG.policy.step_estimates
    images = ScriptedImages([red()], provider="openrouter", cost_usd=est["generate"].cost_usd)
    run_store = RecordingStore()

    env = await create(
        build(
            ScriptedChat([enhanced(), verdict(9)]),
            images,
            store,
            cfg=cfg_pro_usd_cap(1.0),
            run_store=run_store,
            allow_paid=True,
        ),
        "pro",
    )

    assert env.state.status in (Status.DONE, Status.DONE_WITH_ISSUES)
    assert final_state(run_store).budget.cost_spent_usd == pytest.approx(est["generate"].cost_usd)


async def test_the_dollar_estimate_is_ignored_when_paid_providers_are_off(store):
    env = await create(
        build(
            ScriptedChat([enhanced(), verdict(9)]),
            ScriptedImages([red()]),
            store,
            cfg=cfg_pro_usd_cap(0.0),
            allow_paid=False,
        ),
        "pro",
    )
    assert env.state.status is Status.DONE


# -- RunState, events, labels -------------------------------------------------------------------


async def test_run_state_round_trips_at_every_boundary_with_a_critic_re_ask(store):
    run_store = RecordingStore()
    llm = ScriptedChat([enhanced(), "junk", "junk", verdict(9)])

    env = await create(build(llm, ScriptedImages([red()]), store, run_store=run_store), "free")

    for raw in run_store.snapshots:
        assert RunState.model_validate_json(raw).model_dump_json() == raw
    state = final_state(run_store)
    assert env.state.status is Status.DONE and state.status is Status.DONE
    assert [(a.number, a.lever, a.seed) for a in state.attempts] == [(1, None, 100)]


async def test_no_internal_agent_name_reaches_an_event_or_the_output(store):
    sink = ListEventSink()
    llm = ScriptedChat([enhanced(), "junk", "junk", verdict(9)])

    env = await create(build(llm, ScriptedImages([red()]), store, events=sink), "free")

    allowed = {"Understanding your request", "Creating your image", "Finishing up"}
    assert {e["public_label"] for _, e in sink.events} <= allowed
    blob = json.dumps([e for _, e in sink.events]) + json.dumps(env.output)
    assert not [name for name in AGENT_NAMES if name in blob]


async def test_a_raising_event_sink_does_not_fail_the_run(store):
    sink = RaisingSink()
    llm = ScriptedChat([enhanced(), "junk", "junk", verdict(9)])

    env = await create(build(llm, ScriptedImages([red()]), store, events=sink), "free")

    assert env.state.status is Status.DONE and sink.attempts > 0


async def test_shape_issues_are_reported_for_every_delivered_image(store):
    # A paid provider returns square images for a 16:9 request, on both branches.
    llm = ScriptedChat([enhanced(), verdict(8), verdict(7)])
    images = ScriptedImages(
        [image_bytes(512, 512, color="red"), image_bytes(512, 512, color="blue")],
        provider="openrouter",
    )

    env = await create(build(llm, images, store), "ultra", n=2, settings={"aspect_ratio": "16:9"})

    shape = [i for i in env.output["issues"] if i.startswith("image ")]
    assert len(env.output["image_ids"]) == 2
    assert shape == [
        "image 1: requested 16:9, got 512x512",
        "image 2: requested 16:9, got 512x512",
    ]


# -- review round 2: the generate guard must not refuse images the plan is meant to give --------


@pytest.mark.parametrize(("plan", "n"), [("free", 2), ("pro", 4), ("max", 4), ("ultra", 4)])
async def test_the_production_budget_affords_the_plans_images(store, plan, n):
    # With reserve_fraction 0.30, free n=2 got 1 image (520 neurons / 24,000 tokens needed,
    # 418 / 20,900 usable) and pro n=4 got 3. The production config must give them all.
    llm = ScriptedChat([enhanced(), *[verdict(9)] * n])
    colors = ("red", "blue", "green", "white")
    images = ScriptedImages([image_bytes(color=c) for c in colors[:n]])

    env = await create(build(llm, images, store), plan, n=n)

    assert env.state.status is Status.DONE
    assert len(env.output["image_ids"]) == n and len(images.requests) == n


async def test_a_critic_re_ask_still_fits_on_free_after_two_images(store):
    # A re-ask is checked against the whole remaining budget, not just the reserve.
    llm = ScriptedChat([enhanced(), "junk", "junk", "junk", "junk", verdict(9), verdict(8)])
    images = ScriptedImages([red(), blue()])
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store), "free", n=2)

    assert env.state.status is Status.DONE and len(env.output["image_ids"]) == 2
    assert final_state(run_store).critic_reasks_used == 1 and len(images.requests) == 2
