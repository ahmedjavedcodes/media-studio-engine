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


# -- the Critic retries once INSIDE its step: no step is re-run, no new image -----------------


async def test_a_critic_glitch_is_retried_inside_the_step_without_regenerating(store):
    llm = ScriptedChat([enhanced(), "garbage", "more garbage", verdict(9)])
    images = ScriptedImages([red()])
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store), "free")

    state = final_state(run_store)
    assert env.state.status is Status.DONE
    assert len(images.requests) == 1  # no second image: the same image, checked again
    assert calls_by_agent(llm) == {ENHANCER: 1, CRITIC: 3}  # attempt + repair, then the retry
    assert state.critic_reasks_used == 0  # deprecated: the supervisor re-runs nothing
    assert [s.status for s in state.plan] == ["done", "done", "done"]
    assert await store.orphaned() == []


async def test_a_critic_that_fails_twice_leaves_the_image_unchecked_and_undelivered(store):
    llm = ScriptedChat([enhanced(), "junk", "junk", "junk", "junk"])  # 2 attempts x (reply+repair)
    images = ScriptedImages([red()])

    env = await create(build(llm, images, store), "ultra")

    assert env.state.status is Status.FAILED and len(images.requests) == 1
    assert calls_by_agent(llm)[CRITIC] == 4  # the one in-step retry, then it stops
    assert env.output["image_ids"] == []
    assert list(orphan_reasons(store).values()) == ["unchecked"]


async def test_no_provider_for_the_critic_is_not_retried(store):
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


async def test_run_state_round_trips_at_every_boundary_with_a_critic_retry(store):
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


# -- D4: the enhancer runs at temperature 0.2; nothing else sets one ----------------------------


async def test_the_enhancer_sends_temperature_0_2_and_the_critic_none(store):
    llm = ScriptedChat([enhanced(), verdict(9)])

    await create(build(llm, ScriptedImages([red()]), store), "free")

    by_agent = dict(zip(llm.agents, (req for _, req in llm.requests), strict=True))
    assert by_agent[ENHANCER].temperature == 0.2
    assert by_agent[CRITIC].temperature is None


# -- §4.3: the delivery is also a list of one-layer documents, ranked -------------------------------


async def test_n_images_are_delivered_as_ranked_one_layer_documents(store):
    from media_engine.schemas import Document

    llm = ScriptedChat([enhanced(), verdict(6), verdict(9), verdict(7)])
    images = ScriptedImages([red(), blue(), image_bytes(color="green")])

    env = await create(build(llm, images, store), "ultra", n=3)

    documents = [Document.model_validate(d) for d in env.output["documents"]]
    assert [d.rank for d in documents] == [1, 2, 3]
    assert [d.scores["quality_score"] for d in documents] == [9, 7, 6]  # best first
    assert [d.layers[0].image_id for d in documents] == env.output["image_ids"]  # same order
    assert all(len(d.layers) == 1 and d.layers[0].role == "base" for d in documents)
    assert all(d.version == 1 for d in documents)


async def test_a_rejected_run_has_no_documents(store):
    block = verdict(9, safety_verdict="block", safety_reason="gore")
    env = await create(build(ScriptedChat([enhanced(), block]), ScriptedImages([red()]), store))
    assert env.state.status is Status.REJECTED and "documents" not in env.output


# -- §5.3 streaming: only an image the Critic ALLOWED is ever streamed ------------------------------


async def test_a_blocked_image_is_never_streamed(store):
    sink = ListEventSink()
    block = verdict(9, safety_verdict="block", safety_reason="gore")
    llm = ScriptedChat([enhanced(), block])

    await create(build(llm, ScriptedImages([red()]), store, events=sink), "free")

    assert not [e for _, e in sink.events if e["phase"] == "image_ready"]


async def test_an_unchecked_image_is_never_streamed(store):
    sink = ListEventSink()
    llm = ScriptedChat([enhanced(), "junk", "junk", "junk", "junk"])  # the Critic never decides

    await create(build(llm, ScriptedImages([red()]), store, events=sink), "free")

    assert not [e for _, e in sink.events if e["phase"] == "image_ready"]
