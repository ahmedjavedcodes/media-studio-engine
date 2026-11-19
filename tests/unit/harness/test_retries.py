"""M7: retries and levers. The retry loop wraps the attempt graph; the cap is per plan; completed
steps are reused; a block never reaches a retry path; a content refusal permits only re_enhance;
retries may draw on the budget reserve. The lever choice is the supervisor's ONE LLM call.

UNVERIFIED against real models (no .env yet): that `reseed` gives a different image on
flux-1-schnell (`seed_honoured` is None), and that `raise_tier` changes anything (agents.yaml
maps every plan to the same chain). Everything here runs on fakes."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from media_engine.agents.image_grand.supervisor import (
    ImageGrandAgent,
    create_image_grand_agent,
    permitted_levers,
)
from media_engine.contracts import Status
from media_engine.gateway.errors import (
    AllProvidersFailed,
    ContentRefused,
    DeadlineExceeded,
    ProviderError,
)
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver
from media_engine.harness.run_state import RunState
from media_engine.ports import ListEventSink, LocalArtifactStore
from tests.unit.harness.fakes import RaisingSink, ScriptedChat, ScriptedImages, image_bytes
from tests.unit.harness.test_supervisor import (
    AGENT_NAMES,
    CFG,
    RecordingStore,
    create,
    enhanced,
    verdict,
)

ALL_LEVERS = ["reseed", "re_enhance", "switch_model", "raise_tier"]
REFUSED = ContentRefused(
    ProviderError("cloudflare", "m", "NSFW", status=400, code=3030), "image_create"
)


def lever(name: str, reason: str = "because") -> str:
    return json.dumps({"lever": name, "reason": reason})


def red() -> bytes:
    return image_bytes(color="red")


def blue() -> bytes:
    return image_bytes(color="blue")


def green() -> bytes:
    return image_bytes(color="green")


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
    seeds = itertools.count(100)  # 100 for the first attempt, 101 for the next, ...
    return create_image_grand_agent(
        cfg,
        llm=LLMGateway(llm, chain_resolver(cfg.agents)),
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


def chains(llm: ScriptedChat) -> Counter[str]:
    return Counter(task for task, _ in llm.requests)


# the chain each LLM caller uses on the free plan
ENHANCER = CFG.agent("prompt_enhancer").chain_for("free")
CRITIC = CFG.agent("critic").chain_for("free")
LEVER_CHAIN = CFG.agent("image_grand_agent").chain_for("free")


# -- the cap, per plan --------------------------------------------------------------------------


async def test_free_gets_one_retry_then_delivers_every_attempt_best_first(store):
    llm = ScriptedChat(
        [enhanced(), verdict(3, critique="c1"), lever("reseed"), verdict(5, critique="c2")]
    )
    images = ScriptedImages([red(), blue()])
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store), "free")

    assert CFG.policy.max_retries["free"] == 1
    assert env.state.status is Status.DONE_WITH_ISSUES
    assert len(images.requests) == 2  # the cap: no third attempt
    state = final_state(run_store)
    first, second = (a.image_id for a in state.attempts)
    assert env.output["image_ids"] == [second, first]  # best (5) first, then 3
    assert env.output["issues"] == ["c2"]  # the remaining critique, of the best attempt
    assert env.output["quality_score"] == 5
    assert env.state.spend.retries_used == 1 == state.retries_used


async def test_pro_gets_two_retries(store):
    llm = ScriptedChat(
        [
            enhanced(),
            verdict(3, critique="c1"),
            lever("reseed"),
            verdict(4, critique="c2"),
            lever("reseed"),
            verdict(2, critique="c3"),
        ]
    )
    images = ScriptedImages([red(), blue(), green()])
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store), "pro")

    assert CFG.policy.max_retries["pro"] == 2
    assert len(images.requests) == 3
    ids = [a.image_id for a in final_state(run_store).attempts]
    assert env.output["image_ids"] == [ids[1], ids[0], ids[2]]  # scores 4, 3, 2
    assert env.output["issues"] == ["c2"]
    assert env.state.spend.retries_used == 2
    assert env.state.status is Status.DONE_WITH_ISSUES


async def test_equal_scores_keep_the_earlier_attempt_first(store):
    llm = ScriptedChat([enhanced(), verdict(3), lever("reseed"), verdict(3)])
    run_store = RecordingStore()

    env = await create(
        build(llm, ScriptedImages([red(), blue()]), store, run_store=run_store), "free"
    )

    first, second = (a.image_id for a in final_state(run_store).attempts)
    assert env.output["image_ids"] == [first, second]


async def test_a_retry_that_reaches_the_threshold_delivers_only_that_image(store):
    llm = ScriptedChat([enhanced(), verdict(3), lever("reseed"), verdict(9)])
    run_store = RecordingStore()

    env = await create(
        build(llm, ScriptedImages([red(), blue()]), store, run_store=run_store), "free"
    )

    first, second = (a.image_id for a in final_state(run_store).attempts)
    assert env.state.status is Status.DONE
    assert env.output["image_ids"] == [second]
    assert env.output["issues"] == []
    assert orphan_reasons(store) == {first: "superseded"}  # stored, never delivered: swept


async def test_a_passing_first_attempt_never_retries(store):
    llm = ScriptedChat([enhanced(), verdict(CFG.policy.quality_threshold["free"])])
    images = ScriptedImages([red()])

    env = await create(build(llm, images, store), "free")

    assert env.state.status is Status.DONE
    assert len(images.requests) == 1
    assert chains(llm) == {ENHANCER: 1, CRITIC: 1}  # not one supervisor call


# -- completed steps are reused -----------------------------------------------------------------


async def test_reseed_reuses_the_enhancement_and_sends_a_new_seed(store):
    llm = ScriptedChat([enhanced(), verdict(3), lever("reseed"), verdict(9)])
    images = ScriptedImages([red(), blue()])
    sink = ListEventSink()

    await create(build(llm, images, store, events=sink), "free")

    assert chains(llm) == {ENHANCER: 1, CRITIC: 2, LEVER_CHAIN: 1}  # enhancer ran once
    (_, first, _), (_, second, _) = images.requests
    assert first.prompt == second.prompt  # the same enhanced prompt
    assert (first.seed, second.seed) == (100, 101)
    steps_started = [e["step"] for _, e in sink.events if e["phase"] == "started"]
    assert steps_started == [1, 2, 3, 2, 3]  # the enhance step did not run again, nor re-emit


async def test_re_enhance_reruns_the_enhancer_with_the_critique_as_feedback(store):
    llm = ScriptedChat(
        [
            enhanced(),
            verdict(3, critique="The lamps are missing."),
            lever("re_enhance"),
            enhanced(enhanced_prompt="A coffee shop with many warm lamps."),
            verdict(9),
        ]
    )
    images = ScriptedImages([red(), blue()])

    env = await create(build(llm, images, store), "free")

    assert chains(llm) == {ENHANCER: 2, CRITIC: 2, LEVER_CHAIN: 1}
    second_enhancer_input = llm.requests[3][1].messages[1]["content"]
    assert "The lamps are missing." in second_enhancer_input  # <feedback>
    assert images.requests[1][1].prompt.startswith("A coffee shop with many warm lamps.")
    assert env.state.status is Status.DONE


async def test_re_enhance_does_not_stop_to_ask_the_user_a_question_it_already_got_past(store):
    question = [{"kind": "question", "question": "Which city?"}]
    llm = ScriptedChat(
        [
            enhanced(),
            verdict(3),
            lever("re_enhance"),
            enhanced(missing_info=question),
            verdict(9),
        ]
    )

    env = await create(build(llm, ScriptedImages([red(), blue()]), store), "free", mode="manual")

    assert env.state.status is Status.DONE  # not NEEDS_INPUT, with an image already in hand
    assert env.output["assumptions"] == ["proceeded without an answer to: Which city?"]


async def test_a_failing_re_enhance_still_delivers_the_earlier_checked_image(store):
    llm = ScriptedChat([enhanced(), verdict(3), lever("re_enhance"), "nope", "still nope"])
    images = ScriptedImages([red()])
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store), "free")

    assert env.state.status is Status.DONE_WITH_ISSUES
    assert env.output["image_ids"] == [final_state(run_store).attempts[0].image_id]
    assert env.output["issues"][-1] == "A further attempt could not be completed."
    assert len(images.requests) == 1


# -- the lever choice: the supervisor's one LLM call --------------------------------------------


async def test_the_lever_call_is_counted_and_charged_to_the_run(store):
    llm = ScriptedChat([enhanced(), verdict(3), lever("reseed"), verdict(9)])
    run_store = RecordingStore()

    env = await create(build(llm, ScriptedImages([red(), blue()]), store, run_store=run_store))

    state = final_state(run_store)
    assert state.supervisor_spend.llm_calls == 1
    assert sum(s.spend.llm_calls for s in state.plan) == 4 - 1  # enhancer 1 + critic 2
    assert env.state.spend.llm_calls == len(llm.requests) == 4
    assert state.budget.tokens_spent >= state.supervisor_spend.tokens > 0  # charged to the budget


async def test_the_lever_prompt_carries_the_critique_and_only_permitted_levers(store):
    llm = ScriptedChat(
        [enhanced(), verdict(3, critique="Hands are melted."), lever("reseed"), verdict(9)]
    )

    await create(build(llm, ScriptedImages([red(), blue()]), store), "free")

    content = llm.requests[2][1].messages[1]["content"]
    assert "Hands are melted." in content
    assert "reseed" in content and "re_enhance" in content
    assert "switch_model" not in content and "raise_tier" not in content  # not on free


async def test_a_lever_not_permitted_on_the_plan_falls_back_to_the_first_permitted(store):
    # free may not switch_model: the choice is not obeyed and no second call is made.
    llm = ScriptedChat([enhanced(), verdict(3), lever("switch_model"), verdict(9)])
    images = ScriptedImages([red(), blue()])
    run_store = RecordingStore()

    await create(build(llm, images, store, run_store=run_store), "free")

    assert [a.lever for a in final_state(run_store).attempts] == [None, "reseed"]
    assert images.start_at == [0, 0]  # the model was not switched
    assert chains(llm)[LEVER_CHAIN] == 1


async def test_a_failed_lever_call_falls_back_to_reseed_and_the_retry_still_happens(store):
    llm = ScriptedChat([enhanced(), verdict(3), "garbage", "more garbage", verdict(9)])
    run_store = RecordingStore()

    env = await create(
        build(llm, ScriptedImages([red(), blue()]), store, run_store=run_store), "free"
    )

    state = final_state(run_store)
    assert [a.lever for a in state.attempts] == [None, "reseed"]
    assert state.supervisor_spend.llm_calls == 2  # the call and its one repair, both counted
    assert env.state.status is Status.DONE


# -- the levers ---------------------------------------------------------------------------------


async def test_switch_model_starts_the_chain_at_the_next_model_and_it_sticks(store):
    llm = ScriptedChat(
        [
            enhanced(),
            verdict(3),
            lever("switch_model"),
            verdict(3),
            lever("reseed"),
            verdict(3),
        ]
    )
    images = ScriptedImages([red(), blue(), green()])

    await create(build(llm, images, store), "pro")

    assert images.start_at == [0, 1, 1]  # a later reseed stays on the switched model


async def test_raise_tier_sends_the_next_tier_to_the_image_creator(store):
    # agents.yaml maps every plan to the same chain today, so this is a no-op in production
    # (UNVERIFIED / reported). Here the tiers are given different chains to prove the wiring.
    creator = CFG.agents["image_creator"]
    cfg = CFG.model_copy(
        update={
            "agents": {
                **CFG.agents,
                "image_creator": creator.model_copy(
                    update={"models": {**creator.models, "ultra": "image_edit"}}
                ),
            }
        }
    )
    llm = ScriptedChat([enhanced(), verdict(3), lever("raise_tier"), verdict(9)])
    images = ScriptedImages([red(), blue()])

    await create(build(llm, images, store, cfg=cfg), "max")

    assert [task for task, _, _ in images.requests] == ["image_create", "image_edit"]


def test_permitted_levers_follow_the_plan_table():
    cfg = CFG.policy

    def differs(tier: str) -> str:  # tiers with different chains: raise_tier has an effect
        return f"chain_{tier}"

    assert permitted_levers(
        cfg.levers_for("free"), tier="free", failure=None, chain_of=differs
    ) == [
        "reseed",
        "re_enhance",
    ]
    assert permitted_levers(cfg.levers_for("pro"), tier="pro", failure=None, chain_of=differs) == [
        "reseed",
        "re_enhance",
        "switch_model",
    ]
    assert (
        permitted_levers(cfg.levers_for("max"), tier="max", failure=None, chain_of=differs)
        == ALL_LEVERS
    )
    # nothing above ultra to raise to
    assert (
        permitted_levers(cfg.levers_for("ultra"), tier="ultra", failure=None, chain_of=differs)
        == ALL_LEVERS[:3]
    )


@pytest.mark.parametrize("plan", ["free", "pro", "max", "ultra"])
def test_after_a_content_refusal_the_only_permitted_lever_is_re_enhance(plan):
    levers = permitted_levers(
        CFG.policy.levers_for(plan),
        tier=plan,
        failure="content_refused",
        chain_of=lambda tier: f"chain_{tier}",  # even where raise_tier would have an effect
    )
    assert levers == ["re_enhance"]  # reseed resends to the same filter; the others go around it


# -- failures: which retry, if any --------------------------------------------------------------


async def test_a_critic_failure_checks_the_same_image_again_without_regenerating(store):
    llm = ScriptedChat([enhanced(), "garbage", "more garbage", verdict(9)])
    images = ScriptedImages([red()])

    env = await create(build(llm, images, store), "free")

    assert env.state.status is Status.DONE
    assert len(images.requests) == 1  # no second image: it is the same image, checked again
    assert chains(llm) == {ENHANCER: 1, CRITIC: 3}
    assert env.state.spend.retries_used == 0  # a re-ask is not one of the plan's retries
    assert await store.orphaned() == []


async def test_a_critic_failure_after_the_cap_leaves_the_image_unchecked_and_undelivered(store):
    bad = ["garbage", "more garbage"]
    llm = ScriptedChat([enhanced(), *bad, *bad])  # free: one retry, and it fails too

    env = await create(build(llm, ScriptedImages([red()]), store), "free")

    assert env.state.status is Status.FAILED
    assert env.output["image_ids"] == []
    assert list(orphan_reasons(store).values()) == ["unchecked"]


async def test_no_provider_for_the_critic_is_not_retried(store):
    llm = ScriptedChat([enhanced(), AllProvidersFailed("vision", [])])
    images = ScriptedImages([red()])

    env = await create(build(llm, images, store), "pro")

    assert env.state.status is Status.FAILED
    assert llm.replies == [] and len(images.requests) == 1
    assert env.state.spend.retries_used == 0
    assert list(orphan_reasons(store).values()) == ["unchecked"]


@pytest.mark.parametrize(
    "failure",
    [DeadlineExceeded("image_create", 60.0, []), b"this is not an image"],
    ids=["deadline", "bad_image"],
)
async def test_a_creator_deadline_or_bad_image_is_one_reseed(store, failure):
    llm = ScriptedChat([enhanced(), verdict(9)])  # no lever call: the lever is not a choice
    images = ScriptedImages([failure, blue()])
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store), "free")

    assert env.state.status is Status.DONE
    assert [a.lever for a in final_state(run_store).attempts] == [None, "reseed"]
    assert [r.seed for _, r, _ in images.requests] == [100, 101]
    assert chains(llm) == {ENHANCER: 1, CRITIC: 1}  # the enhancement was reused


@pytest.mark.parametrize("plan", ["free", "pro"])
async def test_no_provider_for_the_creator_fails_at_once(store, plan):
    images = ScriptedImages([AllProvidersFailed("image_create", [])])

    env = await create(build(ScriptedChat([enhanced()]), images, store), plan)

    assert env.state.status is Status.FAILED
    assert len(images.requests) == 1
    assert env.state.spend.retries_used == 0


async def test_a_content_refusal_re_enhances_with_no_lever_call(store):
    llm = ScriptedChat([enhanced(), enhanced(enhanced_prompt="A calm coffee shop."), verdict(9)])
    images = ScriptedImages([REFUSED, red()])
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store), "free")

    assert env.state.status is Status.DONE
    assert [a.lever for a in final_state(run_store).attempts] == [None, "re_enhance"]
    assert chains(llm) == {ENHANCER: 2, CRITIC: 1}  # one permitted lever: nothing to choose
    assert "safety filter declined" in llm.requests[1][1].messages[1]["content"]


@pytest.mark.parametrize(("plan", "retries"), [("free", 1), ("pro", 2)])
async def test_repeated_refusals_only_ever_re_enhance_and_end_failed(store, plan, retries):
    llm = ScriptedChat([enhanced()] * (retries + 1))
    images = ScriptedImages([REFUSED] * (retries + 1))
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store), plan)

    assert env.state.status is Status.FAILED
    assert "content_refused" in (env.state.reason or "")
    levers = [a.lever for a in final_state(run_store).attempts]
    assert levers == [None] + ["re_enhance"] * retries  # never reseed, never a model switch
    assert images.start_at == [0] * (retries + 1)


# -- block is terminal: no retry path reaches it ------------------------------------------------


@pytest.mark.parametrize("score", [0, 10])
async def test_a_block_on_the_first_attempt_is_rejected_with_no_retry(store, score):
    block = verdict(score, safety_verdict="block", safety_reason="graphic violence")
    llm = ScriptedChat([enhanced(), block])  # a lever call would raise: nothing is scripted
    images = ScriptedImages([red(), blue()])

    env = await create(build(llm, images, store), "ultra")  # plenty of retries left

    assert env.state.status is Status.REJECTED
    assert env.output["image_ids"] == []
    assert len(images.requests) == 1
    assert env.state.spend.retries_used == 0
    ultra_enhancer = CFG.agent("prompt_enhancer").chain_for("ultra")
    assert chains(llm) == {ultra_enhancer: 1, CRITIC: 1}  # and no supervisor call


async def test_a_block_on_a_retry_rejects_the_run_and_returns_no_image_at_all(store):
    block = verdict(9, safety_verdict="block", safety_reason="graphic violence")
    llm = ScriptedChat([enhanced(), verdict(3), lever("reseed"), block])
    run_store = RecordingStore()

    env = await create(
        build(llm, ScriptedImages([red(), blue()]), store, run_store=run_store), "pro"
    )

    first, second = (a.image_id for a in final_state(run_store).attempts)
    assert env.state.status is Status.REJECTED
    assert env.output["image_ids"] == []  # not even the earlier, allowed image
    assert orphan_reasons(store) == {first: "run_rejected", second: "safety_block"}


# -- budget -------------------------------------------------------------------------------------


def cfg_free_neurons(total: float) -> Any:
    budgets = dict(CFG.policy.budgets)
    budgets["free"] = budgets["free"].model_copy(update={"neurons_total": total})
    return CFG.model_copy(update={"policy": CFG.policy.model_copy(update={"budgets": budgets})})


# Each LLM call is billed `neurons`, each image 57.6 (ScriptedImages). Estimates (harness.yaml):
# enhance 15, generate 60, critique 200, supervise 10; the steps may use 70% of the budget and a
# retry may use all of it.


async def test_a_retry_draws_on_the_budget_reserve(store):
    # After the first pass the steps' own share is spent (1 attempt: 157.6 of 336), so a retry
    # (the lever call + generate + critique = 270) fits only in the reserve: 480 - 157.6.
    cfg = cfg_free_neurons(480)
    assert 0.7 * 480 - 157.6 < 270 <= 480 - 157.6
    llm = ScriptedChat([enhanced(), verdict(3), lever("reseed"), verdict(9)], neurons=50)
    images = ScriptedImages([red(), blue()])

    env = await create(build(llm, images, store, cfg=cfg), "free")

    assert env.state.status is Status.DONE
    assert len(images.requests) == 2  # the retry ran, on reserve


async def test_a_retry_is_refused_before_its_generation_call_when_the_budget_is_gone(store):
    # Enough for the first pass and the lever call, not for the generate + critique after it.
    cfg = cfg_free_neurons(450)
    llm = ScriptedChat([enhanced(), verdict(3, critique="c1"), lever("reseed")], neurons=50)
    images = ScriptedImages([red(), blue()])
    run_store = RecordingStore()

    env = await create(build(llm, images, store, cfg=cfg, run_store=run_store), "free")

    assert len(images.requests) == 1  # refused BEFORE the second image was paid for
    assert env.state.status is Status.DONE_WITH_ISSUES  # an attempt was delivered: not FAILED
    assert env.output["image_ids"] == [final_state(run_store).attempts[0].image_id]
    assert env.output["issues"] == ["c1", "No budget was left for another attempt."]
    assert [s.status for s in final_state(run_store).plan] == ["done", "skipped", "skipped"]


async def test_the_lever_call_itself_is_refused_when_no_retry_could_follow(store):
    cfg = cfg_free_neurons(520)
    llm = ScriptedChat([enhanced(), verdict(3, critique="c1")], neurons=100)  # no lever reply
    images = ScriptedImages([red(), blue()])

    env = await create(build(llm, images, store, cfg=cfg), "free")

    assert len(llm.requests) == 2 and len(images.requests) == 1  # nothing wasted on a lever
    assert env.state.status is Status.DONE_WITH_ISSUES
    assert env.output["issues"][-1] == "No budget was left for another attempt."


async def test_with_nothing_delivered_a_budget_refusal_is_still_failed(store):
    # M6 behaviour kept: no attempt reached a verdict, so there is nothing to deliver.
    est = CFG.policy.step_estimates
    usable = est["generate"].neurons + est["critique"].neurons - 1
    cfg = cfg_free_neurons(usable / (1 - CFG.policy.budgets["free"].reserve_fraction))
    images = ScriptedImages([red()])

    env = await create(build(ScriptedChat([enhanced()]), images, store, cfg=cfg), "free")

    assert env.state.status is Status.FAILED and images.requests == []


# -- cancel -------------------------------------------------------------------------------------


async def test_cancel_before_the_lever_call_stops_without_calling_the_model(store):
    run_store = RecordingStore(cancel_when=lambda s: s.attempts[0].quality_score is not None)
    llm = ScriptedChat([enhanced(), verdict(3)])  # a lever call would raise
    images = ScriptedImages([red(), blue()])

    env = await create(build(llm, images, store, run_store=run_store), "free")

    assert env.state.status is Status.CANCELLED
    assert len(llm.requests) == 2 and len(images.requests) == 1
    state = final_state(run_store)
    assert [s.status for s in state.plan] == ["done", "done", "done"]  # frozen, not cleared
    assert state.cancelled_at_step == 3
    # the image the Critic already allowed is returned: it was checked, and it was paid for
    assert env.output["image_ids"] == [state.attempts[0].image_id]


async def test_cancel_during_a_retry_is_checked_before_its_generation_call(store):
    # The cancel arrives after the lever call: the retry's generate step must refuse to pay.
    run_store = RecordingStore(cancel_when=lambda s: s.supervisor_spend.llm_calls == 1)
    llm = ScriptedChat([enhanced(), verdict(3), lever("reseed")])
    images = ScriptedImages([red(), blue()])

    env = await create(build(llm, images, store, run_store=run_store), "free")

    assert env.state.status is Status.CANCELLED
    assert len(images.requests) == 1
    assert final_state(run_store).step(2).status == "skipped"


# -- RunState, events, labels -------------------------------------------------------------------


async def test_run_state_round_trips_at_every_boundary_across_a_retry(store):
    run_store = RecordingStore()
    llm = ScriptedChat([enhanced(), verdict(3), lever("re_enhance"), enhanced(), verdict(9)])

    env = await create(
        build(llm, ScriptedImages([red(), blue()]), store, run_store=run_store), "free"
    )

    for raw in run_store.snapshots:
        assert RunState.model_validate_json(raw).model_dump_json() == raw
    state = final_state(run_store)
    assert env.state.status is Status.DONE and state.status is Status.DONE
    assert [(a.number, a.lever, a.seed) for a in state.attempts] == [
        (1, None, 100),
        (2, "re_enhance", 101),
    ]
    assert state.attempts[1].feedback == state.attempts[0].critique


async def test_no_internal_agent_name_reaches_an_event_or_the_output_during_a_retry(store):
    sink = ListEventSink()
    llm = ScriptedChat([enhanced(), verdict(3), lever("reseed"), verdict(9)])

    env = await create(build(llm, ScriptedImages([red(), blue()]), store, events=sink), "free")

    allowed = {"Understanding your request", "Creating your image", "Finishing up"}
    assert {e["public_label"] for _, e in sink.events} <= allowed
    blob = json.dumps([e for _, e in sink.events]) + json.dumps(env.output)
    assert not [name for name in AGENT_NAMES if name in blob]


async def test_a_raising_event_sink_does_not_fail_a_run_with_retries(store):
    sink = RaisingSink()
    llm = ScriptedChat([enhanced(), verdict(3), lever("reseed"), verdict(9)])

    env = await create(build(llm, ScriptedImages([red(), blue()]), store, events=sink), "free")

    assert env.state.status is Status.DONE and sink.attempts > 0


# -- review fix 5: the dollar cap is pre-checked when a paid fallback is reachable ---------------


def cfg_pro_usd_cap(cap: float) -> Any:
    budgets = dict(CFG.policy.budgets)
    budgets["pro"] = budgets["pro"].model_copy(update={"cost_cap_usd": cap})
    return CFG.model_copy(update={"policy": CFG.policy.model_copy(update={"budgets": budgets})})


async def test_a_paid_image_that_would_break_the_dollar_cap_is_refused_before_the_call(store):
    est = CFG.policy.step_estimates
    assert est["generate"].cost_usd > 0 and est["critique"].cost_usd > 0
    # steps may use 70% of the cap; make that less than one generate + its critique
    one_image = est["generate"].cost_usd + est["critique"].cost_usd
    cfg = cfg_pro_usd_cap(one_image * 0.5)
    images = ScriptedImages([red()], provider="openrouter", cost_usd=est["generate"].cost_usd)

    env = await create(
        build(ScriptedChat([enhanced()]), images, store, cfg=cfg, allow_paid=True), "pro"
    )

    assert env.state.status is Status.FAILED and "budget" in (env.state.reason or "")
    assert images.requests == []  # refused BEFORE the paid call


async def test_a_paid_fallback_image_within_the_cap_runs_and_its_cost_is_charged(store):
    est = CFG.policy.step_estimates
    cfg = cfg_pro_usd_cap(1.0)
    images = ScriptedImages([red()], provider="openrouter", cost_usd=est["generate"].cost_usd)
    run_store = RecordingStore()

    env = await create(
        build(
            ScriptedChat([enhanced(), verdict(9)]),
            images,
            store,
            cfg=cfg,
            run_store=run_store,
            allow_paid=True,
        ),
        "pro",
    )

    assert env.state.status is Status.DONE_WITH_ISSUES or env.state.status is Status.DONE
    assert final_state(run_store).budget.cost_spent_usd == pytest.approx(est["generate"].cost_usd)


async def test_the_dollar_estimate_is_ignored_when_paid_providers_are_off(store):
    # ALLOW_PAID is off: the gateway skips every paid entry, so no dollars can be spent.
    cfg = cfg_pro_usd_cap(0.0)
    env = await create(
        build(
            ScriptedChat([enhanced(), verdict(9)]),
            ScriptedImages([red()]),
            store,
            cfg=cfg,
            allow_paid=False,
        ),
        "pro",
    )
    assert env.state.status is Status.DONE


# -- review: smaller changes ---------------------------------------------------------------------


async def test_a_critic_glitch_does_not_spend_the_single_free_retry(store):
    # free has ONE retry. The Critic fails once (re-asked on the same image, its own cap), then
    # scores low: the retry must still be available for the lever.
    llm = ScriptedChat(
        [enhanced(), "junk", "junk", verdict(3, critique="c1"), lever("reseed"), verdict(9)]
    )
    images = ScriptedImages([red(), blue()])
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store), "free")

    state = final_state(run_store)
    assert env.state.status is Status.DONE and len(images.requests) == 2
    assert (state.critic_reasks_used, state.retries_used) == (1, 1)


async def test_critic_re_asks_have_their_own_small_cap(store):
    cap = CFG.policy.max_critic_reasks
    bad = ["junk", "junk"]  # one failed critique: the reply and its one repair
    llm = ScriptedChat([enhanced(), *bad * (cap + 1)])
    images = ScriptedImages([red()])

    env = await create(build(llm, images, store), "ultra")

    assert env.state.status is Status.FAILED and len(images.requests) == 1
    assert chains(llm)[CRITIC] == 2 * (cap + 1)  # asked, then re-asked `cap` times, then stops


def test_raise_tier_is_not_offered_when_the_next_tier_has_the_same_chain():
    from media_engine.agents.image_grand.supervisor import permitted_levers

    same = lambda tier: "image_create"
    differs = lambda tier: "image_edit" if tier == "ultra" else "image_create"
    plan = CFG.policy.levers_for("max")

    assert "raise_tier" not in permitted_levers(plan, tier="max", failure=None, chain_of=same)
    assert "raise_tier" in permitted_levers(plan, tier="max", failure=None, chain_of=differs)


async def test_the_lever_prompt_never_offers_raise_tier_while_it_is_a_no_op(store):
    llm = ScriptedChat([enhanced(), verdict(3), lever("reseed"), verdict(9)])

    await create(build(llm, ScriptedImages([red(), blue()]), store), "max")

    assert "raise_tier" not in llm.requests[2][1].messages[1]["content"]


async def test_shape_issues_are_reported_for_every_delivered_image(store):
    # A paid provider returns a square image for a 16:9 request, for both retried attempts.
    llm = ScriptedChat(
        [enhanced(), verdict(3, critique="c1"), lever("reseed"), verdict(4, critique="c2")]
    )
    images = ScriptedImages(
        [image_bytes(512, 512, color="red"), image_bytes(512, 512, color="blue")],
        provider="openrouter",
    )

    env = await create(build(llm, images, store), "free", settings={"aspect_ratio": "16:9"})

    shape = [i for i in env.output["issues"] if i.startswith("image ")]
    assert len(env.output["image_ids"]) == 2
    assert shape == [
        "image 1: requested 16:9, got 512x512",
        "image 2: requested 16:9, got 512x512",
    ]
