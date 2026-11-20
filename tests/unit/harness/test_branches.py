"""M8 (N parallel branches) and M9 (events and cancellation). All fakes: SlowImages measures
concurrency, HangingImages / HangingChat never return, so only a cancel can end the call."""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from media_engine.contracts import Status
from media_engine.gateway.errors import AllProvidersFailed, DeadlineExceeded
from media_engine.harness.run_state import RunState
from media_engine.ports import ListEventSink, LocalArtifactStore
from tests.unit.harness.fakes import RaisingSink, ScriptedChat, ScriptedImages, image_bytes
from tests.unit.harness.test_linear import (
    CFG,
    CRITIC,
    ENHANCER,
    RecordingStore,
    build,
    create,
    enhanced,
    final_state,
    orphan_reasons,
    verdict,
)
from tests.unit.harness.test_linear import (
    calls_by_agent as chains,
)

COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]


def pics(count: int) -> list[bytes]:
    return [image_bytes(color=c) for c in COLORS[:count]]


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / ".artifacts")


def rich(**policy: Any) -> Any:
    """CFG with room in every budget for several images (the placeholder estimates are sized for
    one), and any other policy field overridden."""
    budgets = {
        plan: b.model_copy(
            update={"tokens_total": b.tokens_total * 20, "neurons_total": b.neurons_total * 20}
        )
        for plan, b in CFG.policy.budgets.items()
    }
    return CFG.model_copy(
        update={"policy": CFG.policy.model_copy(update={"budgets": budgets, **policy})}
    )


def tight() -> Any:
    """CFG whose free budget (14,000 tokens) can afford two images at the measured 5,000 tokens a
    critique, not four: free now fits n=4 for real, so budget refusals need a smaller budget."""
    free = CFG.policy.budgets["free"].model_copy(update={"tokens_total": 14000})
    budgets = {**CFG.policy.budgets, "free": free}
    return CFG.model_copy(update={"policy": CFG.policy.model_copy(update={"budgets": budgets})})


class SlowImages(ScriptedImages):
    """Every call takes a moment, so calls that are allowed to overlap do overlap."""

    def __init__(self, replies: Any, delay: float = 0.02) -> None:
        super().__init__(replies)
        self.delay = delay
        self.inflight = 0
        self.max_inflight = 0

    async def image(self, task: str, request: Any, **kwargs: Any) -> Any:
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.inflight -= 1
        return await super().image(task, request, **kwargs)


class HangingImages(ScriptedImages):
    """The first `answer` calls return as scripted; every later call hangs until cancelled."""

    def __init__(self, replies: Any, answer: int = 0) -> None:
        super().__init__(replies)
        self.answer = answer
        self.calls = 0
        self.hung = 0

    async def image(self, task: str, request: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls > self.answer:
            self.hung += 1
            await asyncio.sleep(3600)
        return await super().image(task, request, **kwargs)


class HangingChat(ScriptedChat):
    """A reply of "HANG" never returns: the call stays in flight until it is cancelled."""

    def __init__(self, replies: Any) -> None:
        super().__init__(replies)
        self.hung = 0

    async def chat(self, task: str, request: Any) -> Any:
        if self.replies and self.replies[0] == "HANG":
            self.replies.pop(0)
            self.requests.append((task, request))
            self.hung += 1
            await asyncio.sleep(3600)
        return await super().chat(task, request)


class FlagStore(RecordingStore):
    """What the backend does: sets cancel_requested on ITS copy of the run, in the store. The
    harness never sees the flag unless it looks in the store."""

    def __init__(self, when: Any) -> None:
        super().__init__()
        self.when = when

    async def load(self, run_id: str) -> RunState | None:
        state = await super().load(run_id)
        if state is not None and self.when():
            state.cancel_requested = True
        return state

    async def cancel_requested(self, run_id: str) -> bool:
        return bool(self.when())


def auth_for_tests() -> Any:
    from tests.unit.harness.test_supervisor import auth

    return auth("free")


def slow_poll(cfg: Any) -> Any:
    return cfg.model_copy(update={"policy": cfg.policy.model_copy(update={"cancel_poll_ms": 150})})


class SlowFirstThenHang(ScriptedImages):
    """Call 1 takes 60 ms and returns; call 2 hangs until cancelled."""

    def __init__(self) -> None:
        super().__init__([image_bytes(color="red")])
        self.calls = 0
        self.hung = 0

    async def image(self, task: str, request: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == 2:
            self.hung += 1
            await asyncio.sleep(3600)
        await asyncio.sleep(0.06)
        return await super().image(task, request, **kwargs)


def fast_poll(cfg: Any) -> Any:
    return cfg.model_copy(update={"policy": cfg.policy.model_copy(update={"cancel_poll_ms": 10})})


async def run(coro: Any) -> Any:
    return await asyncio.wait_for(coro, timeout=10)  # a missed cancel would hang: fail instead


# -- M8: N parallel branches --------------------------------------------------------------------


async def test_n_images_are_all_returned_ranked_best_first(store):
    llm = ScriptedChat([enhanced(), verdict(9), verdict(5), verdict(7), verdict(3)])
    images = ScriptedImages(pics(4))
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store, cfg=rich()), "free", n=4)

    assert env.state.status is Status.DONE
    assert len(env.output["image_ids"]) == 4 == len(set(env.output["image_ids"]))
    assert env.output["quality_scores"] == [9, 7, 5, 3]  # best first
    assert env.output["quality_score"] == 9
    # ONE enhancement; the vision model takes 3 images a request, so 4 images are two calls
    assert chains(llm) == {ENHANCER: 1, CRITIC: 2}
    assert sorted(r.seed for _, r, _ in images.requests) == [100, 101, 102, 103]  # distinct seeds
    plan = final_state(run_store).plan
    assert (plan[1].branches_total, plan[1].branches_done, plan[1].branches_aborted) == (4, 4, 0)
    assert (plan[2].branches_total, plan[2].branches_done, plan[2].branches_aborted) == (4, 4, 0)
    assert env.summary == "Your images are ready."


async def test_branches_run_concurrently_up_to_the_configured_cap(store):
    images = SlowImages(pics(4))
    llm = ScriptedChat([enhanced(), *[verdict(9)] * 4])

    await create(build(llm, images, store, cfg=rich()), "free", n=4)

    assert CFG.policy.max_parallel_branches == 4
    assert images.max_inflight == 4  # four at once, not one after another


async def test_concurrency_is_capped_at_min_of_n_and_the_configured_maximum(store):
    images = SlowImages(pics(4))
    llm = ScriptedChat([enhanced(), *[verdict(9)] * 4])
    env = await create(build(llm, images, store, cfg=rich(max_parallel_branches=2)), "free", n=4)
    assert images.max_inflight == 2 and env.state.status is Status.DONE

    small = SlowImages(pics(2))
    await create(
        build(ScriptedChat([enhanced(), verdict(9), verdict(9)]), small, store, cfg=rich()),
        "free",
        n=2,
    )
    assert small.max_inflight == 2  # n below the cap: n


async def test_one_failing_branch_does_not_cancel_the_others(store):
    failing = DeadlineExceeded("image_create", 60.0, [])
    images = ScriptedImages([*pics(1), failing, *pics(3)[1:]])
    llm = ScriptedChat([enhanced(), *[verdict(8)] * 3])
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store, cfg=rich()), "pro", n=4)

    state = final_state(run_store)
    assert len(images.requests) == 4  # every branch got its call
    assert env.state.status is Status.DONE_WITH_ISSUES
    assert len(env.output["image_ids"]) == 3  # the successful images are returned
    assert env.output["issues"] == ["1 of 4 requested images could not be created."]
    assert [r.failure for r in state.attempts].count("deadline") == 1  # recorded: refunded
    assert (state.plan[1].branches_done, state.plan[1].branches_aborted) == (3, 1)
    assert env.state.spend.retries_used == 0  # one branch failing is not a replan


async def test_a_branch_that_raises_is_a_failed_branch_not_a_failed_run(store):
    images = ScriptedImages([RuntimeError("adapter bug"), *pics(2)])
    llm = ScriptedChat([enhanced(), verdict(8), verdict(8)])

    env = await create(build(llm, images, store, cfg=rich()), "pro", n=3)

    assert env.state.status is Status.DONE_WITH_ISSUES
    assert len(env.output["image_ids"]) == 2


async def test_every_branch_failing_fails_the_run_with_no_replan(store):
    # v2.2: create is linear. When every branch fails, nothing is regenerated.
    failing = DeadlineExceeded("image_create", 60.0, [])
    images = ScriptedImages([failing] * 3 + pics(3))  # a replan would take the pics
    llm = ScriptedChat([enhanced()])
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store, cfg=rich()), "free", n=3)

    state = final_state(run_store)
    assert env.state.status is Status.FAILED
    assert env.output["image_ids"] == []
    assert len(images.requests) == 3  # one per branch, never a second round
    assert (state.retries_used, state.replans_used) == (0, 0)
    assert chains(llm) == {ENHANCER: 1}


async def test_every_branch_failing_for_a_reason_with_no_retry_fails_the_run(store):
    images = ScriptedImages([AllProvidersFailed("image_create", [])] * 3)

    env = await create(build(ScriptedChat([enhanced()]), images, store, cfg=rich()), "free", n=3)

    assert env.state.status is Status.FAILED
    assert env.output["image_ids"] == []
    assert len(images.requests) == 3 and env.state.spend.retries_used == 0


async def test_a_low_score_does_not_retry_an_n_image_request(store):
    llm = ScriptedChat([enhanced(), verdict(2, critique="weak"), verdict(1, critique="weaker")])
    images = ScriptedImages(pics(2))

    env = await create(build(llm, images, store, cfg=rich()), "pro", n=2)

    assert len(images.requests) == 2  # no retry
    assert env.state.status is Status.DONE_WITH_ISSUES
    assert env.output["issues"] == ["weak"]  # the best image's critique
    assert len(env.output["image_ids"]) == 2  # every image, whatever its score


async def test_a_block_among_n_images_drops_that_image_and_delivers_the_rest(store):
    # Settle rule (spec §5.5): one flagged image is dropped, the other three were paid for and
    # are allowed, so they are delivered. (It used to reject the whole run.)
    block = verdict(9, safety_verdict="block", safety_reason="graphic violence")
    llm = ScriptedChat([enhanced(), verdict(9), verdict(9), block, verdict(9)])

    env = await create(build(llm, ScriptedImages(pics(4)), store, cfg=rich()), "ultra", n=4)

    assert env.state.status is Status.DONE_WITH_ISSUES
    assert len(env.output["image_ids"]) == 3
    assert "1 of 4 images were removed by the safety review." in env.output["issues"]
    assert Counter(orphan_reasons(store).values()) == {"safety_block": 1}
    assert env.output["billable"] is True


async def test_every_image_blocked_rejects_the_run_and_logs_a_gate_miss(store):
    block = verdict(9, safety_verdict="block", safety_reason="graphic violence")
    llm = ScriptedChat([enhanced(), block, block])
    run_store = RecordingStore()

    env = await create(
        build(llm, ScriptedImages(pics(2)), store, run_store=run_store, cfg=rich()), "ultra", n=2
    )

    assert env.state.status is Status.REJECTED and env.output["image_ids"] == []
    assert Counter(orphan_reasons(store).values()) == {"safety_block": 2}
    assert any("GATE MISS" in d.text for d in final_state(run_store).decisions)


async def test_the_budget_is_checked_before_each_branchs_generation_call(store):
    # A small free budget cannot cover four images: later branches are refused BEFORE
    # their paid call, never after, and the budget is never exceeded.
    llm = ScriptedChat([enhanced(), *[verdict(9)] * 4])
    images = ScriptedImages(pics(4))
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store, cfg=tight()), "free", n=4)

    state = final_state(run_store)
    assert 1 <= len(images.requests) < 4
    assert env.state.status is Status.DONE_WITH_ISSUES
    assert [r.failure for r in state.attempts].count("budget") == 4 - len(images.requests)
    assert state.budget.neurons_spent <= state.budget.neurons_total
    assert state.budget.tokens_spent <= state.budget.tokens_total


async def test_branch_progress_events_use_public_labels_only(store):
    sink = ListEventSink()
    llm = ScriptedChat([enhanced(), *[verdict(9)] * 3])

    await create(build(llm, ScriptedImages(pics(3)), store, events=sink, cfg=rich()), "free", n=3)

    events = [e for _, e in sink.events]
    progress = Counter((e["step"], e["phase"]) for e in events)
    assert progress[(2, "progress")] == 3 and progress[(3, "progress")] == 1  # one Critic call
    assert {e["public_label"] for e in events} <= {
        "Understanding your request",
        "Creating your image",
        "Finishing up",
    }
    base = {"run_id", "step", "public_label", "phase", "ts"}
    for e in events:  # only image_ready adds a key: the image id, after the Critic allowed it
        assert set(e) == (base | {"image_id"} if e["phase"] == "image_ready" else base)
    assert progress[(3, "image_ready")] == 3  # each image streamed as soon as it was allowed


async def test_a_raising_event_sink_does_not_fail_a_run_with_branches(store):
    sink = RaisingSink()
    llm = ScriptedChat([enhanced(), *[verdict(9)] * 3])

    env = await create(
        build(llm, ScriptedImages(pics(3)), store, events=sink, cfg=rich()), "free", n=3
    )

    assert env.state.status is Status.DONE and sink.attempts > 0


async def test_run_state_round_trips_at_every_boundary_with_branches(store):
    run_store = RecordingStore()
    llm = ScriptedChat([enhanced(), *[verdict(9)] * 3])

    await create(
        build(llm, ScriptedImages(pics(3)), store, run_store=run_store, cfg=rich()), "free", n=3
    )

    for raw in run_store.snapshots:
        assert RunState.model_validate_json(raw).model_dump_json() == raw


# -- M9: cancellation ---------------------------------------------------------------------------


async def test_cancel_is_checked_before_each_branchs_generation_call(store):
    # One branch at a time: the cancel arrives while branch 0 finishes, so branch 1's own check
    # (immediately before ITS paid call) must refuse. The image of branch 0 was never checked.
    run_store = RecordingStore(cancel_when=lambda s: s.step(2).branches_done == 1)
    images = ScriptedImages(pics(4))
    llm = ScriptedChat([enhanced()])  # no critic reply: no critique may run

    env = await create(
        build(llm, images, store, run_store=run_store, cfg=rich(max_parallel_branches=1)),
        "free",
        n=4,
    )

    state = final_state(run_store)
    assert len(images.requests) == 1  # nothing was paid for after the cancel
    assert env.state.status is Status.CANCELLED
    gen = state.step(2)
    assert gen.status == "cancelled" and state.cancelled_at_step == 2
    assert (gen.branches_total, gen.branches_done, gen.branches_aborted) == (4, 1, 3)
    assert len(gen.partial_outputs) == 1
    assert [s.status for s in state.plan] == ["done", "cancelled", "skipped"]
    assert env.output["image_ids"] == []  # unchecked: never returned
    assert list(orphan_reasons(store).values()) == ["unchecked"]


async def test_cancel_aborts_the_in_flight_generation_calls(store):
    images = HangingImages([], answer=0)  # every call hangs
    run_store = FlagStore(lambda: images.hung == 2)
    llm = ScriptedChat([enhanced()])
    agent = build(llm, images, store, run_store=run_store, cfg=fast_poll(rich()))

    env = await run(create(agent, "free", n=2))

    state = final_state(run_store)
    assert env.state.status is Status.CANCELLED
    assert [s.status for s in state.plan] == ["done", "cancelled", "skipped"]
    gen = state.step(2)
    assert (gen.branches_total, gen.branches_done, gen.branches_aborted) == (2, 0, 2)
    assert gen.partial_outputs == [] and env.output["image_ids"] == []
    # only actual spend: the enhancer's. The aborted image calls report nothing.
    assert env.state.spend.neurons == state.step(1).spend.neurons
    assert env.state.spend.cost_usd == 0.0
    assert await store.orphaned() == []  # nothing was stored


async def test_cancel_keeps_finished_branches_and_never_returns_them_unchecked(store):
    images = HangingImages(pics(2), answer=2)  # two branches finish, two hang
    run_store = FlagStore(lambda: images.hung == 2)
    llm = ScriptedChat([enhanced()])  # the critique never starts
    agent = build(llm, images, store, run_store=run_store, cfg=fast_poll(rich()))

    env = await run(create(agent, "free", n=4))

    state = final_state(run_store)
    gen = state.step(2)
    assert env.state.status is Status.CANCELLED
    assert (gen.status, gen.branches_done, gen.branches_aborted) == ("cancelled", 2, 2)
    assert len(gen.partial_outputs) == 2  # recorded on the plan step...
    assert env.output["image_ids"] == []  # ...but never returned: the Critic did not rule
    assert sorted(orphan_reasons(store).values()) == ["unchecked", "unchecked"]
    assert state.step(3).status == "skipped" and state.cancelled_at_step == 2
    assert env.state.spend.neurons == sum(s.spend.neurons for s in state.plan)  # actual only


async def test_cancel_during_the_critique_returns_only_checked_images(store):
    # One call reviews both images, so a cancel in flight leaves BOTH unchecked.
    llm = HangingChat([enhanced(), "HANG"])
    run_store = FlagStore(lambda: llm.hung == 1)
    agent = build(llm, ScriptedImages(pics(2)), store, run_store=run_store, cfg=fast_poll(rich()))

    env = await run(create(agent, "free", n=2))

    state = final_state(run_store)
    crit = state.step(3)
    assert env.state.status is Status.CANCELLED
    assert (crit.status, crit.branches_done, crit.branches_aborted) == ("cancelled", 0, 2)
    assert env.output["image_ids"] == []  # nothing was checked, so nothing is returned
    assert sorted(orphan_reasons(store).values()) == ["unchecked", "unchecked"]


async def test_in_flight_cancel_of_a_single_image_run(store):
    images = HangingImages([], answer=0)
    run_store = FlagStore(lambda: images.hung == 1)
    agent = build(
        ScriptedChat([enhanced()]), images, store, run_store=run_store, cfg=fast_poll(CFG)
    )

    env = await run(create(agent, "free"))

    state = final_state(run_store)
    assert env.state.status is Status.CANCELLED
    assert [s.status for s in state.plan] == ["done", "cancelled", "skipped"]
    assert state.step(1).output["enhanced_prompt"]  # completed steps keep their outputs
    assert state.cancelled_at_step == 2


async def test_cancel_during_the_enhancer_freezes_the_plan(store):
    llm = HangingChat(["HANG"])
    run_store = FlagStore(lambda: llm.hung == 1)
    images = ScriptedImages([])
    agent = build(llm, images, store, run_store=run_store, cfg=fast_poll(CFG))

    env = await run(create(agent, "free"))

    state = final_state(run_store)
    assert env.state.status is Status.CANCELLED
    assert [s.status for s in state.plan] == ["cancelled", "skipped", "skipped"]
    assert state.cancelled_at_step == 1 and images.requests == []


async def test_the_cancel_flag_in_the_store_is_seen_at_the_next_boundary_too(store):
    # No watcher involved: the steps finish instantly, the flag is found at the step guard.
    run_store = FlagStore(lambda: len(run_store.snapshots) >= 3)
    llm = ScriptedChat([enhanced()])
    images = ScriptedImages([])

    env = await create(build(llm, images, store, run_store=run_store), "free")

    assert env.state.status is Status.CANCELLED
    assert images.requests == []


# -- review fix 1: the backend owns cancel_requested; save() must never write it ----------------


async def test_a_save_never_clobbers_a_cancel_the_backend_set():
    from media_engine.ports import InMemoryRunStore

    store = InMemoryRunStore()
    state = RunState(
        run_id="r1",
        user_id="u",
        auth=auth_for_tests(),
        budget=CFG.policy.budget_for("free"),
    )
    await store.save(state)
    await store.request_cancel("r1")  # the backend's write, on ITS side of the store
    await store.save(state)  # the harness saves its own copy, where the flag is False

    assert state.cancel_requested is False
    assert await store.cancel_requested("r1") is True  # still set
    loaded = await store.load("r1")
    assert loaded is not None and loaded.cancel_requested is True


async def test_a_cancel_set_in_the_store_survives_the_harness_saving_over_it(store):
    from media_engine.ports import InMemoryRunStore

    run_store = InMemoryRunStore()
    images = SlowFirstThenHang()
    agent = build(
        ScriptedChat([enhanced()]), images, store, run_store=run_store, cfg=slow_poll(rich())
    )

    async def backend() -> None:  # sets the flag as soon as the second call hangs
        while images.hung == 0:
            await asyncio.sleep(0.001)
        await run_store.request_cancel("r-clobber")

    task = asyncio.ensure_future(backend())
    env = await run(create(agent, "free", n=2, run_id="r-clobber"))
    await task

    # Branch 0 finishes (and the harness SAVES) after the flag was set and before the first
    # poll. With a save that overwrites the row, the flag is lost and the hung call never ends.
    assert env.state.status is Status.CANCELLED


# -- a block drops ONE image; a cancel still returns only images the Critic allowed --------------


# (block + cancel in flight no longer exists: ONE Critic call rules on every image at once, so a
# cancel in flight leaves all of them unchecked: see the cancel-during-critique test above.)


async def test_a_branch_gate_cannot_overwrite_a_block_set_while_it_awaited_the_store(store):
    from media_engine.agents.image_generation.create.agent import Outcome

    agent = build(ScriptedChat([]), ScriptedImages([]), store, cfg=rich())
    ctx = agent.new_context(auth_for_tests(), {"prompt": "x", "n": 2})
    step, record = ctx.step("critique"), ctx.state.round[1]

    async def sibling_blocks_during_the_await(c: Any) -> None:
        c.outcome = Outcome(Status.REJECTED, "blocked")  # a sibling critic finished meanwhile
        c.state.cancel_requested = True  # ...and the store said cancel

    agent._refresh_cancel = sibling_blocks_during_the_await  # type: ignore[method-assign]

    assert await agent._branch_gate(ctx, step, record) is False
    assert ctx.outcome is not None and ctx.outcome.status is Status.REJECTED


async def test_settle_never_delivers_a_blocked_image_whatever_the_outcome(store):
    from media_engine.agents.image_generation.create.agent import Outcome

    agent = build(ScriptedChat([]), ScriptedImages([]), store, cfg=rich())
    ctx = agent.new_context(auth_for_tests(), {"prompt": "x", "n": 2})
    first, second = ctx.state.round
    first.image_id, first.verdict, first.quality_score = "img_a", "allow", 9
    second.image_id, second.verdict = "img_b", "block"
    ctx.outcome = Outcome(Status.CANCELLED, "cancelled")  # however the outcome got to be this

    outcome = await agent._settle(ctx)

    assert outcome.status is Status.CANCELLED and outcome.output == {"image_ids": ["img_a"]}


async def test_settle_rejects_when_every_reviewed_image_was_blocked(store):
    agent = build(ScriptedChat([]), ScriptedImages([]), store, cfg=rich())
    ctx = agent.new_context(auth_for_tests(), {"prompt": "x", "n": 2})
    for record, image_id in zip(ctx.state.round, ("img_a", "img_b"), strict=True):
        record.image_id, record.verdict = image_id, "block"

    outcome = await agent._settle(ctx)

    assert outcome.status is Status.REJECTED


# -- review fix 6: the branch gate does not charge the generate estimate twice -------------------


async def test_the_gate_counts_a_generated_image_as_owing_only_its_critique(store):
    est = CFG.policy.step_estimates
    gen, crit = est["generate"].neurons, est["critique"].neurons
    owed = 2  # two images already generated and charged, their critiques still to come
    exact = gen + crit * (1 + owed)  # this branch's generate + 3 critiques
    old_formula = (gen + crit) * (1 + owed)  # what the gate used to demand
    assert old_formula > exact

    budgets = {
        # reserve_fraction is 0 (create is linear): the whole budget is usable by the steps
        plan: b.model_copy(update={"tokens_total": 10**7, "neurons_total": exact + 0.01})
        for plan, b in CFG.policy.budgets.items()
    }
    cfg = CFG.model_copy(update={"policy": CFG.policy.model_copy(update={"budgets": budgets})})
    agent = build(ScriptedChat([]), ScriptedImages([]), store, cfg=cfg)
    ctx = agent.new_context(auth_for_tests(), {"prompt": "x", "n": 4})

    assert agent._affordable(ctx, "generate", units=1, owed=owed) is True  # fits exactly
    assert agent._affordable(ctx, "generate", units=1, owed=owed + 1) is False
    assert agent._affordable(ctx, "generate", units=2, owed=owed) is False  # one more in flight


# -- D5: ONE guard per step, taking the branch count; cancel still checked per branch ----------


async def test_one_guard_decision_per_step_not_per_branch(store):
    llm = ScriptedChat([enhanced(), *[verdict(9)] * 4])
    run_store = RecordingStore()

    await create(
        build(llm, ScriptedImages(pics(4)), store, run_store=run_store, cfg=rich()), "ultra", n=4
    )

    guards = [d.text for d in final_state(run_store).decisions if d.text.startswith("guard before")]
    assert len(guards) == 3  # enhance, generate, critique: not one per branch (was 9)
    assert any("guard before generate [4 of 4 branches]" in g for g in guards)


async def test_the_one_guard_funds_what_it_can_and_refuses_the_rest_before_any_call(store):
    llm = ScriptedChat([enhanced(), *[verdict(9)] * 4])
    images = ScriptedImages(pics(4))
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store, cfg=tight()), "free", n=4)

    state = final_state(run_store)
    funded = len(images.requests)
    assert 1 <= funded < 4  # free cannot afford four: the rest never reach a provider
    assert [r.failure for r in state.round].count("budget") == 4 - funded
    assert len(env.output["image_ids"]) == funded


async def test_the_real_free_budget_now_funds_four_images(store):
    # Measured (2026-11-20): a critique costs ~5,000 tokens per image at worst, not 12,000, so the
    # free plan's own budget covers n=4 (it used to refuse two of the four).
    llm = ScriptedChat([enhanced(), *[verdict(9)] * 4])
    images = ScriptedImages(pics(4))
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store), "free", n=4)

    assert len(images.requests) == 4 and len(env.output["image_ids"]) == 4
    assert [r.failure for r in final_state(run_store).round] == [None] * 4


@pytest.mark.parametrize("n", [1, 3])
async def test_the_critic_step_budget_grows_with_the_number_of_images(store, n):
    llm = ScriptedChat([enhanced(), *[verdict(9)] * n])
    agent = build(llm, ScriptedImages(pics(n)), store)
    seen: list[int] = []
    real_run = agent._critic.run

    async def spy(goal: Any, stop: Any, auth: Any, inputs: Any) -> Any:
        seen.append(stop.max_tokens)
        return await real_run(goal, stop, auth, inputs)

    agent._critic.run = spy  # type: ignore[method-assign]

    await create(agent, "free", n=n)

    per_image = CFG.policy.stopping_condition("critique", "free", "x").max_tokens
    assert seen == [per_image * n]  # one call; a repair of it re-sends every image
