"""M6: Image Grand Agent happy path, single image. The supervisor makes ZERO LLM calls on a
first-pass success; every call is a step agent's and is counted in that step's Spend. All model
calls are faked: ScriptedChat for LLMs, ScriptedImages for the image gateway."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from media_engine.agents.image_grand.attempt_graph import run_attempt
from media_engine.agents.image_grand.supervisor import ImageGrandAgent, create_image_grand_agent
from media_engine.contracts import AuthContext, Goal, Status
from media_engine.gateway.errors import ContentRefused, ProviderError
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver
from media_engine.harness.run_state import RunState
from media_engine.ports import ListEventSink, LocalArtifactStore
from media_engine.services.create.loader import load_create_config
from tests.unit.harness.fakes import RaisingSink, ScriptedChat, ScriptedImages, image_bytes

CFG = load_create_config()
GOAL = Goal(task="create", instructions="Create the image the user asked for.")
PROMPT = "a cozy coffee shop"
AGENT_NAMES = ("prompt_enhancer", "image_creator", "critic", "image_grand_agent")
SEED = 1234


def auth(plan: str = "free", mode: str = "auto") -> AuthContext:
    return AuthContext(plan=plan, execution_mode=mode, model_demand="balanced")  # type: ignore[arg-type]


def enhanced(**overrides: Any) -> str:
    body: dict[str, Any] = {
        "enhanced_prompt": "A cozy coffee shop interior at dusk, warm lamps, wooden tables.",
        "must_have": ["a coffee shop", "cozy atmosphere"],
        "missing_info": [],
        "text_wanted": [],
        "summary": "A warm coffee shop.",
    }
    body.update(overrides)
    return json.dumps(body)


def verdict(score: int = 9, **overrides: Any) -> str:
    body: dict[str, Any] = {
        "quality_score": score,
        "safety_verdict": "allow",
        "safety_reason": None,
        "critique": "Matches the request; the lamps are slightly blown out.",
        "summary": "A warm coffee shop interior at dusk.",
    }
    body.update(overrides)
    return json.dumps(body)


class RecordingStore:
    """A RunStore that keeps every saved JSON snapshot, and can poke the live state (the way a
    caller's cancel would) when a condition on it first holds."""

    def __init__(self, cancel_when: Any = None) -> None:
        self.snapshots: list[str] = []
        self.cancel_when = cancel_when

    async def save(self, state: RunState) -> None:
        self.snapshots.append(state.model_dump_json())
        if self.cancel_when is not None and self.cancel_when(state):
            state.cancel_requested = True

    async def load(self, run_id: str) -> RunState | None:
        return RunState.model_validate_json(self.snapshots[-1]) if self.snapshots else None


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / ".artifacts")


def build(
    llm: ScriptedChat,
    images: ScriptedImages,
    store: LocalArtifactStore,
    *,
    events: Any = None,
    run_store: Any = None,
    cfg: Any = CFG,
    paid: tuple[str, ...] = ("openrouter",),
) -> ImageGrandAgent:
    return create_image_grand_agent(
        cfg,
        llm=LLMGateway(llm, chain_resolver(cfg.agents)),
        images=images,
        artifacts=store,
        events=events if events is not None else ListEventSink(),
        run_store=run_store,
        paid_providers=paid,
        seed_source=lambda: SEED,
    )


async def create(agent: ImageGrandAgent, plan: str = "free", mode: str = "auto", **inputs: Any):  # type: ignore[no-untyped-def]
    stop = CFG.policy.stopping_condition("supervise", plan, "finished images")  # type: ignore[arg-type]
    return await agent.run(GOAL, stop, auth(plan, mode), {"prompt": PROMPT, **inputs})


# -- the M6 acceptance test ---------------------------------------------------------------------


@pytest.mark.parametrize("plan", ["free", "pro", "max", "ultra"])
async def test_first_pass_success_makes_zero_supervisor_llm_calls(store, plan):
    llm = ScriptedChat([enhanced(), verdict(9)])  # an extra call (e.g. a lever choice) raises
    images = ScriptedImages([image_bytes(1024, 1024)])
    run_store = RecordingStore()
    agent = build(llm, images, store, run_store=run_store)

    env = await create(agent, plan)

    assert env.state.status is Status.DONE
    assert len(env.output["image_ids"]) == 1
    # Count calls by summing Spend.llm_calls across the plan, not on the gateway.
    final = RunState.model_validate_json(run_store.snapshots[-1])
    per_step = {s.agent: s.spend.llm_calls for s in final.plan}
    assert per_step == {"prompt_enhancer": 1, "image_creator": 0, "critic": 1}
    assert env.state.spend.llm_calls == 2 == len(llm.requests)
    assert llm.replies == []  # the supervisor itself asked the model nothing
    assert len(images.requests) == 1


async def test_happy_path_envelope(store):
    llm = ScriptedChat([enhanced(text_wanted=["OPEN"]), verdict(9)])
    agent = build(llm, ScriptedImages([image_bytes(1024, 1024)]), store)

    env = await create(agent)

    image_id = env.output["image_ids"][0]
    assert await store.get(image_id)  # a stored, deliverable image
    assert env.output == {
        "run_id": env.output["run_id"],
        "image_ids": [image_id],
        "quality_score": 9,
        "issues": [],
        "assumptions": [],
        "text_wanted": ["OPEN"],  # handed on for the separate text layer
        "width": 1024,
        "height": 1024,
    }
    assert env.summary == "A warm coffee shop interior at dusk."  # the Critic's summary
    assert await store.orphaned() == []


async def test_steps_are_wired_with_the_right_inputs(store):
    llm = ScriptedChat([enhanced(), verdict()])
    images = ScriptedImages([image_bytes()])

    await create(build(llm, images, store), settings={"aspect_ratio": "16:9"})

    (enh_task, enh_req), (crit_task, crit_req) = llm.requests
    assert enh_task == CFG.agent("prompt_enhancer").chain_for("free")
    assert "aspect_ratio: 16:9" in enh_req.messages[1]["content"]  # settings reach the Enhancer
    assert crit_task == "vision"
    task, request, deadline_s = images.requests[0]
    assert task == "image_create"
    assert request.prompt.startswith("A cozy coffee shop interior")  # the ENHANCED prompt
    assert (request.width, request.height, request.seed) == (1024, 576, SEED)
    assert deadline_s == CFG.policy.deadlines_ms["generate"] / 1000
    critic_text = crit_req.messages[1]["content"][0]["text"]
    assert PROMPT in critic_text and "golden" not in critic_text  # raw prompt to the Critic


# -- the attempt graph: one run = one attempt ---------------------------------------------------


def test_attempt_graph_is_linear_with_one_way_out(store):
    agent = build(ScriptedChat([]), ScriptedImages([]), store)
    edges = [(e.source, e.target, e.conditional) for e in agent.graph.get_graph().edges]

    assert edges == [
        ("__start__", "enhance", False),
        ("enhance", "generate", False),
        ("generate", "critique", False),
        ("critique", "__end__", False),
    ]  # no conditional edges, nothing loops back to enhance, a single END edge


async def test_one_graph_run_makes_at_most_one_image_call(store):
    # A low score would trigger a retry in M7. The retry must wrap the graph, so a single graph
    # run still makes exactly one image call even with more images and replies on offer.
    llm = ScriptedChat([enhanced(), verdict(0), enhanced(), verdict(9)])
    images = ScriptedImages([image_bytes(), image_bytes()])
    agent = build(llm, images, store)
    ctx = agent.new_context(auth(), {"prompt": PROMPT})

    await run_attempt(agent.graph, ctx.state.run_id, ctx)

    assert len(images.requests) == 1
    assert len(llm.requests) == 2


async def test_graph_state_carries_only_a_reference(store):
    agent = build(ScriptedChat([enhanced(), verdict()]), ScriptedImages([image_bytes()]), store)
    ctx = agent.new_context(auth(), {"prompt": PROMPT})

    final = await agent.graph.ainvoke(
        {"run_id": ctx.state.run_id}, config={"configurable": {"run": ctx}}
    )

    assert final == {"run_id": ctx.state.run_id}  # no plan, budget, spend or status copy
    assert ctx.state.step(3).status == "done"  # ...those live on RunState only


# -- RunState: JSON at every step boundary ------------------------------------------------------


async def test_run_state_round_trips_at_every_boundary(store):
    run_store = RecordingStore()
    agent = build(
        ScriptedChat([enhanced(), verdict()]),
        ScriptedImages([image_bytes()]),
        store,
        run_store=run_store,
    )

    env = await create(agent)

    # initial save + one before each of the 3 steps + one after each + the final save
    assert len(run_store.snapshots) == 1 + 3 + 3 + 1
    for raw in run_store.snapshots:
        assert RunState.model_validate_json(raw).model_dump_json() == raw
    final = await run_store.load(env.output["run_id"])
    assert final is not None and final.status is Status.DONE
    assert [s.status for s in final.plan] == ["done", "done", "done"]
    assert final.step(2).output_ref == env.output["image_ids"][0]
    assert final.step(1).output["enhanced_prompt"]  # completed steps keep their outputs (M7)


# -- events -------------------------------------------------------------------------------------


async def test_events_use_public_labels_only(store):
    sink = ListEventSink()
    agent = build(
        ScriptedChat([enhanced(), verdict()]), ScriptedImages([image_bytes()]), store, events=sink
    )

    env = await create(agent)

    phases = [(e["step"], e["public_label"], e["phase"]) for _, e in sink.events]
    assert phases == [
        (1, "Understanding your request", "started"),
        (1, "Understanding your request", "finished"),
        (2, "Creating your image", "started"),
        (2, "Creating your image", "finished"),
        (3, "Finishing up", "started"),
        (3, "Finishing up", "finished"),
    ]
    blob = json.dumps([e for _, e in sink.events]) + env.model_dump_json()
    assert not any(name in blob for name in AGENT_NAMES)  # never in events or output


async def test_a_raising_event_sink_does_not_fail_the_run(store):
    sink = RaisingSink()
    agent = build(
        ScriptedChat([enhanced(), verdict()]), ScriptedImages([image_bytes()]), store, events=sink
    )

    env = await create(agent)

    assert env.state.status is Status.DONE
    assert sink.attempts >= 6  # every emit was tried, and every failure swallowed


# -- missing_info: resolved by kind from presets, never escalated in auto -----------------------


@pytest.mark.parametrize("mode", ["auto", "manual"])
async def test_preset_kinds_are_resolved_from_defaults_in_both_modes(store, mode):
    missing = [{"kind": "aspect_ratio"}, {"kind": "style"}, {"kind": "size"}]
    llm = ScriptedChat([enhanced(missing_info=missing), verdict()])
    images = ScriptedImages([image_bytes()])

    env = await create(build(llm, images, store), mode=mode)

    assert env.state.status is Status.DONE  # nothing escalated: defaults answered it all
    assert env.output["assumptions"] == [
        "aspect_ratio: 1:1",
        "style: natural, photographic",
        "size: 1024x1024",
    ]
    request = images.requests[0][1]
    assert (request.width, request.height) == (1024, 1024)
    assert request.prompt.endswith("Style: natural, photographic.")


async def test_a_user_chosen_value_is_never_overridden_by_a_default(store):
    missing = [{"kind": "aspect_ratio"}, {"kind": "style"}]
    llm = ScriptedChat([enhanced(missing_info=missing), verdict()])
    images = ScriptedImages([image_bytes()])

    env = await create(
        build(llm, images, store), settings={"aspect_ratio": "9:16", "style": "watercolour"}
    )

    assert env.output["assumptions"] == []
    request = images.requests[0][1]
    assert (request.width, request.height) == (576, 1024)
    assert "Style:" not in request.prompt  # the Enhancer already had the user's style


async def test_question_in_manual_mode_returns_needs_input_without_generating(store):
    question = "Which dog breed should it be?"
    llm = ScriptedChat([enhanced(missing_info=[{"kind": "question", "question": question}])])
    images = ScriptedImages([])
    run_store = RecordingStore()

    env = await create(build(llm, images, store, run_store=run_store), mode="manual")

    assert env.state.status is Status.NEEDS_INPUT
    assert env.output["questions"] == [question]
    assert env.output["image_ids"] == []
    assert images.requests == []  # the harness never talks to the user: it returns upward
    final = RunState.model_validate_json(run_store.snapshots[-1])
    assert [s.status for s in final.plan] == ["done", "skipped", "skipped"]


async def test_question_in_auto_mode_is_never_escalated(store):
    question = "Which dog breed should it be?"
    missing = [{"kind": "question", "question": question}, {"kind": "aspect_ratio"}]
    llm = ScriptedChat([enhanced(missing_info=missing), verdict()])

    env = await create(build(llm, ScriptedImages([image_bytes()]), store), mode="auto")

    assert env.state.status is Status.DONE
    assert f"proceeded without an answer to: {question}" in env.output["assumptions"]
    assert "aspect_ratio: 1:1" in env.output["assumptions"]


async def test_unsupported_aspect_ratio_falls_back_to_the_default(store):
    images = ScriptedImages([image_bytes()])
    env = await create(
        build(ScriptedChat([enhanced(), verdict()]), images, store),
        settings={"aspect_ratio": "7:3"},
    )

    assert env.output["assumptions"] == ["aspect_ratio: 1:1 (7:3 is not supported)"]
    assert (images.requests[0][1].width, images.requests[0][1].height) == (1024, 1024)


# -- quality and shape --------------------------------------------------------------------------


async def test_low_score_delivers_with_the_critique_as_an_issue(store):
    llm = ScriptedChat([enhanced(), verdict(2, critique="No coffee shop visible.")])

    env = await create(build(llm, ScriptedImages([image_bytes()]), store))

    assert env.state.status is Status.DONE_WITH_ISSUES  # TODO(M7): retries come first
    assert env.output["issues"] == ["No coffee shop visible."]
    assert len(env.output["image_ids"]) == 1


async def test_threshold_is_per_plan(store):
    # ultra's threshold is above free's: the same score passes one and not the other.
    score = CFG.policy.quality_threshold["free"]
    assert CFG.policy.quality_threshold["ultra"] > score
    free = await create(
        build(ScriptedChat([enhanced(), verdict(score)]), ScriptedImages([image_bytes()]), store),
        "free",
    )
    ultra = await create(
        build(ScriptedChat([enhanced(), verdict(score)]), ScriptedImages([image_bytes()]), store),
        "ultra",
    )
    assert free.state.status is Status.DONE
    assert ultra.state.status is Status.DONE_WITH_ISSUES


async def test_shape_mismatch_is_ignored_on_a_free_provider(store):
    images = ScriptedImages([image_bytes(1024, 1024)], provider="cloudflare")
    env = await create(
        build(ScriptedChat([enhanced(), verdict()]), images, store),
        settings={"aspect_ratio": "16:9"},
    )

    assert env.state.status is Status.DONE
    assert env.output["issues"] == []
    assert (env.output["width"], env.output["height"]) == (1024, 1024)  # the truth is reported


async def test_shape_mismatch_is_an_issue_on_a_paid_provider(store):
    images = ScriptedImages(
        [image_bytes(1024, 1024)], provider="openrouter", model="google/gemini-2.5-flash-image"
    )
    env = await create(
        build(ScriptedChat([enhanced(), verdict()]), images, store),
        settings={"aspect_ratio": "16:9"},
    )

    assert env.state.status is Status.DONE_WITH_ISSUES
    assert env.output["issues"] == ["requested 16:9, got 1024x1024"]


async def test_matching_shape_on_a_paid_provider_is_fine(store):
    images = ScriptedImages([image_bytes(1024, 576)], provider="openrouter")
    env = await create(
        build(ScriptedChat([enhanced(), verdict()]), images, store),
        settings={"aspect_ratio": "16:9"},
    )
    assert env.state.status is Status.DONE


# -- failures leave through END as envelopes, never exceptions ----------------------------------


async def test_critic_failure_never_delivers_the_unchecked_image(store):
    llm = ScriptedChat([enhanced(), '{"quality_score": 99}', '{"quality_score": "x"}'])

    env = await create(build(llm, ScriptedImages([image_bytes()]), store))

    assert env.state.status is Status.FAILED
    assert env.output["image_ids"] == []
    assert len(await store.orphaned()) == 1  # paid for, never delivered: swept


async def test_creator_refusal_fails_without_calling_the_critic(store):
    refused = ContentRefused(
        ProviderError("cloudflare", "m", "NSFW", status=400, code=3030), "image_create"
    )
    llm = ScriptedChat([enhanced()])  # no critic reply: a critic call would raise

    env = await create(build(llm, ScriptedImages([refused]), store))

    assert env.state.status is Status.FAILED
    assert "content_refused" in (env.state.reason or "")
    assert env.output["image_ids"] == []


async def test_enhancer_failure_makes_no_image_call(store):
    images = ScriptedImages([])
    env = await create(build(ScriptedChat(["not json", "still not json"]), images, store))

    assert env.state.status is Status.FAILED
    assert images.requests == []


async def test_an_unexpected_exception_inside_a_node_becomes_a_failed_envelope(store):
    # A non-gateway exception escapes the Image Creator; the node must not raise past the graph.
    images = ScriptedImages([RuntimeError("bug in a backend adapter")])
    run_store = RecordingStore()

    env = await create(build(ScriptedChat([enhanced()]), images, store, run_store=run_store))

    assert env.state.status is Status.FAILED
    assert env.state.reason == "internal error: RuntimeError"
    final = RunState.model_validate_json(run_store.snapshots[-1])
    assert [s.status for s in final.plan] == ["done", "failed", "skipped"]


async def test_more_than_one_image_is_rejected_until_m8(store):
    env = await create(build(ScriptedChat([]), ScriptedImages([]), store), n=2)
    assert env.output == {"failure": "invalid_input"}


# -- budget: refused BEFORE a generation call, never after --------------------------------------


def _cfg_with_free_neurons(neurons: float) -> Any:
    budgets = dict(CFG.policy.budgets)
    budgets["free"] = budgets["free"].model_copy(update={"neurons_total": neurons})
    return CFG.model_copy(update={"policy": CFG.policy.model_copy(update={"budgets": budgets})})


async def test_budget_is_refused_before_the_generation_call(store):
    est = CFG.policy.step_estimates
    # Enough for enhance; never enough for generate + the critique that must follow it, even
    # if enhance spent nothing. (Actual spend, not the estimate, is what gets charged.)
    usable = est["generate"].neurons + est["critique"].neurons - 1
    assert est["enhance"].neurons <= usable
    cfg = _cfg_with_free_neurons(usable / (1 - CFG.policy.budgets["free"].reserve_fraction))
    images = ScriptedImages([image_bytes()])

    env = await create(build(ScriptedChat([enhanced()]), images, store, cfg=cfg))

    assert env.state.status is Status.FAILED
    assert "budget" in (env.state.reason or "")
    assert images.requests == []  # never paid for an image it could not then check


# -- cancellation at a step boundary (full cancellation is M9) ----------------------------------


async def test_cancel_before_generation_freezes_the_plan_and_spends_nothing_on_images(store):
    run_store = RecordingStore(cancel_when=lambda s: s.step(1).status == "done")
    images = ScriptedImages([image_bytes()])

    env = await create(build(ScriptedChat([enhanced()]), images, store, run_store=run_store))

    assert env.state.status is Status.CANCELLED
    assert images.requests == []
    final = RunState.model_validate_json(run_store.snapshots[-1])
    assert [s.status for s in final.plan] == ["done", "skipped", "skipped"]  # frozen, not cleared
    assert final.cancelled_at_step == 2
    assert final.step(1).output["enhanced_prompt"]  # completed step keeps its output
    assert env.state.spend.neurons == final.step(1).spend.neurons  # only actual spend


async def test_cancel_before_critique_never_delivers_the_unchecked_image(store):
    run_store = RecordingStore(cancel_when=lambda s: s.step(2).status == "done")

    env = await create(
        build(
            ScriptedChat([enhanced()]), ScriptedImages([image_bytes()]), store, run_store=run_store
        )
    )

    assert env.state.status is Status.CANCELLED
    assert env.output["image_ids"] == []
    assert len(await store.orphaned()) == 1
