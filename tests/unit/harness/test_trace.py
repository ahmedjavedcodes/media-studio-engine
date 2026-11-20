"""The developer trace: one record per model call, a decisions log, step timing. It is a SEPARATE
channel: the Envelope, the events and the public labels must not change, no image bytes may appear
in it, and capturing it adds no model call."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from media_engine.gateway.errors import AllProvidersFailed
from media_engine.gateway.types import Attempt
from media_engine.harness import trace as tracing
from media_engine.harness.run_state import RunState
from media_engine.harness.trace import Trace
from media_engine.ports import InMemoryRunStore, ListEventSink, LocalArtifactStore
from media_engine.services.create.entry import create_images
from tests.unit.harness.fakes import ScriptedChat, ScriptedImages, image_bytes
from tests.unit.harness.test_branches import FlagStore, HangingImages, fast_poll, run
from tests.unit.harness.test_entry import ROUTE_SIMPLE, Backend
from tests.unit.harness.test_linear import (
    CFG,
    RecordingStore,
    build,
    create,
    enhanced,
    final_state,
    red,
    verdict,
)
from tests.unit.harness.test_supervisor import AGENT_NAMES


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / ".artifacts")


class RichChat(ScriptedChat):
    """ScriptedChat that can also report what a real gateway reports: a fallback, the chain
    walk, usage details, reasoning (or no usage at all)."""

    def __init__(self, replies: Any, **extra: Any) -> None:
        super().__init__(replies)
        self.extra = extra

    async def chat(self, task: str, request: Any) -> Any:
        result = await super().chat(task, request)
        update: dict[str, Any] = {}
        if "usage" in self.extra:
            update["usage"] = self.extra["usage"]
        if "reasoning" in self.extra:
            update["reasoning"] = self.extra["reasoning"]
        return dataclasses.replace(
            result,
            output=result.output.model_copy(update=update),
            attempts=self.extra.get("attempts", []),
            is_fallback=self.extra.get("fallback", False),
        )


# -- one record per model call ----------------------------------------------------------------


async def test_a_first_pass_records_every_call_in_order_with_what_served_it(store):
    run_store = RecordingStore()
    agent = build(
        ScriptedChat([enhanced(), verdict(9)]), ScriptedImages([red()]), store, run_store=run_store
    )

    await create(agent, "free")

    calls = final_state(run_store).calls
    assert [(c.seq, c.step, c.role, c.kind) for c in calls] == [
        (1, "enhance", "prompt_enhancer", "llm"),
        (2, "generate", "image_creator", "image"),
        (3, "critique", "critic", "llm"),
    ]
    enhancer, creator, critic = calls
    assert (enhancer.provider, enhancer.model) == ("cloudflare", "@cf/openai/gpt-oss-20b")
    assert enhancer.chain == CFG.agent("prompt_enhancer").chain_for("free")
    assert (enhancer.prompt_tokens, enhancer.completion_tokens, enhancer.total_tokens) == (
        60,
        40,
        100,
    )
    assert enhancer.neurons == 2.5 and enhancer.tokens_estimated is False
    assert enhancer.reasoning_text is None and enhancer.reasoning_tokens is None
    assert enhancer.reasoning_effort == CFG.agent("prompt_enhancer").reasoning_effort
    assert enhancer.was_repair is False and enhancer.failure is None
    assert "a cozy coffee shop" in enhancer.request_text and "enhanced_prompt" in (
        enhancer.reply_text or ""
    )
    assert (critic.attempt, critic.branch) == (1, 0)
    assert creator.provider == "cloudflare" and creator.model.endswith("flux-1-schnell")


async def test_an_image_call_records_size_seed_and_file_details_but_no_bytes(store):
    run_store = RecordingStore()
    agent = build(
        ScriptedChat([enhanced(), verdict(9)]),
        ScriptedImages([image_bytes(640, 480)]),
        store,
        run_store=run_store,
    )

    await create(agent, "free", settings={"aspect_ratio": "16:9"})

    image = next(c for c in final_state(run_store).calls if c.kind == "image").image
    assert image is not None
    assert (image.width_requested, image.height_requested) == (1024, 576)
    assert (image.width, image.height) == (640, 480)  # what came back
    assert image.seed == 100 and image.seed_honoured is False and image.seed_sent is False
    assert image.mime == "image/png" and image.bytes and image.bytes > 0


async def test_no_image_data_ever_reaches_the_trace(store):
    run_store = RecordingStore()
    trace = Trace()
    agent = build(
        ScriptedChat([enhanced(), verdict(9)]),
        ScriptedImages([red()]),
        store,
        run_store=run_store,
        trace=trace,
    )

    await create(agent, "free")

    blob = json.dumps(trace.snapshot()) + run_store.snapshots[-1]
    assert "base64" not in blob and "data:image" not in blob
    critic = next(c for c in final_state(run_store).calls if c.role == "critic")
    assert tracing.OMITTED in critic.request_text  # the image the Critic was sent, replaced


def test_redaction_removes_data_urls_and_long_base64():
    assert (
        tracing.redact("see data:image/jpeg;base64,AAAA/BBBB== here")
        == f"see {tracing.OMITTED} here"
    )
    assert tracing.OMITTED in (tracing.redact("x" * 500) or "")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "review it"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
            ],
        }
    ]
    text = tracing.messages_text(messages)
    assert "review it" in text and "QUJD" not in text and tracing.OMITTED in text


async def test_the_one_repair_call_is_flagged_and_the_unusable_reply_is_marked(store):
    run_store = RecordingStore()
    llm = ScriptedChat(["not json", enhanced(), verdict(9)])

    await create(build(llm, ScriptedImages([red()]), store, run_store=run_store), "free")

    first, repair = final_state(run_store).calls[:2]
    assert (first.was_repair, first.failure) == (False, "invalid_output")
    assert (repair.was_repair, repair.failure) == (True, None)


async def test_a_gateway_failure_is_recorded_with_its_kind(store):
    run_store = RecordingStore()
    images = ScriptedImages([AllProvidersFailed("image_create", [])])

    await create(build(ScriptedChat([enhanced()]), images, store, run_store=run_store), "pro")

    image_call = next(c for c in final_state(run_store).calls if c.kind == "image")
    assert image_call.failure == "no_provider" and image_call.model is None


async def test_fallback_gateway_attempts_reasoning_and_estimated_tokens_are_reported(store):
    walk = [
        Attempt(
            "cloudflare", "@cf/openai/gpt-oss-20b", "failed", reason="rate limited", status=429
        ),
        Attempt("groq", "openai/gpt-oss-20b", "success", calls=1),
    ]
    llm = RichChat(
        [enhanced(), verdict(9)],
        usage={},  # the provider sent no usage: estimate
        attempts=walk,
        fallback=True,
        reasoning="I should list the requirements.",
    )
    run_store = RecordingStore()

    await create(build(llm, ScriptedImages([red()]), store, run_store=run_store), "free")

    call = final_state(run_store).calls[0]
    assert call.was_fallback is True
    assert [(a.provider, a.outcome) for a in call.attempts] == [
        ("cloudflare", "rate_limited"),
        ("groq", "ok"),
    ]
    assert call.tokens_estimated is True and call.total_tokens > 0
    assert call.reasoning_text == "I should list the requirements."


async def test_reasoning_tokens_are_reported_when_the_provider_sends_them(store):
    usage = {
        "prompt_tokens": 50,
        "completion_tokens": 90,
        "total_tokens": 140,
        "completion_tokens_details": {"reasoning_tokens": 64},
    }
    run_store = RecordingStore()
    llm = RichChat([enhanced(), verdict(9)], usage=usage)

    await create(build(llm, ScriptedImages([red()]), store, run_store=run_store), "free")

    assert final_state(run_store).calls[0].reasoning_tokens == 64


# -- decisions, timing, orphans ---------------------------------------------------------------


async def test_decisions_cover_guards_enhancement_reasks_and_ranking(store):
    run_store = RecordingStore()
    llm = ScriptedChat([enhanced(), "junk", "junk", verdict(9)])

    await create(build(llm, ScriptedImages([red()]), store, run_store=run_store), "free")

    state = final_state(run_store)
    text = "\n".join(f"{d.step}: {d.text}" for d in state.decisions)
    assert "guard before generate [step]: cancel not requested; budget ok: needs" in text
    assert "guard before generate [branch 0]:" in text
    assert "resolved prompt:" in text and "must_have:" in text
    assert "critic re-ask: the same images are checked again" in text
    assert "critic re-asks 1 of" in text
    assert "passed through at no cost" in text  # enhance and generate were not redone
    assert "ranking (best first):" in text and "score 9" in text
    assert state.supervisor_spend.llm_calls == 0  # the supervisor makes no call of its own


async def test_steps_and_the_run_are_timed(store):
    run_store = RecordingStore()
    trace = Trace()

    await create(
        build(
            ScriptedChat([enhanced(), verdict(9)]),
            ScriptedImages([red()]),
            store,
            run_store=run_store,
            trace=trace,
        ),
        "free",
    )

    state = final_state(run_store)
    assert all(s.started_at and s.finished_at and s.duration_ms >= 0 for s in state.plan)
    assert state.finished_at is not None
    snap = trace.snapshot()
    assert snap["wall_ms"] is not None and snap["wall_ms"] >= 0 and snap["status"] == "done"


async def test_the_snapshot_has_everything_the_inspector_shows(store):
    trace = Trace()
    await create(
        build(ScriptedChat([enhanced(), verdict(9)]), ScriptedImages([red()]), store, trace=trace),
        "free",
    )

    snap = trace.snapshot()
    assert {
        "calls",
        "decisions",
        "steps",
        "attempts",
        "budget",
        "supervisor_spend",
        "retries_used",
        "critic_reasks_used",
        "orphaned",
        "status",
        "totals",
        "meta",
        "wall_ms",
    } <= set(snap)
    assert snap["totals"]["llm_calls"] == 2 and snap["totals"]["image_calls"] == 1
    assert snap["meta"]["critic_reask_cap"] == CFG.policy.max_critic_reasks
    assert snap["budget"]["reserve_fraction"] == CFG.policy.budgets["free"].reserve_fraction
    json.dumps(snap)  # JSON-safe


async def test_the_trace_is_live_while_a_step_is_running(store):
    images = HangingImages([], answer=0)
    trace = Trace()
    run_store = FlagStore(lambda: images.hung == 1)
    agent = build(
        ScriptedChat([enhanced()]),
        images,
        store,
        run_store=run_store,
        cfg=fast_poll(CFG),
        trace=trace,
    )
    task = asyncio.ensure_future(create(agent, "free"))
    while images.hung == 0:
        await asyncio.sleep(0.005)

    live = trace.snapshot()  # the run is still going: the image call is in flight

    assert live["status"] == "running" and live["finished_at"] is None
    assert [s["status"] for s in live["steps"]] == ["done", "running", "pending"]
    assert [c["step"] for c in live["calls"]] == ["enhance"]
    await run(task)


# -- the public channels do not change ---------------------------------------------------------


async def test_the_envelope_and_events_never_carry_model_ids_or_agent_names(store):
    sink = ListEventSink()
    llm = ScriptedChat([enhanced(), verdict(9)])
    env = await create(
        build(llm, ScriptedImages([red()]), store, events=sink, trace=Trace()), "free"
    )

    public = json.dumps(env.output) + env.summary + json.dumps([e for _, e in sink.events])
    for secret in (*AGENT_NAMES, "cloudflare", "gpt-oss", "flux", "llama", "@cf/"):
        assert secret not in public, secret


# -- the router is call 1, through create_images ----------------------------------------------


async def test_the_gate_is_seq_1_and_its_verdict_is_a_decision(store):
    run_store = InMemoryRunStore()
    trace = Trace()
    llm = ScriptedChat([ROUTE_SIMPLE, enhanced(), verdict(9)])

    env = await create_images(
        "a cozy coffee shop",
        gateway=Backend(llm, ScriptedImages([red()])),
        artifacts=store,
        run_store=run_store,
        trace=trace,
    )

    state = await run_store.load(env.output["run_id"])
    assert isinstance(state, RunState)
    assert [(c.seq, c.step, c.role) for c in state.calls] == [
        (1, "route", "gate"),
        (2, "enhance", "prompt_enhancer"),
        (3, "generate", "image_creator"),
        (4, "critique", "critic"),
    ]
    first = state.decisions[0]
    assert (
        first.step == "route" and "safety = allow" in first.text and "route = simple" in first.text
    )
    assert state.gate is not None and state.gate.safety == "allow"  # recorded once on RunState
    assert RunState.model_validate_json(state.model_dump_json()).calls == state.calls  # round trip
    assert trace.snapshot()["totals"]["llm_calls"] == 3  # gate, enhancer, critic
    assert env.state.spend.llm_calls == 3  # unchanged: the trace adds no call


async def test_an_undecided_gate_is_a_visible_decision_and_closes_the_trace(store):
    trace = Trace()
    llm = ScriptedChat(["junk", "junk"])

    await create_images(
        "a cozy coffee shop",
        gateway=Backend(llm, ScriptedImages([])),
        artifacts=store,
        trace=trace,
    )

    snap = trace.snapshot()
    assert any("gate UNDECIDED" in d["text"] for d in snap["decisions"])
    assert snap["status"] == "failed" and snap["finished_at"] is not None


async def test_run_state_round_trips_at_every_boundary_with_the_trace(store):
    run_store = RecordingStore()
    llm = ScriptedChat([enhanced(), "junk", "junk", verdict(9)])

    await create(build(llm, ScriptedImages([red()]), store, run_store=run_store), "free")

    for raw in run_store.snapshots:
        assert RunState.model_validate_json(raw).model_dump_json() == raw


# -- a run that never gets a RunState (escalate) still closes its trace ------------------------


async def test_an_escalated_run_closes_its_trace_with_a_status_and_a_wall_time(store):
    from tests.unit.harness.test_entry import ROUTE_COMPLEX

    trace = Trace()
    llm = ScriptedChat([ROUTE_COMPLEX])

    env = await create_images(
        "a 4-panel comic strip",
        gateway=Backend(llm, ScriptedImages([])),
        artifacts=store,
        trace=trace,
    )

    snap = trace.snapshot()
    assert env.state.status.value == "escalate"
    assert snap["status"] == "escalate"  # not "running" forever
    assert snap["finished_at"] is not None and snap["wall_ms"] is not None and snap["wall_ms"] >= 0
    assert [c["role"] for c in snap["calls"]] == ["gate"]
    assert any("ESCALATE" in d["text"] for d in snap["decisions"])


async def test_the_overall_time_starts_before_the_gate_call(store):
    trace = Trace()
    llm = ScriptedChat([ROUTE_SIMPLE, enhanced(), verdict(9)])

    await create_images(
        "a cozy coffee shop",
        gateway=Backend(llm, ScriptedImages([red()])),
        artifacts=store,
        trace=trace,
    )

    snap = trace.snapshot()
    first_call = min(c["ts"] for c in snap["calls"])
    assert snap["created_at"] <= first_call  # so no timeline offset is negative
    assert snap["status"] == "done" and snap["wall_ms"] >= 0
