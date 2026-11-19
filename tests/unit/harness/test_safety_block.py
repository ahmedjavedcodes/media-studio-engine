"""Spec §7.5 / acceptance: `safety_verdict: block` ends the run as REJECTED no matter the score,
plan tier or retries left, and no retry path can override it.

Written before the supervisor exists (M5). The Critic-level half runs now. The run-level half
skips until `media_engine.agents.image_grand.supervisor` exists (M6), and fixes the factory the
supervisor must provide. M6: remove the importorskip so this can never be silently skipped."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from media_engine.agents.base import AgentConfig
from media_engine.agents.critic.critic import Critic, reply_block_reason
from media_engine.contracts import AuthContext, Goal, Plan, Status
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver
from media_engine.ports import ListEventSink, LocalArtifactStore
from media_engine.schemas import CriticOutput
from media_engine.services.create.loader import PLANS, load_create_config
from tests.unit.harness.fakes import ScriptedChat, ScriptedImages, image_bytes

CFG = load_create_config()
GOAL = Goal(task="critique", instructions="Review the image.")
SCORES = [0, 5, 10]  # 10 = a "perfect" image: still blocked


def block(score: int | None = 10, reason: str | None = "graphic violence") -> str:
    body: dict[str, Any] = {
        "safety_verdict": "block",
        "safety_reason": reason,
        "critique": "x",
        "summary": "x",
    }
    if score is not None:
        body["quality_score"] = score
    return json.dumps(body)


def allow(score: int = 9) -> str:
    return json.dumps(
        {"quality_score": score, "safety_verdict": "allow", "critique": "fine", "summary": "ok"}
    )


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / ".artifacts")


async def critique(  # type: ignore[no-untyped-def]
    replies: list[str], store: LocalArtifactStore, plan: Plan = "free", max_retries: int = 2
):
    auth = AuthContext(plan=plan, execution_mode="auto", model_demand="balanced")
    stop = CFG.policy.stopping_condition("critique", plan, "verdict").model_copy(
        update={"max_retries": max_retries}
    )
    config = AgentConfig.from_policy("critic", CFG.agent("critic"), CriticOutput)
    backend = ScriptedChat(replies)
    agent = Critic(
        config, LLMGateway(backend, chain_resolver(CFG.agents)), ListEventSink(), artifacts=store
    )
    image_id = await store.put(image_bytes(), "image/png")
    env = await agent.run(GOAL, stop, auth, {"image_id": image_id, "prompt": "a cat"})
    return image_id, env, backend


# -- Critic level (runs now) --------------------------------------------------------------------


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("max_retries", [0, 2])
async def test_block_is_rejected_for_every_plan_score_and_retry_budget(
    store, plan, score, max_retries
):
    image_id, env, backend = await critique([block(score)], store, plan, max_retries)

    assert env.state.status is Status.REJECTED
    assert env.output == {
        "blocked_image_id": image_id,
        "safety_verdict": "block",
        "safety_reason": "graphic violence",
    }
    assert "image_id" not in env.output  # never readable as a deliverable
    assert "quality_score" not in env.output  # a verdict, not a score: nothing to compare
    assert len(backend.requests) == 1


async def test_an_invalid_block_reply_cannot_be_repaired_into_allow(store):
    # First reply says block but fails validation (no quality_score) -> the base repairs, and
    # the repair says allow with a high score. Block must still win.
    _, env, backend = await critique([block(score=None), allow(10)], store)

    assert env.state.status is Status.REJECTED
    assert len(backend.requests) == 2  # the repair happened, and was overruled


async def test_an_unparseable_block_on_the_repair_still_rejects(store):
    # Repair reply says block in different case and is missing fields.
    _, env, _ = await critique(['{"quality_score": 99}', '{"safety_verdict": "BLOCK"}'], store)

    assert env.state.status is Status.REJECTED
    assert env.output["safety_reason"] == "safety policy"  # default when none given


async def test_block_without_a_reason_is_still_rejected(store):
    _, env, _ = await critique([block(score=7, reason=None)], store)
    assert env.state.status is Status.REJECTED
    assert env.output["safety_reason"] == "safety policy"


async def test_allow_with_low_score_is_not_rejected(store):
    # Low quality is not unsafe: the two never mix.
    _, env, _ = await critique([allow(0)], store)
    assert env.state.status is Status.DONE
    assert env.output["quality_score"] == 0


# -- run level (activates with the M6 supervisor) ---------------------------------------------


def enhancer_ok() -> str:
    return json.dumps(
        {
            "enhanced_prompt": "A cat on a sofa.",
            "must_have": ["a cat"],
            "missing_info": [],
            "text_wanted": [],
            "summary": "A cat.",
        }
    )


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("score", SCORES)
async def test_run_ends_rejected_on_block_with_retries_left(store, plan, score):
    supervisor = pytest.importorskip("media_engine.agents.image_grand.supervisor")
    assert CFG.policy.max_retries[plan] >= 1  # retries are available, and must not be used

    llm = ScriptedChat([enhancer_ok(), block(score)])  # an extra LLM call raises
    images = ScriptedImages([image_bytes()])  # an extra (retry) image call raises
    auth = AuthContext(plan=plan, execution_mode="auto", model_demand="balanced")
    agent = supervisor.create_image_grand_agent(
        CFG,
        llm=LLMGateway(llm, chain_resolver(CFG.agents)),
        images=images,
        artifacts=store,
        events=ListEventSink(),
    )
    stop = CFG.policy.stopping_condition("supervise", plan, "finished images")

    env = await agent.run(
        Goal(task="create", instructions="make it"), stop, auth, {"prompt": "a cat"}
    )

    assert env.state.status is Status.REJECTED
    assert not env.output.get("image_ids")
    assert len(images.requests) == 1  # no retry image after the block
    assert llm.replies == []  # enhancer + critic, and no lever-choice call


# -- truncated replies (cut off at the token cap mid-JSON) --------------------------------------


async def test_a_truncated_block_cannot_be_repaired_into_allow(store):
    truncated = '{"quality_score": 3, "safety_verdict": "block", "safety_reason": "graphic vio'
    _, env, backend = await critique([truncated, allow(10)], store)

    assert env.state.status is Status.REJECTED
    assert env.output["safety_reason"] == "safety policy"  # reason was cut off: default
    assert len(backend.requests) == 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # decodable JSON: its verdict decides
        ('{"safety_verdict": "block", "safety_reason": "gore"}', "gore"),
        ('{"safety_verdict": "allow"}', None),
        # a decoded allow is never overruled by stray text elsewhere in the reply
        ('note: "safety_verdict": "block"\n{"safety_verdict": "allow"}', None),
        # truncated: text fallback, case and spacing tolerant, reason only if it closed
        (
            '{"quality_score": 9, "safety_verdict" : " BLOCK ", "safety_reason": "hate", "cri',
            "hate",
        ),
        ('{"safety_verdict": "block", "safety_reason": "gra', "safety policy"),
        ('{"safety_verdict": "Block"', "safety policy"),
        # JSON without a verdict + a verdict in the text: fail closed
        ('{"quality_score": 2} "safety_verdict": "block"', "safety policy"),
        # nothing about a verdict at all
        ('{"quality_score": 2, "critique": "fine"', None),
        ("", None),
    ],
)
def test_reply_block_reason(text, expected):
    assert reply_block_reason(text) == expected
