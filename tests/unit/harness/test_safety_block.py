"""Spec §7.5 / §5.5 / acceptance: `safety_verdict: block` drops that image whatever its score or
the plan tier, no retry path can override it, and when EVERY image is blocked the run ends
REJECTED. The Critic reviews all images in one call, so a block is per image.

Written before the supervisor existed (M5), against the factory it had to provide. Since M6
the run-level half imports the supervisor directly: it can fail, never silently skip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from media_engine.agents.base import AgentConfig
from media_engine.agents.critic.critic import Critic, says_block
from media_engine.agents.image_grand import supervisor
from media_engine.contracts import AuthContext, Goal, Plan, Status
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver
from media_engine.ports import ListEventSink, LocalArtifactStore
from media_engine.schemas import CriticOutput
from media_engine.services.create.loader import PLANS, load_create_config
from tests.unit.harness.fakes import ScriptedChat, ScriptedImages, image_bytes, verdict

CFG = load_create_config()
GOAL = Goal(task="critique", instructions="Review the image.")
SCORES = [0, 5, 10]  # 10 = a "perfect" image: still blocked


def block(score: int = 10, reason: str | None = "graphic violence") -> Any:
    return verdict(score, safety_verdict="block", safety_reason=reason)


def allow(score: int = 9) -> Any:
    return verdict(score)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / ".artifacts")


async def critique(  # type: ignore[no-untyped-def]
    replies: list[Any], store: LocalArtifactStore, plan: Plan = "free", images: int = 1
):
    auth = AuthContext(plan=plan, execution_mode="auto", model_demand="balanced")
    stop = CFG.policy.stopping_condition("critique", plan, "verdict")
    config = AgentConfig.from_policy("critic", CFG.agent("critic"), CriticOutput)
    backend = ScriptedChat(replies)
    agent = Critic(
        config, LLMGateway(backend, chain_resolver(CFG.agents)), ListEventSink(), artifacts=store
    )
    ids = [await store.put(image_bytes(), "image/png") for _ in range(images)]
    env = await agent.run(GOAL, stop, auth, {"image_ids": ids, "prompt": "a cat"})
    return ids, env, backend


# -- Critic level -------------------------------------------------------------------------------


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("score", SCORES)
async def test_a_block_is_final_for_every_plan_and_score(store, plan, score):
    ids, env, backend = await critique([block(score)], store, plan)

    assert env.state.status is Status.DONE  # the step ran; the IMAGE is blocked
    (review,) = env.output["reviews"]
    assert review == {
        "image_id": ids[0],
        "verdict": "block",
        "safety_reason": "graphic violence",
    }  # no score, no rank: a verdict, never something to compare
    assert len(backend.requests) == 1
    assert await store.orphaned() == [ids[0]]


async def test_a_block_drops_that_image_only(store):
    ids, env, _ = await critique([allow(7), block(9), allow(5)], store, images=3)

    by_id = {r["image_id"]: r for r in env.output["reviews"]}
    assert by_id[ids[1]]["verdict"] == "block"
    assert (by_id[ids[0]]["rank"], by_id[ids[2]]["rank"]) == (1, 2)  # ranks skip the blocked
    assert await store.orphaned() == [ids[1]]


async def test_an_unreadable_block_cannot_be_repaired_into_allow(store):
    # The first reply says block but fails validation; the repair says allow. The block wins:
    # nothing in the batch is trusted (it cannot be attributed to an image).
    bad = '{"images": [{"index": 1, "safety_verdict": "block"}]}'
    _, env, backend = await critique([bad, allow(10)], store)

    assert env.state.status is Status.FAILED
    assert len(backend.requests) == 2  # the repair happened, and was overruled
    assert await store.orphaned() == []  # the run orphans unchecked images when it settles


async def test_an_unreadable_block_is_not_retried(store):
    truncated = '{"images": [{"index": 1, "safety_verdict": "block", "safety_reason": "gra'
    _, env, backend = await critique([truncated, truncated, allow(10), allow(10)], store)

    assert env.state.status is Status.FAILED
    assert len(backend.requests) == 2  # the one repair, and no in-step retry after it


async def test_block_without_a_reason_gets_the_default(store):
    _, env, _ = await critique([block(7, reason=None)], store)
    assert env.output["reviews"][0]["safety_reason"] == "safety policy"


async def test_allow_with_low_score_is_not_blocked(store):
    # Low quality is not unsafe: the two never mix.
    _, env, _ = await critique([allow(0)], store)
    (review,) = env.output["reviews"]
    assert review["verdict"] == "allow" and review["quality_score"] == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"safety_verdict": "block", "safety_reason": "gore"}', True),
        ('{"safety_verdict" : " BLOCK ", "safety_reason": "hate", "cri', True),
        ('{"safety_verdict": "Block"', True),
        ('{"safety_verdict": "allow"}', False),
        ('{"quality_score": 2, "critique": "fine"', False),
        ("", False),
    ],
)
def test_says_block(text, expected):
    assert says_block(text) is expected


# -- run level ----------------------------------------------------------------------------------


def enhancer_ok() -> str:
    return json.dumps(
        {
            "subject": "a cat on a sofa",
            "lighting": "window light from the right",
            "detail_anchors": ["grey fur", "a red cushion"],
            "must_have": ["a cat"],
            "missing_info": [],
            "text_wanted": [],
            "summary": "A cat.",
        }
    )


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("score", SCORES)
async def test_run_ends_rejected_on_block_with_no_further_call(store, plan, score):
    # create never regenerates (v2.2); a block must also never lead to any further call

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
