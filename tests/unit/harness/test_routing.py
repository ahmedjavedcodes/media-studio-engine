"""The agent hierarchy (ARCHITECTURE-SIMPLIFY-SPEC): Image Grand Agent -> Safety Gate first ->
a child (Image Generation Agent | Designer stub) -> an operation (create | edit/merge/restyle
stubs). Explicit operations route by plain code with no call; ambiguous text asks the Decider
and takes the safe default without an answer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from media_engine.contracts import Status
from media_engine.ports import LocalArtifactStore
from media_engine.services.create.entry import create_images
from media_engine.services.create.loader import RoutingPolicy
from tests.unit.harness.fakes import ScriptedChat, ScriptedImages, image_bytes
from tests.unit.harness.test_entry import ROUTE_SIMPLE, Backend, gate_reply
from tests.unit.harness.test_linear import CFG, enhanced, verdict
from tests.unit.harness.test_supervisor import auth

PROMPT = "a cozy coffee shop"


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / ".artifacts")


class FakeDecider:
    """Answers the child question with `child` and the operation question with `operation`,
    and records every question it was asked."""

    def __init__(self, child: str | None = None, operation: str | None = None) -> None:
        self.child, self.operation = child, operation
        self.asked: list[tuple[str, ...]] = []

    async def choose(self, question: str, text: str, options: Sequence[str]) -> str | None:
        self.asked.append(tuple(options))
        return self.child if "design" in options else self.operation


def happy_llm() -> ScriptedChat:
    return ScriptedChat([ROUTE_SIMPLE, enhanced(), verdict(9)])


async def run(store, llm, images=None, **kwargs):  # type: ignore[no-untyped-def]
    images = images if images is not None else ScriptedImages([image_bytes()])
    return await create_images(PROMPT, gateway=Backend(llm, images), artifacts=store, **kwargs)


# -- explicit operations: plain code, no routing call ------------------------------------------


@pytest.mark.parametrize("operation", ["edit", "merge", "restyle"])
async def test_an_explicit_stub_operation_escalates_with_its_own_reason(store, operation):
    llm, images, decider = ScriptedChat([ROUTE_SIMPLE]), ScriptedImages([]), FakeDecider()

    env = await run(store, llm, images, operation=operation, decider=decider)

    assert env.state.status is Status.ESCALATE
    assert env.state.reason == f"{operation} not implemented"
    assert env.output["billable"] is False and env.output["image_ids"] == []
    assert env.state.spend.llm_calls == 1  # the Gate ran first; nothing else did
    assert decider.asked == [] and images.requests == []


async def test_an_explicit_create_runs_the_pipeline_with_three_llm_calls_and_no_routing_call(
    store,
):
    decider = FakeDecider(child="design", operation="edit")  # would mis-route if it were asked

    env = await run(store, happy_llm(), operation="create", decider=decider)

    assert env.state.status is Status.DONE and env.state.spend.llm_calls == 3
    assert decider.asked == []


async def test_an_unknown_operation_is_invalid_before_any_model_call(store):
    llm = ScriptedChat([])

    env = await run(store, llm, ScriptedImages([]), operation="poster")

    assert env.state.status is Status.FAILED
    assert env.output == {"failure": "invalid_input", "billable": False}
    assert llm.requests == []


# -- the Safety Gate is first on every path -----------------------------------------------------


@pytest.mark.parametrize("operation", [None, "create", "edit"])
async def test_a_blocked_request_never_reaches_routing(store, operation):
    decider = FakeDecider(child="design", operation="edit")

    env = await run(
        store, ScriptedChat([gate_reply("block")]), operation=operation, decider=decider
    )

    assert env.state.status is Status.REJECTED and decider.asked == []


# -- ambiguous text: the Decider, then the safe default ----------------------------------------


async def test_without_a_decider_ambiguous_text_is_a_create(store):
    env = await run(store, happy_llm())

    assert env.state.status is Status.DONE and env.state.spend.llm_calls == 3


@pytest.mark.parametrize("answer", [None, "something else"])
async def test_no_usable_answer_takes_the_safe_default_at_both_levels(store, answer):
    decider = FakeDecider(child=answer, operation=answer)

    env = await run(store, happy_llm(), decider=decider)

    assert env.state.status is Status.DONE
    assert len(decider.asked) == 2  # child, then operation: Decider calls are not LLM calls
    assert env.state.spend.llm_calls == 3


async def test_the_grand_agent_routes_design_to_the_designer_stub(store):
    decider = FakeDecider(child="design")
    images = ScriptedImages([])

    env = await run(store, ScriptedChat([ROUTE_SIMPLE]), images, decider=decider)

    assert env.state.status is Status.ESCALATE
    assert env.state.reason == "design not implemented"
    assert env.output["billable"] is False and images.requests == []


async def test_ambiguous_text_classified_as_a_stub_runs_as_create_while_stubs_exist(store):
    decider = FakeDecider(child="generation", operation="edit")

    env = await run(store, happy_llm(), decider=decider)

    assert env.state.status is Status.DONE  # review round 2: still served as a new image


async def test_the_flag_sends_ambiguous_stub_operations_to_the_stub(store):
    policy = CFG.policy.model_copy(update={"routing": RoutingPolicy(ambiguous_to_stubs=True)})
    cfg = CFG.model_copy(update={"policy": policy})
    decider = FakeDecider(child="generation", operation="merge")

    env = await run(store, ScriptedChat([ROUTE_SIMPLE]), decider=decider, cfg=cfg)

    assert env.state.status is Status.ESCALATE and env.state.reason == "merge not implemented"


# -- what passes up unchanged ------------------------------------------------------------------


async def test_needs_input_passes_up_to_the_caller_unchanged(store):
    question = "Which dog breed should it be?"
    llm = ScriptedChat(
        [ROUTE_SIMPLE, enhanced(missing_info=[{"kind": "question", "question": question}])]
    )
    images = ScriptedImages([])

    env = await run(store, llm, images, auth=auth(mode="manual"), operation="create")

    assert env.state.status is Status.NEEDS_INPUT
    assert env.output["questions"] == [question] and images.requests == []


async def test_the_create_envelope_output_shape_is_unchanged(store):
    env = await run(store, happy_llm(), operation="create")

    assert set(env.output) == {
        "run_id",
        "image_ids",
        "documents",
        "quality_score",
        "issues",
        "assumptions",
        "text_wanted",
        "width",
        "height",
        "billable",
    }
    (document,) = env.output["documents"]
    assert [layer["type"] for layer in document["layers"]] == ["image"]


# -- the Brief always runs ---------------------------------------------------------------------


async def test_the_brief_runs_even_when_the_gate_calls_the_prompt_detailed(store):
    detailed = (
        "Wildlife photograph of a red fox mid-stride through deep powder snow in a spruce "
        "forest, low winter sun backlighting the fur, rim light on the guard hairs, paw "
        "tracks in the foreground, 200mm telephoto, shallow depth of field"
    )
    llm = ScriptedChat([gate_reply(prompt_is_detailed=True), enhanced(), verdict(9)])

    env = await create_images(
        detailed,
        gateway=Backend(llm, ScriptedImages([image_bytes()])),
        artifacts=store,
    )

    assert env.state.status is Status.DONE and env.state.spend.llm_calls == 3
    # Gate, then the Brief (its own chain), then the Critic: the flag skips nothing.
    assert [task for task, _ in llm.requests] == ["fast_llm", "reasoning", "vision"]
