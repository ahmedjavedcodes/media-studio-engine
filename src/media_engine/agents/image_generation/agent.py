"""ImageGenerationAgent: routes one request to an operation, then delegates. No model call.

    explicit operation (a button, a slash command)  -> plain code, no call
    ambiguous free text  -> the Decider; no answer -> create (the safe default)
    create -> image-create-agent (REAL)   edit / merge / restyle -> stubs (ESCALATE)

While edit, merge and restyle are stubs, an AMBIGUOUS request classified as one of them runs as
create and the choice is logged (review round 2: a text-only "change the background to blue"
is still served as a new image). `routing.ambiguous_to_stubs` in harness.yaml flips this once
the real operations exist. An EXPLICIT operation always goes where it was sent.

The child's envelope is returned unchanged, so NEEDS_INPUT passes up to the Image Grand Agent.

inputs: everything the create agent takes, plus `operation` (optional, one of OPERATIONS).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, get_args

from media_engine.agents.decider import Decider, decide
from media_engine.agents.delegate import Delegate
from media_engine.agents.stub import StubAgent
from media_engine.contracts import AuthContext, Envelope, Goal, StoppingCondition
from media_engine.core.logging import get_logger
from media_engine.harness.trace import Trace

log = get_logger("agents.image_generation")

Operation = Literal["create", "edit", "merge", "restyle"]
OPERATIONS: tuple[str, ...] = get_args(Operation)
DEFAULT_OPERATION: Operation = "create"
QUESTION = "Which image operation does this request ask for?"

STUBS: dict[str, StubAgent] = {
    "edit": StubAgent("edit", "Editing an image is not available yet."),
    "merge": StubAgent("merge", "Merging images is not available yet."),
    "restyle": StubAgent("restyle", "Restyling an image is not available yet."),
}


class ImageGenerationAgent:
    def __init__(
        self,
        *,
        create: Callable[[], Delegate],
        trace: Trace,
        decider: Decider | None = None,
        ambiguous_to_stubs: bool = False,
    ) -> None:
        self._create = create
        self._trace = trace
        self._decider = decider
        self._ambiguous_to_stubs = ambiguous_to_stubs

    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope:
        operation = await self.route(inputs)
        child = self._create() if operation == "create" else STUBS[operation]
        return await child.run(goal, stop, auth, inputs)

    async def route(self, inputs: dict[str, Any]) -> Operation:
        explicit = inputs.get("operation")
        if explicit is not None:
            self._trace.decide("route", f"operation {explicit}: explicit, routed by code")
            return explicit  # type: ignore[no-any-return]  # validated by the entry point
        answer = await decide(self._decider, QUESTION, str(inputs.get("prompt", "")), OPERATIONS)
        if answer is None:
            self._trace.decide("route", f"operation {DEFAULT_OPERATION}: no decision, default")
            return DEFAULT_OPERATION
        if answer != "create" and not self._ambiguous_to_stubs:
            log.info("generation.ambiguous_stub_as_create", classified=answer)
            self._trace.decide(
                "route", f"operation classified {answer}; not built yet, so run as create"
            )
            return "create"
        self._trace.decide("route", f"operation {answer}: classified")
        return answer  # type: ignore[return-value]  # one of OPERATIONS (decide checks)


__all__ = ["OPERATIONS", "ImageGenerationAgent", "Operation"]
