"""ImageGrandAgent: the top of the image side. The Safety Gate runs first, on every path, then
the request is handed to a child.

    [0] code checks (no model, no cost)
    [1] Gate (1 LLM call): safety + route        review -> safety reviewer (1 more call)
        block -> REJECTED before any spend    Gate or reviewer undecided -> FAILED (fail closed)
    then pick a child: an explicit operation -> generation (plain code, no call);
          ambiguous text -> the Decider; no answer -> generation (the safe default)
            generation -> Image Generation Agent    design -> Designer & Layout Agent (stub)

The Gate decides SAFETY only. Its `route`, `task_type` and `prompt_is_detailed` are recorded on
RunState and read by nothing. The LLM plan path for complex requests is future work, added
with the layout agent.

The Gate's verdict is decided ONCE here, recorded on RunState, and travels with the request on
every route, so no path skips it (IMAGE-CREATE-OPTIMIZE §4.1). It is the only agent that may
end a run on a user-facing decision; nothing below it talks to the user (NEEDS_INPUT and
ESCALATE are returned upward unchanged).

The Gate's (and reviewer's) spend is added to the returned envelope's spend. It is not charged
to the run's budget (the run does not exist yet; both are free Groq calls).

Billing and failure signals (for the backend). Every envelope carries `output["billable"]`:
False for every run that ends before an image step (a step 0 refusal, a Gate or review block,
an undecided Gate or review, an escalation), and for a delegated run whatever the child says.
A run that ends before an image step also carries `output["failure"]`, one of EARLY_FAILURES;
RETRYABLE_FAILURES are outages the user may simply retry, the rest are decisions about the
request itself.

inputs:
    prompt:   str             the user's request (required)
    settings: dict[str, str]  user-chosen values (optional; judged by the Gate as user text)
    operation: str            optional explicit operation (create|edit|merge|restyle)
    n, run_id, user_id        passed through to the child (run_id is required)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, get_args

from media_engine.agents.base import AgentConfig
from media_engine.agents.decider import Decider, decide
from media_engine.agents.delegate import Delegate
from media_engine.agents.image_grand.designer import designer_layout_agent
from media_engine.agents.image_grand.gate import Gate, SafetyReviewer
from media_engine.agents.image_grand.limiter import review_limiter
from media_engine.contracts import (
    AuthContext,
    Envelope,
    EnvelopeState,
    Goal,
    Spend,
    Status,
    StoppingCondition,
)
from media_engine.core.logging import get_logger
from media_engine.gateway.llm_gateway import LLMGateway
from media_engine.guardrails.code_checks import check_prompt, check_settings
from media_engine.harness import trace as tracing
from media_engine.harness.run_state import GateRecord
from media_engine.harness.trace import Trace
from media_engine.ports import EventSink
from media_engine.schemas import GateDecision, SafetyReview
from media_engine.services.create.loader import CreateConfig

log = get_logger("create")

# Why a run ended before any image step (`output["failure"]`).
EarlyFailure = Literal[
    "invalid_input",  # bad arguments or a malformed prompt (step 0 format rules)
    "code_check_refused",  # step 0 refused the content (e.g. an injection phrase)
    "safety_block",  # the Gate (or its reviewer) blocked the request
    "gate_undecided",  # the Gate could not decide: an outage, not a verdict
    "review_undecided",  # the safety reviewer could not decide: an outage, not a verdict
    "review_rate_limited",  # no slot under the reviewer's rate limit within the wait
]
EARLY_FAILURES: frozenset[str] = frozenset(get_args(EarlyFailure))
RETRYABLE_FAILURES: frozenset[str] = frozenset(
    {"gate_undecided", "review_undecided", "review_rate_limited"}
)

Child = Literal["generation", "design"]
CHILDREN: tuple[str, ...] = get_args(Child)
DEFAULT_CHILD: Child = "generation"
CHILD_QUESTION = "Does this request ask to generate an image, or to design a layout with text?"

REFUSED_SUMMARY = "This request can't be made into an image."
UNCHECKED_SUMMARY = "The request could not be checked right now. Please try again."
# Step 0 rules that are an input-format problem (FAILED), not a refusal of the content.
_FORMAT_RULES = frozenset({"empty", "too_long", "control_chars"})


def invalid(reason: str) -> Envelope:
    """Bad arguments: FAILED before any step, nothing to bill."""
    return Envelope(
        output={"failure": "invalid_input", "billable": False},
        summary="The request could not be started.",
        state=EnvelopeState(status=Status.FAILED, spend=Spend(), reason=reason),
    )


def _ended(
    run_id: str,
    status: Status,
    failure: EarlyFailure,
    summary: str,
    reason: str,
    spend: Spend,
    trace: Trace,
) -> Envelope:
    """A run that ends before any image step: nothing generated, nothing to deliver, nothing
    to bill. `failure` tells the backend whether it was a decision or a retryable outage."""
    trace.finish(status.value)
    return Envelope(
        output={"run_id": run_id, "image_ids": [], "failure": failure, "billable": False},
        summary=summary,
        state=EnvelopeState(status=status, spend=spend, reason=reason),
    )


class ImageGrandAgent:
    """Safety Gate, then delegate. Makes no LLM call of its own: the Gate and the reviewer are
    its only calls. `generation` builds the generation child lazily, so a request the Gate
    stops never builds it. `decider` (optional) picks the child for ambiguous text."""

    def __init__(
        self,
        cfg: CreateConfig,
        *,
        llm: LLMGateway,
        events: EventSink,
        trace: Trace,
        generation: Callable[[], Delegate],
        decider: Decider | None = None,
    ) -> None:
        self._cfg = cfg
        self._llm = llm
        self._events = events
        self._trace = trace
        self._generation = generation
        self._decider = decider

    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope:
        cfg, sink, trace = self._cfg, self._events, self._trace
        prompt: str = inputs["prompt"]
        settings: dict[str, str] = inputs.get("settings") or {}
        run_id: str = inputs["run_id"]

        # 0. code checks: no model, no cost. A refusal never reaches the Gate.
        checked = check_prompt(prompt)
        if checked.ok:
            checked = check_settings(settings)
        if not checked.ok:
            trace.decide("route", f"code check refused the request: {checked.rule}")
            log.info("create.code_check_refused", run_id=run_id, rule=checked.rule)
            if checked.rule in _FORMAT_RULES:
                trace.finish(Status.FAILED.value)
                return invalid(checked.reason or "invalid prompt")
            return _ended(
                run_id,
                Status.REJECTED,
                "code_check_refused",
                REFUSED_SUMMARY,
                "the request was refused before any model call",
                Spend(),
                trace,
            )

        # 1. Gate: safety + route, decided once, before any spend. It fails CLOSED.
        gate = Gate(
            AgentConfig.from_policy("gate", cfg.agent("gate"), GateDecision), self._llm, sink
        )
        gate_stop = cfg.policy.stopping_condition(
            "route", auth.plan, "a safety verdict and a route"
        )
        with tracing.scope(trace, "route", 0):  # the Gate's call is seq 1 of the run's trace
            gated = await gate.run(
                Goal(task="gate", instructions="Check the request and decide how to handle it."),
                gate_stop,
                auth,
                {"prompt": prompt, "settings": settings},  # settings are user text too
            )
        spend = gated.state.spend
        if gated.state.status is not Status.DONE:
            trace.decide(
                "route", f"gate UNDECIDED, run stopped (fail closed): {gated.state.reason}"
            )
            return _ended(
                run_id,
                Status.FAILED,
                "gate_undecided",
                UNCHECKED_SUMMARY,
                gated.state.reason or "gate failed",
                spend,
                trace,
            )
        verdict = GateRecord.model_validate(gated.output)
        trace.decide(
            "route",
            f"gate: safety = {verdict.safety}"
            + (f" ({', '.join(verdict.safety_categories)})" if verdict.safety_categories else "")
            + f"; route = {verdict.route}; task_type = {verdict.task_type}; "
            f"prompt_is_detailed = {verdict.prompt_is_detailed}",
        )

        if verdict.safety == "review":  # rare: the 3 req/min safeguard model, never the hot path
            limits = cfg.policy.safety_review
            reviewer = SafetyReviewer(
                AgentConfig.from_policy(
                    "safety_reviewer", cfg.agent("safety_reviewer"), SafetyReview
                ),
                self._llm,
                sink,
                limiter=review_limiter(limits.max_per_period, limits.period_s),
                wait_s=limits.wait_s,
            )
            with tracing.scope(trace, "route", 0):
                reviewed = await reviewer.run(
                    Goal(task="review", instructions="Decide whether the request may be made."),
                    gate_stop,
                    auth,
                    {
                        "prompt": prompt,
                        "settings": settings,
                        "safety_categories": verdict.safety_categories,
                    },
                )
            spend = spend.plus(reviewed.state.spend)
            if reviewed.state.status is not Status.DONE:
                trace.decide(
                    "route", f"safety review UNDECIDED, run stopped: {reviewed.state.reason}"
                )
                limited = reviewed.output.get("failure") == "rate_limited"
                return _ended(
                    run_id,
                    Status.FAILED,
                    "review_rate_limited" if limited else "review_undecided",
                    UNCHECKED_SUMMARY,
                    reviewed.state.reason or "safety review failed",
                    spend,
                    trace,
                )
            verdict = verdict.model_copy(
                update={
                    "safety": reviewed.output["safety"],
                    "safety_categories": reviewed.output.get("safety_categories")
                    or verdict.safety_categories,
                    "reviewed": True,
                }
            )
            trace.decide("route", f"safety review: {verdict.safety}")

        if verdict.safety == "block":
            log.info("create.gate_blocked", run_id=run_id, categories=verdict.safety_categories)
            return _ended(
                run_id,
                Status.REJECTED,
                "safety_block",
                REFUSED_SUMMARY,
                "the request did not pass the safety check",
                spend,
                trace,
            )
        # The Gate's route, task_type and prompt_is_detailed travel with the verdict and are
        # stored on RunState, but nothing reads them: routing is code (an explicit operation)
        # or the Decider, and the Brief always runs.

        # 2. hand over: the verdict travels with the request to every child
        child = await self.route(inputs)
        log.info("create.routed", run_id=run_id, child=child)
        delegate = self._generation() if child == "generation" else designer_layout_agent()
        env = await delegate.run(goal, stop, auth, {**inputs, "gate": verdict.model_dump()})
        env.state.spend = env.state.spend.plus(spend)
        trace.finish(env.state.status.value)  # closes the developer trace (status, overall time)
        return env

    async def route(self, inputs: dict[str, Any]) -> Child:
        """Pick the child. An explicit operation is a generation operation: plain code, no
        call. Only ambiguous text asks the Decider; no answer means the safe default."""
        if inputs.get("operation") is not None:
            self._trace.decide("route", "child generation: explicit operation, routed by code")
            return "generation"
        answer = await decide(
            self._decider, CHILD_QUESTION, str(inputs.get("prompt", "")), CHILDREN
        )
        if answer is None:
            self._trace.decide("route", f"child {DEFAULT_CHILD}: no decision, default")
            return DEFAULT_CHILD
        self._trace.decide("route", f"child {answer}: classified")
        return answer  # type: ignore[return-value]  # one of CHILDREN (decide checks)


__all__ = [
    "CHILDREN",
    "EARLY_FAILURES",
    "RETRYABLE_FAILURES",
    "EarlyFailure",
    "ImageGrandAgent",
    "invalid",
]
