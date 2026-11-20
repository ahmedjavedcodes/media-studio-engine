"""`create_images()`: the one public entry point of the image-creation harness (spec §3).

    [0] code checks (no model, no cost)
    [1] Gate (1 LLM call): safety + route        review -> safety reviewer (1 more call)
        block -> REJECTED before any spend    Gate or reviewer undecided -> FAILED (fail closed)
    then: simple -> Image Grand Agent  |  complex -> Grand Agent stub (ESCALATE)

It is a library call: no server, no database, no queue. It runs to completion and returns one
Envelope. A first-pass success makes exactly three LLM calls in total (gate, enhancer, critic):
the supervisor adds none. The Gate's verdict is decided ONCE here, recorded on RunState, and
travels with the request on BOTH routes, so no path skips it (IMAGE-CREATE-OPTIMIZE §4.1).

The provider Gateway is built ONCE per process (`get_gateway()`): the circuit breakers, the
availability state and the daily meters live on it, so a second instance would start blind and
re-pay every timeout. Callers (the backend, tests) may pass their own. Everything else is cheap
and built per call.

Ports (all optional, local implementations by default): `run_store` (default: in-memory),
`artifacts` (default: `./.artifacts/`), `events`. TODO(backend): the backend passes its own.

WARNING: without a `run_store`, every call creates a FRESH `InMemoryRunStore`, and without
`events` a fresh `ListEventSink`, that nothing else can reach. A caller that wants to cancel a
run (the backend sets `cancel_requested` in the run store) or to read its events MUST pass its
own `run_store` and `events`, and a `run_id` to address the run.

The Gate's (and reviewer's) spend is added to the returned envelope's spend. It is not charged
to the run's budget (the run does not exist yet; both are free Groq calls).

Billing and failure signals (for the backend). Every envelope carries `output["billable"]`:
False for every run that ends before an image step (bad arguments, a step 0 refusal, a Gate or
review block, an undecided Gate or review, an escalation), and for a supervised run True exactly
when at least one image was generated. A run that ends before an image step also carries
`output["failure"]`, one of EARLY_FAILURES; RETRYABLE_FAILURES are outages the user may simply
retry, the rest are decisions about the request itself.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Literal, get_args

from media_engine.agents.base import AgentConfig
from media_engine.agents.gate.gate import Gate, SafetyReviewer
from media_engine.agents.gate.limiter import review_limiter
from media_engine.agents.grand.grand_agent import GrandAgent
from media_engine.agents.image_grand.supervisor import create_image_grand_agent
from media_engine.contracts import (
    AuthContext,
    Envelope,
    EnvelopeState,
    Goal,
    Spend,
    Status,
)
from media_engine.core.logging import get_logger
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver
from media_engine.gateway.router import Gateway
from media_engine.guardrails.code_checks import check_prompt
from media_engine.harness import trace as tracing
from media_engine.harness.run_state import GateRecord
from media_engine.harness.trace import Trace
from media_engine.ports import (
    ArtifactStore,
    EventSink,
    InMemoryRunStore,
    ListEventSink,
    LocalArtifactStore,
    RunStore,
)
from media_engine.schemas import GateDecision, SafetyReview
from media_engine.services.create.loader import CreateConfig, get_create_config

log = get_logger("create")

DEFAULT_AUTH = AuthContext(plan="free", execution_mode="auto", model_demand="balanced")


@lru_cache
def get_gateway() -> Gateway:
    """The process-wide provider gateway. One instance: see the module docstring."""
    return Gateway()


async def aclose_gateway() -> None:
    """Close the shared gateway's HTTP client (scripts call this on exit)."""
    if get_gateway.cache_info().currsize:
        await get_gateway().aclose()
        get_gateway.cache_clear()


def _invalid(reason: str) -> Envelope:
    return Envelope(
        output={"failure": "invalid_input", "billable": False},
        summary="The request could not be started.",
        state=EnvelopeState(status=Status.FAILED, spend=Spend(), reason=reason),
    )


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

REFUSED_SUMMARY = "This request can't be made into an image."
UNCHECKED_SUMMARY = "The request could not be checked right now. Please try again."
# Step 0 rules that are an input-format problem (FAILED), not a refusal of the content.
_FORMAT_RULES = frozenset({"empty", "too_long", "control_chars"})


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


async def create_images(
    prompt: str,
    *,
    auth: AuthContext = DEFAULT_AUTH,
    n: int = 1,
    settings: dict[str, str] | None = None,
    user_id: str = "anonymous",
    run_id: str | None = None,
    gateway: Gateway | None = None,
    run_store: RunStore | None = None,
    artifacts: ArtifactStore | None = None,
    events: EventSink | None = None,
    cfg: CreateConfig | None = None,
    seed_source: Callable[[], int] | None = None,
    trace: Trace | None = None,
) -> Envelope:
    """Turn `prompt` into `n` quality-checked images. See the supervisor's docstring for the
    envelope's output. Never talks to the user: NEEDS_INPUT and ESCALATE are returned upward.

    `trace` (optional): a developer `Trace` the caller keeps, to read what the run did while it is
    still going (every model call, every decision). It is a separate channel: nothing in it
    reaches the Envelope or an event."""
    cfg = cfg or get_create_config()
    if not prompt.strip():
        return _invalid("no prompt given")
    if not 1 <= n <= cfg.policy.max_images_per_request:
        return _invalid(f"n must be between 1 and {cfg.policy.max_images_per_request}")

    backend = gateway or get_gateway()
    sink = events if events is not None else ListEventSink()
    run_id = run_id or f"run_{uuid.uuid4().hex}"
    llm = LLMGateway(backend, chain_resolver(cfg.agents))
    trace = trace or Trace()

    # 0. code checks: no model, no cost. A refusal never reaches the Gate.
    checked = check_prompt(prompt)
    if not checked.ok:
        trace.decide("route", f"code check refused the request: {checked.rule}")
        log.info("create.code_check_refused", run_id=run_id, rule=checked.rule)
        if checked.rule in _FORMAT_RULES:
            trace.finish(Status.FAILED.value)
            return _invalid(checked.reason or "invalid prompt")
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
    gate = Gate(AgentConfig.from_policy("gate", cfg.agent("gate"), GateDecision), llm, sink)
    stop = cfg.policy.stopping_condition("route", auth.plan, "a safety verdict and a route")
    with tracing.scope(trace, "route", 0):  # the Gate's call is seq 1 of the run's trace
        gated = await gate.run(
            Goal(task="gate", instructions="Check the request and decide how to handle it."),
            stop,
            auth,
            {"prompt": prompt},
        )
    spend = gated.state.spend
    if gated.state.status is not Status.DONE:
        trace.decide("route", f"gate UNDECIDED, run stopped (fail closed): {gated.state.reason}")
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
            AgentConfig.from_policy("safety_reviewer", cfg.agent("safety_reviewer"), SafetyReview),
            llm,
            sink,
            limiter=review_limiter(limits.max_per_period, limits.period_s),
            wait_s=limits.wait_s,
        )
        with tracing.scope(trace, "route", 0):
            reviewed = await reviewer.run(
                Goal(task="review", instructions="Decide whether the request may be made."),
                stop,
                auth,
                {"prompt": prompt, "safety_categories": verdict.safety_categories},
            )
        spend = spend.plus(reviewed.state.spend)
        if reviewed.state.status is not Status.DONE:
            trace.decide("route", f"safety review UNDECIDED, run stopped: {reviewed.state.reason}")
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
    route = verdict.route
    log.info("create.routed", run_id=run_id, route=route)
    if verdict.task_type != "create":
        # INTENTIONALLY IGNORED (review round 2): create takes no uploads, and the edit and
        # merge services do not exist yet, so an "edit" or "merge" verdict (often a text-only
        # request like "change the background to blue") is handled as a new image. The value is
        # recorded on RunState for later routing (spec §10), never acted on here. Rejecting it
        # instead would refuse requests create can still serve.
        trace.decide("route", f"task_type {verdict.task_type} ignored: create makes a new image")

    # 2. hand over
    if route == "complex":
        grand = GrandAgent(
            AgentConfig.from_policy("grand_agent", cfg.agent("grand_agent"), None), None, sink
        )
        env = await grand.run(
            Goal(task="plan", instructions="Plan the request."),
            cfg.policy.stopping_condition("supervise", auth.plan, "a plan"),
            auth,
            {"prompt": prompt, "gate": verdict.model_dump()},  # the verdict travels with it
        )
        env.output = {"run_id": run_id, **env.output, "billable": False}  # no image made
        trace.decide(
            "route",
            "complex: handed to the Grand Agent stub; result ESCALATE (no planning is done)",
        )
    else:
        supervisor = create_image_grand_agent(
            cfg,
            llm=llm,
            images=backend,
            artifacts=artifacts or LocalArtifactStore(),
            events=sink,
            run_store=run_store or InMemoryRunStore(),
            seed_source=seed_source,
            trace=trace,
        )
        inputs: dict[str, Any] = {
            "prompt": prompt,
            "settings": settings or {},
            "n": n,
            "run_id": run_id,
            "user_id": user_id,
            "gate": verdict.model_dump(),
        }
        env = await supervisor.run(
            Goal(task="create", instructions="Create the image the user asked for."),
            cfg.policy.stopping_condition("supervise", auth.plan, "finished, checked images"),
            auth,
            inputs,
        )
    env.state.spend = env.state.spend.plus(spend)
    trace.finish(env.state.status.value)  # closes the developer trace (status and overall time)
    return env


__all__ = ["DEFAULT_AUTH", "aclose_gateway", "create_images", "get_gateway"]
