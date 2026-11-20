"""image-create-agent: the create pipeline runner (plain path). Plain Python around a linear
LangGraph attempt graph.

    run(): build RunState -> ONE attempt (enhance -> generate -> critique) -> settle

Create is LINEAR (IMAGE-CREATE-OPTIMIZE v2.2): there is no regeneration loop and no edge back to
an earlier step. The run makes its images once and returns every one the Critic allowed, best
first; the user picks. The Critic's quality score is information (ranking, evals), never a
trigger: a low score adds an issue, it does not spend money on another image. The supervisor
makes NO LLM call of its own: every call is a step agent's and is counted in that step's Spend.

The graph runs exactly once; no step is re-run. A Critic that fails without a verdict retries
ONCE inside its own step (critic.py `params.retries`), on the same images, never generating.
Failures another try cannot fix (no_provider / bad_request / invalid_input / content_refused)
are not retried; those images stay unchecked and are never delivered.

N images (spec §8.4): one shared enhancement, then N branches (creator, then critic) with seeds
base+0..N-1 (base derived from the run id, so a run is replayable), run concurrently
(`asyncio.gather(..., return_exceptions=True)`), at most `max_parallel_branches` at a time. One
failing branch never cancels the others: it is marked on the plan step (branches_aborted) and
its record carries the failure, which is what the caller refunds. Every checked image is
returned, ranked best first.

Cancellation (spec §10): cooperative via `RunState.cancel_requested`, seen at every step
boundary and immediately before each branch's paid call (the step guard, `_gate` and
`_branch_gate`: the only places cancel and budget are checked). While a step runs, a watcher
also polls the RunStore (the backend writes the flag there) and aborts the in-flight provider
calls; the in-flight step is then `cancelled`, with branches_done / branches_aborted /
partial_outputs, and pending steps are `skipped`. Images the Critic never ruled on are never
returned (they are marked orphaned); only actual spend is reported.

inputs:
    prompt:   str             the user's request (required)
    settings: dict[str, str]  user-chosen values, e.g. {"aspect_ratio": "16:9"} (optional)
    n:        int             images wanted (optional, default 1, at most the plan's
                              max_images_per_request)
    run_id / user_id: str     (optional) identifiers for RunState

output (envelope): run_id, image_ids (best first; empty unless delivered), plus on delivery
    quality_score, issues, assumptions, text_wanted, width, height (and quality_scores when
    n > 1). NEEDS_INPUT carries questions. A CANCELLED run lists the images the Critic had
    already allowed. Never image bytes, never an internal agent name.

RunState still carries `retries_used`, `replans_used` and the per-record `lever` / `feedback`
fields: they are part of the stored-state contract and are always 0 / None since v2.2.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator

from media_engine.agents.base import Agent, AgentConfig, describe_validation_error
from media_engine.agents.image_generation.create.brief import Brief
from media_engine.agents.image_generation.create.critic import Critic
from media_engine.agents.image_generation.create.graph import (
    NODES,
    build_attempt_graph,
    run_attempt,
)
from media_engine.contracts import (
    AuthContext,
    Envelope,
    EnvelopeState,
    Goal,
    Plan,
    Spend,
    Status,
    StoppingCondition,
)
from media_engine.core.config import get_settings
from media_engine.core.logging import get_logger
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver
from media_engine.gateway.model_gateway import ImageBackend, ModelGateway
from media_engine.harness import trace as tracing
from media_engine.harness.run_state import (
    AttemptRecord,
    GateRecord,
    PlanStep,
    PublicLabel,
    RunState,
    StepStatus,
)
from media_engine.harness.trace import Trace
from media_engine.ports.artifact_store import ArtifactStore
from media_engine.ports.event_sink import EventSink, Phase, make_event, safe_emit
from media_engine.ports.run_store import RunStore
from media_engine.schemas import CriticOutput, Document, DocumentSize, EnhancerOutput, Layer
from media_engine.services.create.loader import CreateConfig, StepKind
from media_engine.tools.image_gen_model import ImageGenModel

log = get_logger("agents.image_grand")


def derive_seed(run_id: str, round_number: int, max_seed: int) -> int:
    """Deterministic base seed for one round of a run (spec §5.3, D1): the same run id and
    round always give the same seed, so a run can be replayed. Branch i uses base + i."""
    digest = hashlib.sha256(f"{run_id}:{round_number}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (max_seed + 1)


PRESET_KINDS = ("aspect_ratio", "size", "style")

# settings["resolution"] -> how many times the preset (1K, long side <= 1024) is scaled. The
# gateway then fits the size to each model's max side (a model that cannot go higher gets 1K).
RESOLUTION_SCALE = {"1K": 1, "2K": 2, "4K": 4}

# Added to a delivered image the user's picked model did not make (it was refused or failed, and
# the gateway fell back down the chain).
PICK_NOT_USED = "The model you chose was not available, so another model made this image."
# Added when the user asked for 2K or 4K and the model that made the image cannot go that high.
RESOLUTION_LIMITED = (
    "The resolution you asked for is higher than the model that made this image supports, "
    "so it was made smaller."
)
# These describe the whole run, not one image: reported once, never "image k: ..." per image.
RUN_LEVEL_NOTES = (PICK_NOT_USED, RESOLUTION_LIMITED)
# Returned images within this fraction of the requested long side count as the size asked for.
RESOLUTION_TOLERANCE = 0.9

# (step kind, internal agent name, public label). Internal names never leave RunState.
PLAN: tuple[tuple[StepKind, str, PublicLabel], ...] = (
    ("enhance", "prompt_enhancer", "Understanding your request"),
    ("generate", "image_creator", "Creating your image"),
    ("critique", "critic", "Finishing up"),
)
if tuple(kind for kind, _, _ in PLAN) != NODES:  # wiring check, not an assert (-O strips those)
    raise RuntimeError(f"PLAN {PLAN} does not match the attempt graph nodes {NODES}")

SUCCESS: dict[StepKind, str] = {
    "enhance": "a detailed image prompt and testable requirements",
    "generate": "the requested images, stored",
    "critique": "a quality score and a safety verdict for each image",
}

SUMMARIES = {
    Status.DONE: "Your image is ready.",
    Status.DONE_WITH_ISSUES: "Your image is ready, with some issues noted.",
    Status.NEEDS_INPUT: "A little more information is needed before the image can be made.",
    Status.REJECTED: "This image could not be delivered because it did not pass the safety review.",
    Status.CANCELLED: "Image creation was cancelled.",
    Status.FAILED: "The image could not be created.",
}
MULTI_SUMMARIES = {
    Status.DONE: "Your images are ready.",
    Status.DONE_WITH_ISSUES: "Your images are ready, with some issues noted.",
}

# A step's envelope status -> its plan status. Anything else (DONE, DONE_WITH_ISSUES,
# NEEDS_INPUT, REJECTED) means the step agent did its job: "done".
_STEP_STATUS: dict[Status, StepStatus] = {Status.FAILED: "failed", Status.CANCELLED: "cancelled"}

# Why an image was stored but never delivered (ArtifactStore.mark_orphaned reasons).
ORPHAN_UNCHECKED = "unchecked"  # the Critic never gave it a verdict
ORPHAN_RUN_REJECTED = "run_rejected"  # allowed, but another image of the run was blocked

# Branch failure markers that are not a FailureKind: the branch never ran its paid call.
BRANCH_ABORTED = "aborted"  # cancelled (or the run was already decided) before or during the call
BRANCH_BUDGET = "budget"  # the budget would have been exceeded, so the call was not made

# Issue added to a delivery when the run stopped early after some images were checked.
STOPPED_ISSUES = {"budget": "The budget ran out before every image could be finished."}
STOPPED_DEFAULT = "Not every image could be finished."


class SupervisorInputs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1)
    settings: dict[str, str] = Field(default_factory=dict)
    n: int = Field(default=1, ge=1)  # capped by max_images_per_request (validation context)
    run_id: str | None = None
    user_id: str = "anonymous"  # TODO(backend): the backend supplies the real user
    # The Gate's verdict from the entry point. Stored once on RunState, never re-derived here:
    # the supervisor only ever runs for a request the Gate (or its reviewer) allowed.
    gate: GateRecord | None = None
    # The user's pick from the plan catalogue (validated by create_images): an OpenRouter slug
    # and the billed quality it fixes. None = auto.
    model: str | None = None
    quality: str | None = None

    @field_validator("n")
    @classmethod
    def _within_the_cap(cls, value: int, info: ValidationInfo) -> int:
        cap = (info.context or {}).get("max_images")
        if cap is not None and value > cap:
            raise ValueError(f"at most {cap} images per request")
        return value


@dataclass
class Outcome:
    status: Status
    reason: str | None = None
    output: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None
    # For a FAILED step: which step and why (for logs and the caller), without parsing text.
    failed_step: StepKind | None = None
    failure: str | None = None  # a FailureKind, or "budget" for a budget refusal


@dataclass
class RunContext:
    """Per-run handles for the attempt graph. RunState is the ONLY state; everything here is
    either a reference to it, an immutable input, or the outcome decided by the step guard."""

    state: RunState
    auth: AuthContext
    user_prompt: str
    settings: dict[str, str]
    n: int = 1  # images wanted (an immutable input)
    model: str | None = None  # the user's pick (immutable input); None = auto
    quality: str | None = None
    trace: Trace = field(default_factory=Trace)  # developer trace (not run state)
    t0: dict[int, float] = field(default_factory=dict)  # step number -> perf_counter at start
    budgeted: int = 0  # branches the current step's ONE guard found affordable (D5)
    _outcome: Outcome | None = field(default=None, init=False, repr=False)

    @property
    def outcome(self) -> Outcome | None:
        """Set when the attempt cannot go on (cancel, budget, block, a failed step, ...). While
        it is None the attempt ran to a verdict."""
        return self._outcome

    @outcome.setter
    def outcome(self, value: Outcome | None) -> None:
        # REJECTED is sticky: nothing (a cancel, a budget refusal, a crash) may replace it, so a
        # block can never be turned into another outcome, whatever order things finish in.
        if self._outcome is not None and self._outcome.status is Status.REJECTED:
            return
        self._outcome = value

    def step(self, kind: StepKind) -> PlanStep:
        return self.state.step(NODES.index(kind) + 1)


class Resolved(BaseModel):
    """How the Enhancer's gaps were filled. Stored in the enhance step's output."""

    prompt: str  # what the Image Creator is sent (the generic compile: the fallback prompt)
    brief: dict[str, Any] | None = None  # compiled per model by the gateway when present
    width: int
    height: int
    aspect_ratio: str
    assumptions: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)  # unanswered user questions


class ImageCreateAgent(Agent):
    def __init__(
        self,
        config: AgentConfig,
        gateway: LLMGateway | None,
        events: EventSink,
        *,
        create: CreateConfig,
        enhancer: Brief,
        creator: ImageGenModel,
        critic: Critic,
        artifacts: ArtifactStore,
        run_store: RunStore | None = None,
        paid_providers: Iterable[str] = (),
        allow_paid: bool = False,
        price_lookup: Callable[[Plan], float | None] | None = None,
        seed_source: Callable[[], int] | None = None,
        trace: Trace | None = None,
    ) -> None:
        # The supervisor makes no LLM call of its own (create is linear, v2.2).
        super().__init__(config, gateway, events)
        self._cfg = create
        self._policy = create.policy
        self._enhancer = enhancer
        self._creator = creator
        self._critic = critic
        self._artifacts = artifacts
        self._store = run_store
        self._paid = frozenset(paid_providers)
        self._allow_paid = allow_paid
        # plan -> the dearest image that plan can be billed (USD), or None when unknown. Passed in
        # by the entry point, so the supervisor does not reach into the gateway's registry.
        self._price_lookup = price_lookup
        self._trace = trace  # developer trace for this run (create_images passes one in)
        # Tests may inject a seed source; otherwise each round's base seed is DERIVED from the
        # run id (spec §5.3 / D1), so a run can be replayed from its id alone.
        self._seed_source = seed_source
        self.graph = build_attempt_graph(
            [
                ("enhance", self._enhance),
                ("generate", self._generate),
                ("critique", self._critique),
            ],
            on_crash=self._on_crash,
        )

    # -- entry ----------------------------------------------------------------------------------

    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope:
        try:
            ctx = self.new_context(auth, inputs)
        except ValidationError as err:
            return self.failed(
                "invalid inputs:\n" + describe_validation_error(err), Spend(), "invalid_input"
            )
        state = ctx.state
        await self._save(ctx)

        if state.gate is not None and state.gate.safety != "allow":
            # Defence in depth: the entry point never hands over a blocked or unreviewed request,
            # but if a bug ever did, no step (and no paid image call) may run for it.
            ctx.trace.decide("guard", f"gate verdict is {state.gate.safety}: nothing is run")
            ctx.outcome = Outcome(Status.REJECTED, "the request did not pass the safety check")
        else:
            await self._drive(ctx)
        outcome = await self._settle(ctx)

        for item in state.plan:  # nothing is removed: undone steps are marked skipped
            if item.status == "pending":
                item.status = "skipped"
        state.status = outcome.status
        state.finished_at = datetime.now(UTC)
        await self._save(ctx)
        log.info(
            "supervisor.done",
            run_id=state.run_id,
            status=outcome.status.value,
            attempts=state.attempts[-1].number,
            llm_calls=self.run_spend(state).llm_calls,
        )
        # billable: at least one image was generated, so the run spent on images (even if it
        # was later blocked or never delivered). The backend decides credits from this.
        billable = any(a.image_id for a in state.attempts)
        return Envelope(
            output={
                "run_id": state.run_id,
                "image_ids": [],
                **outcome.output,
                # why a FAILED run failed (e.g. "bad_request"), for the backend's retry decision
                **(
                    {"failure": outcome.failure}
                    if outcome.failure and outcome.status is Status.FAILED
                    else {}
                ),
                "billable": billable,
            },
            summary=outcome.summary or SUMMARIES[outcome.status],
            state=EnvelopeState(
                status=outcome.status, spend=self.run_spend(state), reason=outcome.reason
            ),
        )

    def new_context(self, auth: AuthContext, inputs: dict[str, Any]) -> RunContext:
        """A fresh RunState (every step pending, the first round open) and its context.
        Raises ValidationError."""
        args = SupervisorInputs.model_validate(
            inputs, context={"max_images": self._policy.max_images_per_request[auth.plan]}
        )
        state = RunState(
            run_id=args.run_id or f"run_{uuid.uuid4().hex}",
            user_id=args.user_id,
            auth=auth,
            budget=self._policy.budget_for(auth.plan),
            gate=args.gate,
            plan=[
                PlanStep(step=i, agent=agent, public_label=label)
                for i, (_, agent, label) in enumerate(PLAN, start=1)
            ],
        )
        state.attempts = self._new_round(state.run_id, 1, args.n, tier=auth.plan)
        trace = self._trace or Trace()
        trace.bind(state)  # what the router recorded moves into RunState.calls
        trace.meta = {
            "plan": auth.plan,
            "n": args.n,
            "quality_threshold": self._policy.quality_threshold[auth.plan],
            "reserve_fraction": state.budget.reserve_fraction,
            "max_parallel_branches": self._policy.max_parallel_branches,
        }
        if args.model:
            trace.meta["model_pick"] = args.model + (f" ({args.quality})" if args.quality else "")
        return RunContext(
            state,
            auth,
            args.prompt,
            dict(args.settings),
            n=args.n,
            model=args.model,
            quality=args.quality,
            trace=trace,
        )

    def _base_seed(self, run_id: str, number: int) -> int:
        if self._seed_source is not None:
            return self._seed_source()
        return derive_seed(run_id, number, self._policy.max_seed)

    def _new_round(self, run_id: str, number: int, n: int, *, tier: Plan) -> list[AttemptRecord]:
        """One record per branch. Branch i gets base + i, where base is derived from
        (run_id, round): replayable from the run id, and the branches are distinct by
        construction, so no two of the N images the user pays for share a seed."""
        base, modulus = self._base_seed(run_id, number), self._policy.max_seed + 1
        return [
            AttemptRecord(number=number, branch=i, seed=(base + i) % modulus, tier=tier)
            for i in range(n)
        ]

    @staticmethod
    def run_spend(state: RunState) -> Spend:
        """The run's spend: the plan's steps (plus `supervisor_spend`, always zero since v2.2:
        the supervisor makes no call), so this stays the single place LLM calls are counted."""
        total = state.supervisor_spend
        for item in state.plan:
            total = total.plus(item.spend)
        return total

    # -- one attempt ------------------------------------------------------------------------------

    async def _drive(self, ctx: RunContext) -> None:
        """Run the attempt graph ONCE (create is linear, v2.2): no step is ever re-run. A Critic
        that fails without a verdict retries once INSIDE its own step (critic.py). Ends with the
        run's verdict on `ctx.outcome` (None: it reached a verdict)."""
        await run_attempt(self.graph, ctx.state.run_id, ctx)
        ended = ctx.outcome
        ctx.trace.decide(
            "supervise",
            "attempt ended: "
            + ("reached a verdict" if ended is None else f"{ended.status.value} ({ended.reason})"),
        )

    # -- the step guard: the only place cancel and budget are checked ----------------------------

    async def _begin(self, ctx: RunContext, kind: StepKind, branches: int = 1) -> PlanStep | None:
        """Boundary save, then the ONE guard for this step, then `running`. Returns the step to
        run, or None when there is nothing to run: the outcome is already decided. `branches` is
        how many paid calls the step will make; `ctx.budgeted` is how many of them the budget
        covers. The node then returns: one way out, END. No await after the checks: the caller's
        next await is the step agent."""
        if ctx.outcome is not None:
            return None
        step = ctx.step(kind)
        await self._save(ctx)  # step boundary: RunState is JSON-complete here
        budgeted = await self._gate(ctx, kind, step, branches)
        if budgeted is None:
            return None
        ctx.budgeted = budgeted
        step.status = "running"
        self._stamp_start(ctx, step)
        self._emit(ctx, step, "started")
        return step

    async def _gate(
        self, ctx: RunContext, kind: StepKind, step: PlanStep, branches: int
    ) -> int | None:
        """D5: ONE guard per step, taking the branch count. Cancel, then budget: how many of the
        `branches` paid calls the run can afford, decided once and logged once. None: the outcome
        is now decided (cancelled, or not even one call is affordable)."""
        await self._refresh_cancel(ctx)
        if ctx.state.cancel_requested:
            await self._cancel(ctx, step)
            return None
        wanted = max(1, branches)
        budgeted = next(
            (k for k in range(wanted, 0, -1) if self._affordable(ctx, kind, units=k, log=False)),
            0,
        )
        self._affordable(ctx, kind, units=max(1, budgeted), who=f"{budgeted} of {wanted}")
        if budgeted == 0:
            self._refuse_budget(ctx, step, kind)
            return None
        return budgeted if branches else 0

    async def _branch_gate(self, ctx: RunContext, step: PlanStep, record: AttemptRecord) -> bool:
        """Immediately before ONE branch's paid call: has the run been decided or cancelled?
        Cancel stays checked per branch (D5); the budget was decided once, by the step's guard.
        A cancel decides the whole run (the outcome). No await after it: the caller's next
        await is the branch's agent call."""
        if ctx.outcome is not None:  # decided while this branch waited its turn
            record.failure = BRANCH_ABORTED
            return False
        await self._refresh_cancel(ctx)
        if ctx.state.cancel_requested:
            ctx.state.cancelled_at_step = step.step
            ctx.outcome = Outcome(Status.CANCELLED, f"cancelled during step {step.step}")
            record.failure = BRANCH_ABORTED
            return False
        return True

    async def _refresh_cancel(self, ctx: RunContext) -> None:
        """The backend sets `cancel_requested` on ITS side of the store. Fold that into the live
        state with the cheap `RunStore.cancel_requested` read (not a full `load`). A store that
        fails is ignored: the live flag still works."""
        if ctx.state.cancel_requested or self._store is None:
            return
        try:
            requested = await self._store.cancel_requested(ctx.state.run_id)
        except Exception:  # noqa: BLE001 - a flaky store must not fail the run
            log.warning("supervisor.cancel_poll_failed", run_id=ctx.state.run_id)
            return
        if requested:
            ctx.state.cancel_requested = True

    def _affordable(
        self,
        ctx: RunContext,
        kind: StepKind,
        *,
        units: int = 1,
        owed: int = 0,
        who: str = "step",
        log: bool = True,
    ) -> bool:
        """Can the run afford `units` of this step's paid call, plus what it already owes?

        generate: each unit is a generate AND the critique that must follow it (an image that
            can't be checked can't be delivered), and `owed` more critiques are due for images
            already generated (their generation is charged, so only the critique is counted).
        Dollars are only counted when paid providers are on: with ALLOW_PAID off the gateway
        skips every paid chain entry, so no dollars can be spent. The estimates are the worst
        reachable chain entry (harness.yaml)."""
        weights: dict[StepKind, int] = {kind: units}
        if kind == "generate":
            weights = {"generate": units, "critique": units + owed}
        est = self._policy.step_estimates
        tokens = sum(est[k].tokens * w for k, w in weights.items())
        cost = 0.0
        if self._allow_paid:
            cost = sum(
                (self._generate_usd(ctx.auth.plan) if k == "generate" else est[k].cost_usd) * w
                for k, w in weights.items()
            )
        neurons = sum(est[k].neurons * w for k, w in weights.items())
        # Nothing may draw on the reserve since create became linear (no re-run of any step);
        # reserve_fraction is 0 in harness.yaml, so the whole budget is usable by the steps.
        reserve = False
        budget = ctx.state.budget
        ok = budget.can_afford(tokens, cost, neurons, use_reserve=reserve)
        if not log:
            return ok
        left = budget.remaining_total() if reserve else budget.remaining_for_steps()
        ctx.trace.decide(
            kind,
            f"guard before {kind} [{who} branches]: cancel not requested; "
            f"budget {'ok' if ok else 'REFUSED'}: "
            f"needs ~{neurons:.0f} neurons / {tokens} tokens / ${cost:.4f}, left "
            f"{left.neurons:.0f} / {left.tokens} / ${left.cost_usd:.4f} "
            f"({'incl. the reserve' if reserve else 'reserve held back'})",
        )
        return ok

    def _refuse_budget(self, ctx: RunContext, step: PlanStep, kind: StepKind) -> None:
        log.warning("supervisor.budget_refused", run_id=ctx.state.run_id, step=kind)
        where = f"step {step.step}"
        step.status = "skipped"
        ctx.outcome = Outcome(
            Status.FAILED, f"budget would be exceeded before {where}", failure="budget"
        )

    async def _cancel(self, ctx: RunContext, step: PlanStep) -> None:
        """Freeze at a boundary: done steps keep outputs and spend, this and later steps are
        skipped (never removed)."""
        ctx.trace.decide("guard", f"guard: cancel requested, frozen before step {step.step}")
        ctx.state.cancelled_at_step = step.step
        step.status = "skipped"
        ctx.outcome = Outcome(Status.CANCELLED, f"cancelled before step {step.step}")

    def _on_crash(self, ctx: RunContext, node: str, err: Exception) -> None:
        for item in ctx.state.plan:
            if item.status == "running":
                item.status = "failed"
                self._stamp_end(ctx, item)
        if ctx.outcome is None:
            ctx.outcome = Outcome(Status.FAILED, f"internal error: {type(err).__name__}")

    # -- running calls: concurrency and in-flight cancel ----------------------------------------

    async def _run_jobs(
        self, ctx: RunContext, jobs: list[Callable[[], Awaitable[Any]]], limit: int
    ) -> tuple[list[Any], bool]:
        """Run `jobs` concurrently, at most `limit` at a time. One job raising never cancels
        the others (`return_exceptions=True`); its exception comes back in the results. While
        they run, a watcher looks for a cancel and aborts the in-flight jobs (the provider
        request is aborted with the task). Returns (results, cancelled_in_flight)."""
        gate = asyncio.Semaphore(max(1, limit))

        async def guarded(job: Callable[[], Awaitable[Any]]) -> Any:
            async with gate:
                return await job()

        tasks = [asyncio.ensure_future(guarded(job)) for job in jobs]
        fired = asyncio.Event()
        watcher = asyncio.ensure_future(self._watch_cancel(ctx, tasks, fired))
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:  # also when the whole run is cancelled from outside: leave nothing running
            watcher.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(watcher, *tasks, return_exceptions=True)
        return list(results), fired.is_set()

    async def _watch_cancel(
        self, ctx: RunContext, tasks: Sequence[asyncio.Future[Any]], fired: asyncio.Event
    ) -> None:
        interval = self._policy.cancel_poll_ms / 1000
        while True:
            await asyncio.sleep(interval)
            await self._refresh_cancel(ctx)
            if ctx.state.cancel_requested:
                if all(task.done() for task in tasks):
                    return
                fired.set()
                for task in tasks:
                    task.cancel()
                return

    # -- nodes ----------------------------------------------------------------------------------

    def _critique_stop(self, ctx: RunContext, images: int) -> StoppingCondition:
        """The Critic reviews every image in ONE call, so its token budget is the per-image
        estimate times the number of images (a repair of a 4-image call re-sends all four)."""
        stop = self._stop(ctx, "critique")
        return stop.model_copy(update={"max_tokens": stop.max_tokens * max(1, images)})

    def _stop(self, ctx: RunContext, kind: StepKind) -> StoppingCondition:
        return self._policy.stopping_condition(kind, ctx.auth.plan, SUCCESS[kind])

    async def _enhance(self, ctx: RunContext) -> None:
        step = await self._begin(ctx, "enhance")
        if step is None:
            return
        attempt = ctx.state.attempt
        inputs: dict[str, Any] = {"prompt": ctx.user_prompt, "settings": ctx.settings}
        stop = self._stop(ctx, "enhance")

        async def job() -> Envelope:
            with tracing.scope(ctx.trace, "enhance", attempt.number):
                return await self._enhancer.run(
                    Goal(task="enhance", instructions="Write a detailed image prompt."),
                    stop,
                    ctx.auth,
                    inputs,
                )

        (result,), cancelled = await self._run_jobs(ctx, [job], limit=1)
        if isinstance(result, asyncio.CancelledError) or cancelled:
            await self._freeze(ctx, step)
            return
        if isinstance(result, BaseException):
            raise result  # the node wrapper records it as this step's failure
        env: Envelope = result
        if env.state.status in (Status.DONE, Status.NEEDS_INPUT):
            resolved = self._resolve(env.output, ctx)
            step.output = {**env.output, "resolved": resolved.model_dump(), "summary": env.summary}
            step.output_ref = f"{ctx.state.run_id}:enhance"
            ctx.trace.decide(
                "enhance",
                f"resolved prompt: {resolved.prompt}; size {resolved.width}x{resolved.height} "
                f"({resolved.aspect_ratio}); assumptions: {resolved.assumptions or 'none'}; "
                f"must_have: {env.output.get('must_have')}; dropped as ungrounded: "
                f"{env.output.get('must_have_dropped') or 'none'}",
            )
            await self._finish(ctx, step, env)
            if resolved.questions and ctx.auth.execution_mode == "manual":
                ctx.outcome = Outcome(
                    Status.NEEDS_INPUT,
                    "the request needs details only the user can give",
                    {"questions": resolved.questions, "assumptions": resolved.assumptions},
                )
            return
        step.output = dict(env.output)
        await self._finish(ctx, step, env)
        ctx.outcome = Outcome(
            Status.FAILED,
            _failure_reason("prepare the request", env),
            failed_step="enhance",
            failure=env.output.get("failure"),
        )

    async def _generate(self, ctx: RunContext) -> None:
        records = ctx.state.round
        step = await self._begin(ctx, "generate", branches=len(records))
        if step is None:
            return
        resolved = Resolved.model_validate(ctx.step("enhance").output["resolved"])
        funded = self._fund(ctx, step, records)

        jobs = [self._generate_job(ctx, step, record, resolved) for record in funded]
        results, cancelled = await self._run_jobs(
            ctx, jobs, min(len(funded), self._policy.max_parallel_branches)
        )
        self._note_crashed_branches(funded, results, "generate")
        await self._close_step(
            ctx, step, records, cancelled, made=lambda r: r.image_id is not None, doing="create"
        )

    def _generate_usd(self, plan: Plan) -> float:
        """The dearest image the plan can be billed (M4: a flat figure under-estimated the ultra
        models). The entry point supplies the lookup; with none, or no price known for the plan,
        harness.yaml's flat figure stands."""
        priced = self._price_lookup(plan) if self._price_lookup is not None else None
        return priced if priced is not None else self._policy.step_estimates["generate"].cost_usd

    def _call_cap_usd(self, ctx: RunContext) -> float | None:
        """What one generate call may cost: the dollars left for steps, shared out among the
        branches that were funded (they run in parallel, so each gets its share)."""
        if not self._allow_paid:
            return None  # the gateway skips every paid entry anyway
        left = ctx.state.budget.remaining_for_steps().cost_usd
        return left / max(1, ctx.budgeted)

    @staticmethod
    def _fund(ctx: RunContext, step: PlanStep, records: list[AttemptRecord]) -> list[AttemptRecord]:
        """Split the step's branches by the ONE guard's verdict (D5): the first `ctx.budgeted`
        run; the rest are refused BEFORE any paid call (BRANCH_BUDGET), in one place."""
        funded, refused = records[: ctx.budgeted], records[ctx.budgeted :]
        for record in refused:
            record.failure = BRANCH_BUDGET
        if refused:
            log.warning("supervisor.branches_budget_refused", count=len(refused))
        step.branches_total, step.branches_done = len(records), 0
        step.branches_aborted = len(refused)
        step.partial_outputs = []
        return funded

    def _generate_job(
        self, ctx: RunContext, step: PlanStep, record: AttemptRecord, resolved: Resolved
    ) -> Callable[[], Awaitable[None]]:
        async def job() -> None:
            if not await self._branch_gate(ctx, step, record):
                return
            with tracing.scope(ctx.trace, "generate", record.number, record.branch):
                env = await self._creator.run(
                    Goal(task="generate", instructions="Create the image."),
                    self._stop(ctx, "generate"),
                    ctx.auth,
                    {
                        "prompt": resolved.prompt,
                        "width": resolved.width,
                        "height": resolved.height,
                        "seed": record.seed,
                        "tier": record.tier,
                        "chain_offset": record.chain_offset,
                        "brief": resolved.brief,
                        "max_cost_usd": self._call_cap_usd(ctx),
                        "model": ctx.model,
                        "quality": ctx.quality,
                    },
                )
            self._charge(ctx, step, env)
            if env.state.status is Status.DONE:
                record.image_id = env.output["image_id"]
                record.provider = env.output.get("provider")
                record.width = int(env.output["width"])
                record.height = int(env.output["height"])
                record.shape_issues = self._shape_issues(env.output, resolved)
                if ctx.model and env.output.get("model") != ctx.model:
                    record.shape_issues.append(PICK_NOT_USED)
                asked = max(resolved.width, resolved.height)
                got = max(record.width, record.height)
                if (
                    ctx.settings.get("resolution", "1K") != "1K"
                    and got < asked * RESOLUTION_TOLERANCE
                ):
                    record.shape_issues.append(RESOLUTION_LIMITED)
                step.branches_done = (step.branches_done or 0) + 1
                step.partial_outputs.append(record.image_id)
                step.output_ref = step.output_ref or record.image_id
            else:
                record.failure = env.output.get("failure") or BRANCH_ABORTED
                step.branches_aborted = (step.branches_aborted or 0) + 1
            await self._branch_done(ctx, step)

        return job

    async def _critique(self, ctx: RunContext) -> None:
        """ONE Critic call sees every image of the round and ranks them (§5.4): create makes one
        Critic call per run, not N. A budget-refused image is simply not in the batch."""
        records = [r for r in ctx.state.round if r.image_id and r.verdict is None]
        step = await self._begin(ctx, "critique", branches=len(records))
        if step is None:
            return
        enhanced = ctx.step("enhance").output
        funded = self._fund(ctx, step, records)

        jobs = [self._critique_job(ctx, step, funded, enhanced)] if funded else []
        results, cancelled = await self._run_jobs(ctx, jobs, 1)
        # A crash (or an in-flight cancel) of the one call is a crash of every image in it.
        self._note_crashed_branches(
            funded, [results[0]] * len(funded) if results else [], "critique"
        )
        await self._close_step(
            ctx, step, records, cancelled, made=lambda r: r.verdict is not None, doing="check"
        )

    def _critique_job(
        self,
        ctx: RunContext,
        step: PlanStep,
        records: list[AttemptRecord],
        enhanced: dict[str, Any],
    ) -> Callable[[], Awaitable[None]]:
        async def job() -> None:
            if not await self._branch_gate(ctx, step, records[0]):
                for record in records[1:]:
                    record.failure = BRANCH_ABORTED
                return
            with tracing.scope(ctx.trace, "critique", records[0].number, 0):
                env = await self._critic.run(
                    Goal(task="critique", instructions="Review and rank the images."),
                    self._critique_stop(ctx, len(records)),
                    ctx.auth,
                    {
                        "image_ids": [r.image_id for r in records],
                        "prompt": ctx.user_prompt,
                        "must_have": enhanced["must_have"],
                        "task_type": "create",
                    },
                )
            self._charge(ctx, step, env)
            if env.state.status is not Status.DONE:
                # No verdict for any image: they stay unchecked and are orphaned only once the
                # run settles (the Critic already retried once inside its step).
                failure = env.output.get("failure") or BRANCH_ABORTED
                for record in records:
                    record.failure = failure
                step.branches_aborted = (step.branches_aborted or 0) + len(records)
                await self._branch_done(ctx, step)
                return
            by_id = {r.image_id: r for r in records}
            for review in env.output["reviews"]:  # best first, blocked last
                found = by_id.get(review["image_id"])
                if found is None:
                    continue
                record = found
                if review["verdict"] == "block":
                    # This IMAGE is blocked, whatever its score: the Critic orphaned it and it
                    # is never delivered. The RUN is decided at settle (spec §5.5): the other
                    # images still stand, and only when every reviewed image is blocked is it
                    # REJECTED.
                    record.verdict = "block"
                    ctx.trace.decide(
                        "critique", f"image {record.image_id} blocked: {review['safety_reason']}"
                    )
                    ctx.trace.orphan(str(record.image_id), "safety_block")
                    continue
                record.verdict = "allow"
                record.quality_score = int(review["quality_score"])
                record.scores = dict(review["scores"])
                record.rank = int(review["rank"])
                record.critique = str(review["critique"]) or None
                record.summary = env.summary
                # Allowed by the Critic: safe to show now, before the run ends (§5.3 streaming).
                self._emit(ctx, step, "image_ready", image_id=record.image_id)
                step.branches_done = (step.branches_done or 0) + 1
                step.output_ref = step.output_ref or record.image_id
            for record in records:  # the model left it out: unchecked, never delivered
                if record.verdict is None:
                    record.failure = "invalid_output"
                    step.branches_aborted = (step.branches_aborted or 0) + 1
            await self._branch_done(ctx, step)

        return job

    # -- branch bookkeeping ---------------------------------------------------------------------

    @staticmethod
    def _charge(ctx: RunContext, step: PlanStep, env: Envelope) -> None:
        """Actual spend of one call, charged when it finishes (never an estimate)."""
        step.spend = step.spend.plus(env.state.spend)
        ctx.state.budget.charge(env.state.spend)

    async def _branch_done(self, ctx: RunContext, step: PlanStep) -> None:
        if ctx.n > 1:  # a single image has no branch progress to report
            self._emit(ctx, step, "progress")
            await self._save(ctx)

    @staticmethod
    def _note_crashed_branches(records: list[AttemptRecord], results: list[Any], step: str) -> None:
        """A branch that raised (a bug, or a cancel that aborted it) is a failed branch, not a
        failed run: the other branches' results stand."""
        for record, result in zip(records, results, strict=True):
            if not isinstance(result, BaseException):
                continue
            aborted = isinstance(result, asyncio.CancelledError)
            record.failure = record.failure or (BRANCH_ABORTED if aborted else "internal")
            if not aborted:
                log.error(
                    "supervisor.branch_crashed", step=step, branch=record.branch, err=repr(result)
                )

    async def _close_step(
        self,
        ctx: RunContext,
        step: PlanStep,
        records: list[AttemptRecord],
        cancelled: bool,
        *,
        made: Callable[[AttemptRecord], bool],
        doing: str,
    ) -> None:
        """After the branches: freeze on a cancel, or settle the step's status and, when no
        branch got anywhere, the attempt's failure."""
        step.output = {
            "branches": [
                {
                    "branch": r.branch,
                    "image_id": r.image_id,
                    "verdict": r.verdict,
                    "quality_score": r.quality_score,
                    "failure": r.failure,
                }
                for r in records
            ]
        }
        outcome = ctx.outcome
        if cancelled or (outcome is not None and outcome.status is Status.CANCELLED):
            await self._freeze(ctx, step)
            return
        if outcome is not None and outcome.status is Status.REJECTED:
            step.status = "done"  # the Critic did its job: the verdict was "block"
        elif any(made(r) for r in records):
            step.status = "done"
        else:
            step.status = "failed"
            failure = _step_failure(records)
            kind: StepKind = "generate" if doing == "create" else "critique"
            ctx.outcome = Outcome(
                Status.FAILED,
                f"could not {doing} the image" + (f" ({failure})" if failure else ""),
                failed_step=kind,
                failure=failure,
            )
        self._stamp_end(ctx, step)
        self._emit(ctx, step, "finished")
        await self._save(ctx)

    async def _freeze(self, ctx: RunContext, step: PlanStep) -> None:
        """Cancel seen while this step was running: it is `cancelled` and keeps what it had
        (branches_done, branches_aborted, partial_outputs); later steps become `skipped`."""
        step.status = "cancelled"
        if step.branches_total is not None:
            step.branches_aborted = step.branches_total - (step.branches_done or 0)
        ctx.state.cancelled_at_step = step.step
        ctx.outcome = Outcome(Status.CANCELLED, f"cancelled during step {step.step}")
        ctx.trace.decide("guard", f"in-flight cancel: step {step.step} aborted, plan frozen")
        self._stamp_end(ctx, step)
        self._emit(ctx, step, "finished")
        await self._save(ctx)

    @staticmethod
    def _stamp_start(ctx: RunContext, step: PlanStep) -> None:
        step.started_at, step.finished_at = tracing.now(), None
        ctx.t0[step.step] = time.perf_counter()

    @staticmethod
    def _stamp_end(ctx: RunContext, step: PlanStep) -> None:
        """Developer trace: when the step stopped, and its running time (summed over reruns)."""
        began = ctx.t0.pop(step.step, None)
        if began is not None:
            step.duration_ms += int((time.perf_counter() - began) * 1000)
        step.finished_at = tracing.now()

    async def _finish(self, ctx: RunContext, step: PlanStep, env: Envelope) -> None:
        self._charge(ctx, step, env)
        step.status = _STEP_STATUS.get(env.state.status, "done")
        self._stamp_end(ctx, step)
        self._emit(ctx, step, "finished")
        await self._save(ctx)

    # -- the run's verdict ----------------------------------------------------------------------

    async def _settle(self, ctx: RunContext) -> Outcome:
        """Turn the attempts into the run's outcome, and clear up every stored image that will
        not be delivered. REJECTED delivers nothing; CANCELLED returns only images the Critic
        had allowed; otherwise the checked images are delivered, best first."""
        state, outcome = ctx.state, ctx.outcome
        await self._orphan_unchecked(ctx)
        checked = [a for a in state.attempts if a.verdict == "allow" and a.image_id]
        # Best first; a tie goes to the lower branch (deterministic).
        ranked = sorted(checked, key=lambda a: (a.rank or 10**6, -(a.quality_score or 0), a.branch))
        ctx.trace.decide(
            "settle",
            "ranking (best first): "
            + (
                ", ".join(
                    f"{a.image_id} score {a.quality_score} (branch {a.branch})" for a in ranked
                )
                or "no checked images"
            ),
        )

        blocked = [a for a in state.attempts if a.verdict == "block"]
        if outcome is not None and outcome.status is Status.REJECTED:
            # The Gate's verdict (defence in depth): nothing ran, nothing is delivered.
            for a in checked:
                await self._mark_orphaned(ctx, str(a.image_id), ORPHAN_RUN_REJECTED)
            return outcome
        if blocked and not checked:
            # Settle rule (spec §5.5): EVERY reviewed image was blocked. The input Gate let
            # through a request whose every output is unsafe: a gate miss, logged so the Gate's
            # filter can be tuned. Nothing is delivered.
            log.warning(
                "supervisor.gate_miss",
                run_id=state.run_id,
                blocked=len(blocked),
                gate=state.gate.safety if state.gate else None,
            )
            ctx.trace.decide(
                "settle",
                f"GATE MISS: all {len(blocked)} reviewed image(s) blocked by the Critic after "
                f"the Gate allowed the request",
            )
            return Outcome(Status.REJECTED, "the images did not pass the safety review")
        if outcome is not None and outcome.status is Status.CANCELLED:
            return Outcome(
                Status.CANCELLED, outcome.reason, {"image_ids": [a.image_id for a in ranked]}
            )
        if outcome is not None and outcome.status is Status.NEEDS_INPUT:
            return outcome
        if not checked:
            return outcome or Outcome(Status.FAILED, "the attempt ended without an outcome")

        # Every checked image is delivered, best first, whatever its score: the user picks.
        # The score is information (ranking and an issue note), never a trigger.
        best, delivered = ranked[0], ranked
        threshold = self._policy.quality_threshold[ctx.auth.plan]
        weak = (best.quality_score or 0) < threshold
        issues: list[str] = [best.critique or ""] if weak else []
        if blocked:  # settle rule (spec §5.5): flagged images are dropped, the rest delivered
            issues.append(f"{len(blocked)} of {ctx.n} images were removed by the safety review.")
        missing = ctx.n - len(delivered) - len(blocked)
        if missing:
            issues.append(f"{missing} of {ctx.n} requested images could not be created.")
        run_notes: list[str] = []
        for position, a in enumerate(delivered, start=1):  # every delivered image
            for issue in a.shape_issues:
                if issue in RUN_LEVEL_NOTES:  # about the run: said once, not once per image
                    if issue not in run_notes:
                        run_notes.append(issue)
                    continue
                issues.append(f"image {position}: {issue}" if len(delivered) > 1 else issue)
        issues.extend(run_notes)
        if outcome is not None:  # FAILED after some images were checked (e.g. budget)
            issues.append(STOPPED_ISSUES.get(outcome.failure or "", STOPPED_DEFAULT))

        enhanced = state.step(NODES.index("enhance") + 1).output
        resolved = Resolved.model_validate(enhanced["resolved"])
        status = Status.DONE_WITH_ISSUES if issues else Status.DONE
        output: dict[str, Any] = {
            "image_ids": [a.image_id for a in delivered],
            # §4.3: the same delivery as documents (one image layer each, ranked). Additive:
            # `image_ids` stays until the backend has moved over (an open owner decision).
            "documents": [
                _document(a, rank).model_dump() for rank, a in enumerate(delivered, start=1)
            ],
            "quality_score": best.quality_score,
            "issues": issues,
            "assumptions": resolved.assumptions,
            "text_wanted": list(enhanced.get("text_wanted", [])),
            "width": best.width,
            "height": best.height,
        }
        if ctx.n > 1:
            output["quality_scores"] = [a.quality_score for a in delivered]
        # One image: the enhancer's plain-language line about what it shows (the ranking Critic
        # writes no user-facing summary, to keep its reply short).
        summary = (
            str(enhanced.get("summary") or best.summary or "") or MULTI_SUMMARIES[status]
            if ctx.n == 1
            else MULTI_SUMMARIES[status]
        )
        return Outcome(status, None, output, summary)

    async def _orphan_unchecked(self, ctx: RunContext) -> None:
        """A generated image the Critic never ruled on is never delivered (spec §10)."""
        for a in ctx.state.attempts:
            if a.image_id and a.verdict is None:
                await self._mark_orphaned(ctx, a.image_id, ORPHAN_UNCHECKED)

    # -- helpers --------------------------------------------------------------------------------

    def _resolve(self, enhanced: dict[str, Any], ctx: RunContext) -> Resolved:
        """Fill preset-kind gaps from preset defaults (spec §8.3), by `kind`. Questions only the
        user can answer are kept; in auto mode they are recorded as assumptions instead and
        never escalated."""
        presets = self._cfg.presets
        chosen = dict(ctx.settings)
        assumptions: list[str] = []
        questions: list[str] = []
        style_from_default = False
        for item in enhanced.get("missing_info", []):
            kind, question = item.get("kind"), item.get("question")
            if kind in PRESET_KINDS:
                if kind in chosen:
                    continue
                if kind == "aspect_ratio":
                    chosen["aspect_ratio"] = presets.defaults.aspect_ratio
                elif kind == "style":
                    chosen["style"] = presets.defaults.style
                    style_from_default = True
                assumptions.append(f"{kind}: {self._describe_default(kind)}")
            elif question:
                questions.append(question)

        brief: dict[str, Any] = dict(enhanced.get("brief") or {})
        # The user's setting wins; then the Brief's own pick (it never defaults to 1:1); then
        # the preset default.
        ratio = (
            chosen.get("aspect_ratio")
            or (
                brief.get("aspect_ratio")
                if brief.get("aspect_ratio") in presets.aspect_ratios
                else None
            )
            or presets.defaults.aspect_ratio
        )
        if ratio not in presets.aspect_ratios:
            assumptions.append(
                f"aspect_ratio: {presets.defaults.aspect_ratio} ({ratio} is not supported)"
            )
            ratio = presets.defaults.aspect_ratio
        base = presets.aspect_ratios[ratio]
        scale = RESOLUTION_SCALE.get(chosen.get("resolution", "1K"), 1)  # validated upstream
        size = base.model_copy(update={"width": base.width * scale, "height": base.height * scale})

        prompt = str(enhanced["enhanced_prompt"])
        if style_from_default:  # a user-chosen style already reached the Enhancer in settings
            prompt = f"{prompt} Style: {chosen['style']}."
            if brief:  # the gateway compiles the Brief, so the default style rides along in it
                brief["detail_anchors"] = [
                    *brief.get("detail_anchors", []),
                    f"{chosen['style']} style",
                ]
        if questions and ctx.auth.execution_mode == "auto":
            assumptions.extend(f"proceeded without an answer to: {q}" for q in questions)
            questions = []
        return Resolved(
            prompt=prompt,
            brief=brief or None,
            width=size.width,
            height=size.height,
            aspect_ratio=ratio,
            assumptions=assumptions,
            questions=questions,
        )

    def _describe_default(self, kind: str) -> str:
        d = self._cfg.presets.defaults
        if kind == "size":
            return f"{d.size.width}x{d.size.height}"
        return str(getattr(d, kind))

    def _shape_issues(self, generated: dict[str, Any], resolved: Resolved) -> list[str]:
        """Requested vs returned shape. Reported only when a PAID provider served the image;
        on the free path it is logged and ignored (user's decision, 2026-11-19)."""
        width, height = int(generated["width"]), int(generated["height"])
        wanted, got = resolved.width / resolved.height, width / height
        if abs(wanted - got) <= self._policy.aspect_ratio_tolerance:
            return []
        if generated.get("provider") in self._paid:
            return [f"requested {resolved.aspect_ratio}, got {width}x{height}"]
        log.info(
            "supervisor.shape_mismatch_ignored",
            requested=resolved.aspect_ratio,
            got=f"{width}x{height}",
            provider=generated.get("provider"),
        )
        return []

    async def _mark_orphaned(self, ctx: RunContext, image_id: str, reason: str) -> None:
        ctx.trace.orphan(image_id, reason)
        try:
            await self._artifacts.mark_orphaned(image_id, reason)
        except Exception:  # bookkeeping must never change the outcome
            log.exception("supervisor.mark_orphaned_failed", reason=reason)

    def _emit(
        self, ctx: RunContext, step: PlanStep, phase: Phase, image_id: str | None = None
    ) -> None:
        event = make_event(ctx.state.run_id, step.step, step.public_label, phase, image_id)
        ctx.state.events.append(event.model_dump(exclude_none=True))
        safe_emit(self._events, event)

    async def _save(self, ctx: RunContext) -> None:
        if self._store is not None:
            await self._store.save(ctx.state)


def _scores(record: AttemptRecord) -> dict[str, int]:
    """The Critic's four dimensions and their mean as `quality_score`."""
    if record.quality_score is None:
        return {}
    return {**record.scores, "quality_score": record.quality_score}


def _document(record: AttemptRecord, rank: int) -> Document:
    """A delivered image as a one-layer document (version 1). Deterministic ids: the document is
    named after its image, so the same run always describes the same documents."""
    image_id = str(record.image_id)
    suffix = image_id.removeprefix("img_")
    return Document(
        document_id=f"doc_{suffix}",
        size=DocumentSize(w=record.width or 1, h=record.height or 1),
        layers=[Layer(id=f"lyr_{suffix}_1", type="image", image_id=image_id, role="base")],
        scores=_scores(record),
        rank=rank,
    )


def _step_failure(records: list[AttemptRecord]) -> str | None:
    """Why a step produced nothing, as one failure kind: a refusal wins, then whichever came
    first."""
    failures = [r.failure for r in records if r.failure]
    if "content_refused" in failures:
        return "content_refused"
    return failures[0] if failures else None


def _failure_reason(doing: str, env: Envelope) -> str:
    failure = env.output.get("failure")
    return f"could not {doing}" + (f" ({failure})" if failure else "")


def create_image_create_agent(
    cfg: CreateConfig,
    *,
    llm: LLMGateway,
    images: ImageBackend,
    artifacts: ArtifactStore,
    events: EventSink,
    run_store: RunStore | None = None,
    paid_providers: Iterable[str] | None = None,
    allow_paid: bool | None = None,
    price_lookup: Callable[[Plan], float | None] | None = None,
    seed_source: Callable[[], int] | None = None,
    trace: Trace | None = None,
) -> ImageCreateAgent:
    """Wire the supervisor and its step agents from config. Ports are passed in: nothing here
    builds a concrete store or sink. `images` is the gateway's image backend (Gateway), which
    ModelGateway wraps for the Image Creator."""
    enhancer = Brief(
        AgentConfig.from_policy("prompt_enhancer", cfg.agent("prompt_enhancer"), EnhancerOutput),
        llm,
        events,
    )
    creator = ImageGenModel(
        AgentConfig.from_policy("image_creator", cfg.agent("image_creator"), None),
        events,
        images=ModelGateway(images, chain_resolver(cfg.agents)),
        artifacts=artifacts,
    )
    critic = Critic(
        AgentConfig.from_policy("critic", cfg.agent("critic"), CriticOutput),
        llm,
        events,
        artifacts=artifacts,
    )
    return ImageCreateAgent(
        AgentConfig.from_policy("image_grand_agent", cfg.agent("image_grand_agent"), None),
        llm,
        events,
        create=cfg,
        enhancer=enhancer,
        creator=creator,
        critic=critic,
        artifacts=artifacts,
        run_store=run_store,
        paid_providers=get_settings().paid_providers if paid_providers is None else paid_providers,
        allow_paid=get_settings().allow_paid if allow_paid is None else allow_paid,
        price_lookup=price_lookup,
        seed_source=seed_source,
        trace=trace,
    )


__all__ = ["ImageCreateAgent", "RunContext", "create_image_create_agent", "derive_seed"]
