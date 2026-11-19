"""Image Grand Agent: the supervisor. Plain Python around a linear LangGraph attempt graph.

    run(): build RunState -> one attempt (attempt_graph) -> envelope
           TODO(M7): the retry loop wraps run_attempt(): pick a lever, run the graph again.

The LLM is used at exactly two points (choosing a retry lever, deciding escalation), and both
arrive in M7. On a first-pass success the supervisor makes ZERO LLM calls; every call in the
run is made by a step agent and counted in that step's Spend.llm_calls.

The step guard (`_begin`) is the only place the cancel and budget checks happen. It runs them
LAST, after the boundary save, with no await between the checks and the step agent's call: on
the generate step that is immediately before the paid image call.

inputs:
    prompt:   str             the user's request (required)
    settings: dict[str, str]  user-chosen values, e.g. {"aspect_ratio": "16:9"} (optional)
    n:        int             images wanted (optional, default 1; >1 arrives in M8)
    run_id / user_id: str     (optional) identifiers for RunState

output (envelope): run_id, image_ids (best first; empty unless delivered), plus on delivery
    quality_score, issues, assumptions, text_wanted, width, height. NEEDS_INPUT carries
    questions. Never image bytes, never an internal agent name.
"""

from __future__ import annotations

import secrets
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from media_engine.agents.base import Agent, AgentConfig, describe_validation_error
from media_engine.agents.critic.critic import Critic
from media_engine.agents.generation.image_creator import ImageCreator
from media_engine.agents.image_grand.attempt_graph import NODES, build_attempt_graph, run_attempt
from media_engine.agents.prompt.enhancer import PromptEnhancer
from media_engine.contracts import (
    AuthContext,
    Envelope,
    EnvelopeState,
    Goal,
    Spend,
    Status,
    StoppingCondition,
)
from media_engine.core.config import get_settings
from media_engine.core.logging import get_logger
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver
from media_engine.gateway.model_gateway import ImageBackend, ModelGateway
from media_engine.harness.run_state import PlanStep, PublicLabel, RunState, StepStatus
from media_engine.ports.artifact_store import ArtifactStore
from media_engine.ports.event_sink import EventSink, make_event, safe_emit
from media_engine.ports.run_store import RunStore
from media_engine.schemas import CriticOutput, EnhancerOutput
from media_engine.services.create.loader import CreateConfig, StepKind

log = get_logger("agents.image_grand")

PRESET_KINDS = ("aspect_ratio", "size", "style")

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
    "generate": "one stored image",
    "critique": "a quality score and a safety verdict",
}

SUMMARIES = {
    Status.DONE: "Your image is ready.",
    Status.DONE_WITH_ISSUES: "Your image is ready, with some issues noted.",
    Status.NEEDS_INPUT: "A little more information is needed before the image can be made.",
    Status.REJECTED: "This image could not be delivered because it did not pass the safety review.",
    Status.CANCELLED: "Image creation was cancelled.",
    Status.FAILED: "The image could not be created.",
}

# A step's envelope status -> its plan status. Anything else (DONE, DONE_WITH_ISSUES,
# NEEDS_INPUT, REJECTED) means the step agent did its job: "done".
_STEP_STATUS: dict[Status, StepStatus] = {Status.FAILED: "failed", Status.CANCELLED: "cancelled"}

# Why an image was stored but never delivered (ArtifactStore.mark_orphaned reasons).
ORPHAN_UNCHECKED = "unchecked"  # the Critic never gave it a verdict


class SupervisorInputs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1)
    settings: dict[str, str] = Field(default_factory=dict)
    n: Literal[1] = 1  # TODO(M8): N parallel branches
    run_id: str | None = None
    user_id: str = "anonymous"  # TODO(backend): the backend supplies the real user


@dataclass
class Outcome:
    status: Status
    reason: str | None = None
    output: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None


@dataclass
class RunContext:
    """Per-run handles for the attempt graph. RunState is the ONLY state; everything here is
    either a reference to it, an immutable input, or the outcome decided by the step guard."""

    state: RunState
    auth: AuthContext
    user_prompt: str
    settings: dict[str, str]
    outcome: Outcome | None = None  # set once; the remaining nodes then pass through

    def step(self, kind: StepKind) -> PlanStep:
        return self.state.step(NODES.index(kind) + 1)


class Resolved(BaseModel):
    """How the Enhancer's gaps were filled. Stored in the enhance step's output."""

    prompt: str  # what the Image Creator is sent
    width: int
    height: int
    aspect_ratio: str
    assumptions: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)  # unanswered user questions


class ImageGrandAgent(Agent):
    def __init__(
        self,
        config: AgentConfig,
        gateway: LLMGateway | None,
        events: EventSink,
        *,
        create: CreateConfig,
        enhancer: PromptEnhancer,
        creator: ImageCreator,
        critic: Critic,
        artifacts: ArtifactStore,
        run_store: RunStore | None = None,
        paid_providers: Iterable[str] = (),
        seed_source: Callable[[], int] | None = None,
    ) -> None:
        # The gateway is for the two M7 decision points only; nothing in M6 calls it.
        super().__init__(config, gateway, events)
        self._cfg = create
        self._policy = create.policy
        self._enhancer = enhancer
        self._creator = creator
        self._critic = critic
        self._artifacts = artifacts
        self._store = run_store
        self._paid = frozenset(paid_providers)
        self._seed = seed_source or (lambda: secrets.randbelow(self._policy.max_seed + 1))
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

        await run_attempt(self.graph, state.run_id, ctx)  # TODO(M7): retry loop wraps this

        outcome = ctx.outcome or Outcome(Status.FAILED, "the attempt ended without an outcome")
        for item in state.plan:  # nothing is removed: undone steps are marked skipped
            if item.status == "pending":
                item.status = "skipped"
        state.status = outcome.status
        await self._save(ctx)
        log.info(
            "supervisor.done",
            run_id=state.run_id,
            status=outcome.status.value,
            llm_calls=self.run_spend(state).llm_calls,
        )
        return Envelope(
            output={"run_id": state.run_id, "image_ids": [], **outcome.output},
            summary=outcome.summary or SUMMARIES[outcome.status],
            state=EnvelopeState(
                status=outcome.status, spend=self.run_spend(state), reason=outcome.reason
            ),
        )

    def new_context(self, auth: AuthContext, inputs: dict[str, Any]) -> RunContext:
        """A fresh RunState (every step pending) and its context. Raises ValidationError."""
        args = SupervisorInputs.model_validate(inputs)
        state = RunState(
            run_id=args.run_id or f"run_{uuid.uuid4().hex}",
            user_id=args.user_id,
            auth=auth,
            budget=self._policy.budget_for(auth.plan),
            plan=[
                PlanStep(step=i, agent=agent, public_label=label)
                for i, (_, agent, label) in enumerate(PLAN, start=1)
            ],
        )
        return RunContext(state, auth, args.prompt, dict(args.settings))

    @staticmethod
    def run_spend(state: RunState) -> Spend:
        """The run's spend: the sum over the plan. The supervisor's own calls (M7) will be a
        plan step too, so this stays the single place LLM calls are counted."""
        total = Spend()
        for item in state.plan:
            total = total.plus(item.spend)
        return total

    # -- the step guard: the only place cancel and budget are checked ----------------------------

    async def _begin(self, ctx: RunContext, kind: StepKind) -> PlanStep | None:
        """Boundary save, then the checks, then `running`. Returns the step to run, or None
        when the outcome is already decided (the node then returns: one way out, END).
        No await after the checks: the caller's next await is the step agent itself."""
        if ctx.outcome is not None:
            return None
        step = ctx.step(kind)
        await self._save(ctx)  # step boundary: RunState is JSON-complete here
        if ctx.state.cancel_requested:
            await self._cancel(ctx, step)
            return None
        if not self._affordable(ctx, kind):
            await self._refuse_budget(ctx, step, kind)
            return None
        step.status = "running"
        self._emit(ctx, step, "started")
        return step

    async def _finish(self, ctx: RunContext, step: PlanStep, env: Envelope) -> None:
        step.spend = step.spend.plus(env.state.spend)
        ctx.state.budget.charge(env.state.spend)
        step.status = _STEP_STATUS.get(env.state.status, "done")
        self._emit(ctx, step, "finished")
        await self._save(ctx)

    def _affordable(self, ctx: RunContext, kind: StepKind) -> bool:
        kinds: tuple[StepKind, ...] = (kind,)
        if kind == "generate":
            # An image that can't then be checked can't be delivered: refuse before paying
            # for it unless the critique that must follow is affordable too.
            kinds = ("generate", "critique")
        est = [self._policy.step_estimates[k] for k in kinds]
        return ctx.state.budget.can_afford(
            sum(e.tokens for e in est),
            sum(e.cost_usd for e in est),
            sum(e.neurons for e in est),
        )

    async def _refuse_budget(self, ctx: RunContext, step: PlanStep, kind: StepKind) -> None:
        log.warning("supervisor.budget_refused", run_id=ctx.state.run_id, step=kind)
        step.status = "skipped"
        await self._orphan_unchecked(ctx)
        ctx.outcome = Outcome(Status.FAILED, f"budget would be exceeded before step {step.step}")

    async def _cancel(self, ctx: RunContext, step: PlanStep) -> None:
        """Freeze at a step boundary: done steps keep outputs and spend, this and later steps
        are skipped (never removed). TODO(M9): in-flight cancel and branch partial outputs."""
        ctx.state.cancelled_at_step = step.step
        step.status = "skipped"
        await self._orphan_unchecked(ctx)
        ctx.outcome = Outcome(Status.CANCELLED, f"cancelled before step {step.step}")

    async def _orphan_unchecked(self, ctx: RunContext) -> None:
        """A generated image the Critic never ruled on is never delivered (spec §10)."""
        gen, crit = ctx.step("generate"), ctx.step("critique")
        if gen.output_ref and crit.status != "done":
            await self._mark_orphaned(gen.output_ref, ORPHAN_UNCHECKED)

    def _on_crash(self, ctx: RunContext, node: str, err: Exception) -> None:
        for item in ctx.state.plan:
            if item.status == "running":
                item.status = "failed"
        if ctx.outcome is None:
            ctx.outcome = Outcome(Status.FAILED, f"internal error: {type(err).__name__}")

    # -- nodes ----------------------------------------------------------------------------------

    def _stop(self, ctx: RunContext, kind: StepKind) -> StoppingCondition:
        return self._policy.stopping_condition(kind, ctx.auth.plan, SUCCESS[kind])

    async def _enhance(self, ctx: RunContext) -> None:
        step = await self._begin(ctx, "enhance")
        if step is None:
            return
        env = await self._enhancer.run(
            Goal(task="enhance", instructions="Write a detailed image prompt."),
            self._stop(ctx, "enhance"),
            ctx.auth,
            {"prompt": ctx.user_prompt, "settings": ctx.settings},
        )
        if env.state.status in (Status.DONE, Status.NEEDS_INPUT):
            resolved = self._resolve(env.output, ctx)
            step.output = {**env.output, "resolved": resolved.model_dump()}
            step.output_ref = f"{ctx.state.run_id}:enhance"
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
        ctx.outcome = Outcome(Status.FAILED, _failure_reason("prepare the request", env))

    async def _generate(self, ctx: RunContext) -> None:
        step = await self._begin(ctx, "generate")
        if step is None:
            return
        resolved = Resolved.model_validate(ctx.step("enhance").output["resolved"])
        env = await self._creator.run(
            Goal(task="generate", instructions="Create the image."),
            self._stop(ctx, "generate"),
            ctx.auth,
            {
                "prompt": resolved.prompt,
                "width": resolved.width,
                "height": resolved.height,
                "seed": self._seed(),
                "tier": ctx.auth.plan,
            },
        )
        step.output = dict(env.output)
        step.output_ref = env.output.get("image_id")
        await self._finish(ctx, step, env)
        if env.state.status is Status.DONE:
            return
        if env.state.status is Status.CANCELLED:
            ctx.state.cancelled_at_step = step.step
            ctx.outcome = Outcome(Status.CANCELLED, env.state.reason)
            return
        # TODO(M7): content_refused -> re_enhance is the only permitted lever.
        ctx.outcome = Outcome(Status.FAILED, _failure_reason("create the image", env))

    async def _critique(self, ctx: RunContext) -> None:
        step = await self._begin(ctx, "critique")
        if step is None:
            return
        enhanced, generated = ctx.step("enhance").output, ctx.step("generate")
        image_id = generated.output_ref
        env = await self._critic.run(
            Goal(task="critique", instructions="Review the image."),
            self._stop(ctx, "critique"),
            ctx.auth,
            {"image_id": image_id, "prompt": ctx.user_prompt, "must_have": enhanced["must_have"]},
        )
        step.output = dict(env.output)
        step.output_ref = image_id
        await self._finish(ctx, step, env)

        if env.state.status is Status.REJECTED:
            # Terminal, whatever the score, plan or retries left. The Critic marked the image
            # orphaned; it is not delivered and no retry path starts from here.
            ctx.outcome = Outcome(Status.REJECTED, env.state.reason)
            return
        if env.state.status is not Status.DONE:
            if image_id:
                await self._mark_orphaned(image_id, ORPHAN_UNCHECKED)
            ctx.outcome = Outcome(Status.FAILED, _failure_reason("check the image", env))
            return

        resolved = Resolved.model_validate(enhanced["resolved"])
        score = int(env.output["quality_score"])
        issues: list[str] = []
        if score < self._policy.quality_threshold[ctx.auth.plan]:
            issues.append(str(env.output["critique"]))  # TODO(M7): retry before settling
        issues.extend(self._shape_issues(generated.output, resolved))
        ctx.outcome = Outcome(
            Status.DONE_WITH_ISSUES if issues else Status.DONE,
            None,
            {
                "image_ids": [image_id],
                "quality_score": score,
                "issues": issues,
                "assumptions": resolved.assumptions,
                "text_wanted": list(enhanced.get("text_wanted", [])),
                "width": generated.output["width"],
                "height": generated.output["height"],
            },
            env.summary,
        )

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

        ratio = chosen.get("aspect_ratio") or presets.defaults.aspect_ratio
        if ratio not in presets.aspect_ratios:
            assumptions.append(
                f"aspect_ratio: {presets.defaults.aspect_ratio} ({ratio} is not supported)"
            )
            ratio = presets.defaults.aspect_ratio
        size = presets.aspect_ratios[ratio]

        prompt = str(enhanced["enhanced_prompt"])
        if style_from_default:  # a user-chosen style already reached the Enhancer in settings
            prompt = f"{prompt} Style: {chosen['style']}."
        if questions and ctx.auth.execution_mode == "auto":
            assumptions.extend(f"proceeded without an answer to: {q}" for q in questions)
            questions = []
        return Resolved(
            prompt=prompt,
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

    async def _mark_orphaned(self, image_id: str, reason: str) -> None:
        try:
            await self._artifacts.mark_orphaned(image_id, reason)
        except Exception:  # bookkeeping must never change the outcome
            log.exception("supervisor.mark_orphaned_failed", reason=reason)

    def _emit(self, ctx: RunContext, step: PlanStep, phase: Literal["started", "finished"]) -> None:
        event = make_event(ctx.state.run_id, step.step, step.public_label, phase)
        ctx.state.events.append(event.model_dump())
        safe_emit(self._events, event)

    async def _save(self, ctx: RunContext) -> None:
        if self._store is not None:
            await self._store.save(ctx.state)


def _failure_reason(doing: str, env: Envelope) -> str:
    failure = env.output.get("failure")
    return f"could not {doing}" + (f" ({failure})" if failure else "")


def create_image_grand_agent(
    cfg: CreateConfig,
    *,
    llm: LLMGateway,
    images: ImageBackend,
    artifacts: ArtifactStore,
    events: EventSink,
    run_store: RunStore | None = None,
    paid_providers: Iterable[str] | None = None,
    seed_source: Callable[[], int] | None = None,
) -> ImageGrandAgent:
    """Wire the supervisor and its step agents from config. Ports are passed in: nothing here
    builds a concrete store or sink. `images` is the gateway's image backend (Gateway), which
    ModelGateway wraps for the Image Creator."""
    enhancer = PromptEnhancer(
        AgentConfig.from_policy("prompt_enhancer", cfg.agent("prompt_enhancer"), EnhancerOutput),
        llm,
        events,
    )
    creator = ImageCreator(
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
    return ImageGrandAgent(
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
        seed_source=seed_source,
    )


__all__ = ["ImageGrandAgent", "RunContext", "create_image_grand_agent"]
