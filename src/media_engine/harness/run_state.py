"""Run state for one harness run. Must serialise to plain JSON at every point: no live handles,
no open connections, nothing that `model_dump_json()` cannot represent."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, Field

from media_engine.contracts import AuthContext, Plan, Spend, Status
from media_engine.schemas import Lever, Verdict

# The only strings that may reach a user. Internal agent names never appear in events/output.
PublicLabel = Literal["Understanding your request", "Creating your image", "Finishing up"]
StepStatus = Literal["pending", "running", "done", "failed", "cancelled", "skipped"]
RunStatus = Status | Literal["running"]


class PlanStep(BaseModel):
    step: int
    agent: str  # internal name: never emitted
    public_label: PublicLabel
    status: StepStatus = "pending"
    output_ref: str | None = None
    # The step agent's envelope output (refs and scalars only; Envelope already rejects bytes).
    # Kept here, not in any in-memory context, so RunState alone can resume or retry a run
    # (completed steps keep their outputs). Not in the spec's PlanStep: added for M6/M7.
    output: dict[str, Any] = Field(default_factory=dict)
    branches_total: int | None = None
    branches_done: int | None = None
    branches_aborted: int | None = None
    partial_outputs: list[str] = Field(default_factory=list)
    spend: Spend = Field(default_factory=Spend)


class AttemptRecord(BaseModel):
    """One image of one pass through enhance -> generate -> critique (one record per branch per
    round), kept in RunState so a retry can resume from it and the final answer can rank every
    image. Not in the spec's RunState: added in M7
    (a PlanStep holds one output per step, not one per attempt). JSON-safe, refs and scalars only.

    The first group is what the attempt was run WITH (the knobs a lever changes; they carry over
    to the next attempt). The second is what came out of it."""

    number: int  # 1-based: the round (one pass of the graph); all N branches of a round share it
    branch: int = 0  # 0-based image within the round (M8); a single-image run only has 0
    lever: Lever | None = None  # the lever that started this attempt; None for the first
    seed: int
    tier: Plan  # the chain tier sent to the Image Creator (raise_tier moves it up)
    chain_offset: int = 0  # leading chain entries skipped (switch_model moves it on)
    feedback: str | None = None  # what the Enhancer is told about the last attempt (re_enhance)

    image_id: str | None = None  # set once an image was stored
    provider: str | None = None
    width: int | None = None
    height: int | None = None
    shape_issues: list[str] = Field(default_factory=list)
    verdict: Verdict | None = None  # None: the Critic never ruled on it, so it is never delivered
    quality_score: int | None = None
    critique: str | None = None
    summary: str | None = None
    failure: str | None = None  # why the attempt ended without a verdict, if it did


class Remaining(NamedTuple):
    tokens: int
    cost_usd: float
    neurons: float


class Budget(BaseModel):
    """Per-run budget on three meters: tokens, USD and Cloudflare neurons. Neurons are the meter
    that actually moves on the free path (images cost neurons, not dollars), so image generation
    is capped by `neurons_total`.

    A `reserve_fraction` of each cap is held back for retries: ordinary steps may only spend the
    rest; retries may also draw on the reserve. The fraction comes from harness.yaml."""

    tokens_total: int
    cost_cap_usd: float
    neurons_total: float
    reserve_fraction: float = Field(ge=0.0, lt=1.0)
    tokens_spent: int = 0
    cost_spent_usd: float = 0.0
    neurons_spent: float = 0.0

    def remaining_for_steps(self) -> Remaining:
        usable = 1 - self.reserve_fraction
        return Remaining(
            max(0, int(self.tokens_total * usable) - self.tokens_spent),
            max(0.0, self.cost_cap_usd * usable - self.cost_spent_usd),
            max(0.0, self.neurons_total * usable - self.neurons_spent),
        )

    def remaining_total(self) -> Remaining:
        return Remaining(
            max(0, self.tokens_total - self.tokens_spent),
            max(0.0, self.cost_cap_usd - self.cost_spent_usd),
            max(0.0, self.neurons_total - self.neurons_spent),
        )

    def can_afford(
        self,
        est_tokens: int,
        est_cost: float,
        est_neurons: float = 0.0,
        *,
        use_reserve: bool = False,
    ) -> bool:
        left = self.remaining_total() if use_reserve else self.remaining_for_steps()
        return (
            est_tokens <= left.tokens and est_cost <= left.cost_usd and est_neurons <= left.neurons
        )

    def charge(self, spend: Spend) -> None:
        self.tokens_spent += spend.tokens
        self.cost_spent_usd += spend.cost_usd
        self.neurons_spent += spend.neurons


class RunState(BaseModel):
    run_id: str
    user_id: str
    status: RunStatus = "running"
    auth: AuthContext
    budget: Budget
    plan: list[PlanStep] = Field(default_factory=list)
    replans_used: int = 0
    attempts: list[AttemptRecord] = Field(default_factory=list)  # one per attempt, oldest first
    retries_used: int = 0  # levers applied so far; capped per plan (harness.yaml max_retries)
    critic_reasks_used: int = 0  # Critic re-asks on the same image (harness.yaml max_critic_reasks)
    # The supervisor's own LLM calls (the lever choice). Kept apart from the plan so that "the
    # supervisor made zero LLM calls" can be read straight off RunState; run_spend adds it in.
    supervisor_spend: Spend = Field(default_factory=Spend)
    cancel_requested: bool = False
    cancelled_at_step: int | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def attempt(self) -> AttemptRecord:
        """The newest record. Every record of a round shares the knobs a lever changes."""
        return self.attempts[-1]

    @property
    def round(self) -> list[AttemptRecord]:
        """The records of the round in progress (the newest), one per branch."""
        number = self.attempts[-1].number
        return [a for a in self.attempts if a.number == number]

    def step(self, number: int) -> PlanStep:
        for item in self.plan:
            if item.step == number:
                return item
        raise KeyError(f"No plan step {number} in run {self.run_id}")
