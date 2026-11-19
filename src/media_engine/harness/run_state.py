"""Run state for one harness run. Must serialise to plain JSON at every point: no live handles,
no open connections, nothing that `model_dump_json()` cannot represent."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, Field

from media_engine.contracts import AuthContext, Spend, Status

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
    branches_total: int | None = None
    branches_done: int | None = None
    branches_aborted: int | None = None
    partial_outputs: list[str] = Field(default_factory=list)
    spend: Spend = Field(default_factory=Spend)


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
    cancel_requested: bool = False
    cancelled_at_step: int | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def step(self, number: int) -> PlanStep:
        for item in self.plan:
            if item.step == number:
                return item
        raise KeyError(f"No plan step {number} in run {self.run_id}")
