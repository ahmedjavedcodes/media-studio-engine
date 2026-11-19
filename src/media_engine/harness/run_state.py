"""Run state for one harness run. Must serialise to plain JSON at every point: no live handles,
no open connections, nothing that `model_dump_json()` cannot represent."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

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


class Budget(BaseModel):
    """Per-run budget. A `reserve_fraction` of each cap is held back for retries: ordinary steps
    may only spend the rest; retries may also draw on the reserve."""

    tokens_total: int
    cost_cap_usd: float
    reserve_fraction: float = Field(default=0.30, ge=0.0, lt=1.0)
    tokens_spent: int = 0
    cost_spent_usd: float = 0.0

    def remaining_for_steps(self) -> tuple[int, float]:
        usable_tokens = int(self.tokens_total * (1 - self.reserve_fraction))
        usable_cost = self.cost_cap_usd * (1 - self.reserve_fraction)
        return (
            max(0, usable_tokens - self.tokens_spent),
            max(0.0, usable_cost - self.cost_spent_usd),
        )

    def remaining_total(self) -> tuple[int, float]:
        return (
            max(0, self.tokens_total - self.tokens_spent),
            max(0.0, self.cost_cap_usd - self.cost_spent_usd),
        )

    def can_afford(self, est_tokens: int, est_cost: float, *, use_reserve: bool = False) -> bool:
        tokens_left, cost_left = (
            self.remaining_total() if use_reserve else self.remaining_for_steps()
        )
        return est_tokens <= tokens_left and est_cost <= cost_left

    def charge(self, spend: Spend) -> None:
        self.tokens_spent += spend.tokens
        self.cost_spent_usd += spend.cost_usd


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
