"""Agent model policy, as declared per agent in agents.yaml."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources.abc import Traversable
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from media_engine.contracts import Plan
from media_engine.core.prompts import PROMPTS_DIR
from media_engine.gateway.types import ReasoningEffort

PLANS: tuple[str, ...] = get_args(Plan)
Capability = Literal["structured_output", "vision", "image"]


def require_all_plans(values: Mapping[Any, object], field: str) -> None:
    missing = [p for p in PLANS if p not in values]
    if missing:
        raise ValueError(f"{field} is missing plan(s): {missing}")


class AgentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Relative to media_engine/prompts/. None only for an agent that makes no LLM call.
    prompt: str | None = None
    requires: list[Capability]
    models: dict[Plan, str]  # plan -> gateway chain name
    reasoning_effort: ReasoningEffort | None = None
    # Sampling temperature sent with every call; None = the provider's default.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    params: dict[str, Any] = Field(default_factory=dict)  # per-agent tunables, agent-validated

    @model_validator(mode="after")
    def _complete(self) -> AgentPolicy:
        require_all_plans(self.models, "models")
        return self

    def chain_for(self, plan: Plan) -> str:
        return self.models[plan]

    def prompt_path(self) -> Traversable:
        if self.prompt is None:
            raise ValueError("This agent has no prompt (it makes no LLM call)")
        return PROMPTS_DIR / self.prompt
