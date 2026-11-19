"""Load and validate the image-creation harness config (agents, policy, presets)."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from media_engine.contracts import Plan

CONFIG_DIR = Path(__file__).with_name("config")
PRESETS_PATH = Path(__file__).parents[2] / "presets" / "library" / "image_create.yaml"
REPO_ROOT = Path(__file__).parents[4]  # src/media_engine/services/create/loader.py -> repo

PLANS: tuple[str, ...] = get_args(Plan)
Capability = Literal["structured_output", "vision", "image"]
Lever = Literal["reseed", "re_enhance", "switch_model", "raise_tier"]
StepKind = Literal["route", "enhance", "generate", "critique", "supervise"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _all_plans(values: Mapping[Any, object], field: str) -> None:
    missing = [p for p in PLANS if p not in values]
    if missing:
        raise ValueError(f"{field} is missing plan(s): {missing}")


class AgentPolicy(_Strict):
    prompt: str | None = None  # None only for the no-LLM image creator
    requires: list[Capability]
    models: dict[Plan, str]  # plan -> gateway chain name

    @model_validator(mode="after")
    def _complete(self) -> AgentPolicy:
        _all_plans(self.models, "models")
        return self

    def chain_for(self, plan: Plan) -> str:
        return self.models[plan]

    def prompt_path(self) -> Path:
        if self.prompt is None:
            raise ValueError("This agent has no prompt (it makes no LLM call)")
        return REPO_ROOT / self.prompt


class BudgetPolicy(_Strict):
    tokens_total: int = Field(gt=0)
    cost_cap_usd: float = Field(ge=0.0)
    reserve_fraction: float = Field(ge=0.0, lt=1.0)


class StepEstimate(_Strict):
    tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class HarnessPolicy(_Strict):
    quality_threshold: dict[Plan, int]
    max_retries: dict[Plan, int]
    levers: dict[Lever, list[Plan]]
    max_parallel_branches: int = Field(gt=0)
    max_images_per_request: int = Field(gt=0)
    budgets: dict[Plan, BudgetPolicy]
    step_estimates: dict[StepKind, StepEstimate]
    deadlines_ms: dict[StepKind, int]

    @model_validator(mode="after")
    def _complete(self) -> HarnessPolicy:
        _all_plans(self.quality_threshold, "quality_threshold")
        _all_plans(self.max_retries, "max_retries")
        _all_plans(self.budgets, "budgets")
        for name, table in (
            ("step_estimates", self.step_estimates),
            ("deadlines_ms", self.deadlines_ms),
        ):
            missing = [s for s in get_args(StepKind) if s not in table]
            if missing:
                raise ValueError(f"{name} is missing step(s): {missing}")
        missing_levers = [lv for lv in get_args(Lever) if lv not in self.levers]
        if missing_levers:
            raise ValueError(f"levers is missing: {missing_levers}")
        return self

    def levers_for(self, plan: Plan) -> list[Lever]:
        return [lever for lever, plans in self.levers.items() if plan in plans]


class Size(_Strict):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class PresetDefaults(_Strict):
    aspect_ratio: str
    style: str
    size: Size


class Presets(_Strict):
    defaults: PresetDefaults
    aspect_ratios: dict[str, Size]

    @model_validator(mode="after")
    def _default_ratio_known(self) -> Presets:
        if self.defaults.aspect_ratio not in self.aspect_ratios:
            raise ValueError(f"default aspect_ratio {self.defaults.aspect_ratio!r} not listed")
        return self


class CreateConfig(_Strict):
    agents: dict[str, AgentPolicy]
    policy: HarnessPolicy
    presets: Presets

    def agent(self, name: str) -> AgentPolicy:
        try:
            return self.agents[name]
        except KeyError:
            raise KeyError(
                f"No agent {name!r} in agents.yaml; known: {sorted(self.agents)}"
            ) from None


def _read(path: Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_create_config(
    config_dir: Path = CONFIG_DIR, presets_path: Path = PRESETS_PATH
) -> CreateConfig:
    return CreateConfig.model_validate(
        {
            "agents": _read(config_dir / "agents.yaml"),
            "policy": _read(config_dir / "harness.yaml"),
            "presets": _read(presets_path),
        }
    )


@lru_cache
def get_create_config() -> CreateConfig:
    return load_create_config()
