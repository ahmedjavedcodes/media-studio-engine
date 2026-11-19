"""Structured outputs of the image-creation agents. Every LLM output is parsed into one of these."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: Literal["simple", "complex"]


class EnhancerOutput(BaseModel):
    enhanced_prompt: str
    # Testable requirements taken ONLY from the user's own words, never the enhancer's additions.
    must_have: list[str] = Field(min_length=1, max_length=6)
    missing_info: list[str] = Field(default_factory=list)  # non-empty -> NEEDS_INPUT
    summary: str = ""  # envelope summary, returned in the same call (no extra LLM call)


class GenerationResult(BaseModel):
    image_id: str
    provider: str
    model: str
    seed: int
    cost_usd: float
    neurons: float = 0.0


class CriticOutput(BaseModel):
    quality_score: int = Field(ge=0, le=10)
    # Safety is a verdict, not a score: separate field, never compared with quality_score.
    safety_verdict: Literal["allow", "block"]
    safety_reason: str | None = None
    critique: str
    summary: str = ""  # envelope summary, returned in the same call (no extra LLM call)
