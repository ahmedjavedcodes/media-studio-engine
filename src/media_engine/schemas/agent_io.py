"""Structured outputs of the image-creation agents. Every LLM output is parsed into one of these."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_MUST_HAVE = 6  # spec: must_have holds 1..6 items


class RouteDecision(BaseModel):
    route: Literal["simple", "complex"]


# Preset-resolvable gaps use their key as `kind`; anything else is a question for the user.
MissingKind = Literal["aspect_ratio", "size", "style", "question"]


class MissingItem(BaseModel):
    kind: MissingKind
    question: str | None = None  # required when kind == "question"; optional context otherwise

    @model_validator(mode="after")
    def _question_needs_text(self) -> MissingItem:
        if self.kind == "question" and not (self.question and self.question.strip()):
            raise ValueError("a missing_info item of kind 'question' needs question text")
        return self


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


class EnhancerOutput(BaseModel):
    enhanced_prompt: str = Field(min_length=1)
    # Testable requirements taken ONLY from the user's own words, never the enhancer's additions.
    must_have: list[str] = Field(min_length=1, max_length=MAX_MUST_HAVE)
    missing_info: list[MissingItem] = Field(default_factory=list)  # non-empty -> NEEDS_INPUT
    # Words the user wants shown in the image. The image model cannot render text, so these are
    # kept out of enhanced_prompt and handed on for a separate text layer.
    text_wanted: list[str] = Field(default_factory=list)
    summary: str = ""  # envelope summary, returned in the same call (no extra LLM call)

    @field_validator("must_have", mode="before")
    @classmethod
    def _dedupe_and_cap(cls, value: Any) -> Any:
        """Drop blanks and case/space-insensitive duplicates, then keep the first six rather
        than failing validation (and burning the repair) on a seventh."""
        if not isinstance(value, list):
            return value
        seen: set[str] = set()
        kept: list[Any] = []
        for item in value:
            if not isinstance(item, str):
                kept.append(item)  # let normal validation reject it
                continue
            key = _normalise(item)
            if key and key not in seen:
                seen.add(key)
                kept.append(item.strip())
        return kept[:MAX_MUST_HAVE]

    @field_validator("text_wanted", mode="before")
    @classmethod
    def _strip_blank_text(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [t.strip() for t in value if not isinstance(t, str) or t.strip()]


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
