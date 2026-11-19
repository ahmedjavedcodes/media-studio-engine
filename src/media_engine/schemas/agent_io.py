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


# Why a step FAILED, in `Envelope.output["failure"]`. Lets the supervisor choose what to do
# (re-enhance after a refusal, give up when no provider is left) without parsing reason text.
FailureKind = Literal[
    "invalid_input",  # the step was called with bad inputs: our bug
    "invalid_output",  # the model's reply failed validation, even after the one repair
    "content_refused",  # a provider's safety filter declined (may be a false positive)
    "no_provider",  # every chain entry failed or was unavailable
    "deadline",  # the step's deadline ran out
    "bad_request",  # HTTP 400: our request is wrong
    "gateway_error",  # any other gateway failure
    "bad_image",  # the provider returned bytes that are not a known image format
    "storage",  # the artifact store failed
    "internal",  # an unexpected bug
]

Verdict = Literal["allow", "block"]

# What the supervisor may change on a retry (spec §8.2). Which plans may use which lever is
# config (harness.yaml `levers`).
Lever = Literal["reseed", "re_enhance", "switch_model", "raise_tier"]


class LeverChoice(BaseModel):
    """The supervisor's one lever-choice LLM call (spec §8.2). Only the lever is acted on; the
    reason is kept for the logs."""

    lever: Lever
    reason: str = ""

    @field_validator("lever", mode="before")
    @classmethod
    def _lever_case(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


class GenerationResult(BaseModel):
    image_id: str
    provider: str
    model: str
    seed: int  # the seed sent
    # Whether the model reproduces an image from its seed (ModelCard.honours_seed). None =
    # not verified yet: the reseed lever (M7) must not rely on `seed` until this is True.
    seed_honoured: bool | None = None
    cost_usd: float
    neurons: float = 0.0
    # What came back, read from the image itself. May differ from the request: flux-1-schnell
    # is sent no width/height at all, so the supervisor must not assume the requested size.
    width: int
    height: int
    mime: str


def _normalise_verdict(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


class CriticOutput(BaseModel):
    quality_score: int = Field(ge=0, le=10)
    # Safety is a verdict, not a score: separate field, never compared with quality_score.
    safety_verdict: Verdict
    safety_reason: str | None = None
    critique: str
    summary: str = ""  # envelope summary, returned in the same call (no extra LLM call)

    @field_validator("safety_verdict", mode="before")
    @classmethod
    def _verdict_case(cls, value: Any) -> Any:
        # "Block" / " BLOCK " must still be a block, not a validation error.
        return _normalise_verdict(value)
