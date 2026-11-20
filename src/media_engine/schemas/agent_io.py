"""Structured outputs of the image-creation agents. Every LLM output is parsed into one of these."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_MUST_HAVE = 6  # spec: must_have holds 1..6 items


Safety = Literal["allow", "block", "review"]
Route = Literal["simple", "complex"]
TaskType = Literal["create", "edit", "merge"]
MAX_SAFETY_CATEGORIES = 5


def _lower(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


class GateDecision(BaseModel):
    """The Gate's verdict (IMAGE-CREATE-OPTIMIZE §4.1). Decided ONCE per run, before any spend,
    and never re-derived. `block` ends the run; `review` asks the safety reviewer."""

    safety: Safety
    safety_categories: list[str] = Field(default_factory=list, max_length=MAX_SAFETY_CATEGORIES)
    route: Route
    task_type: TaskType = "create"
    # True when the request already names a subject, a setting and a style or camera note and
    # is over ~25 words. Recorded now; it will skip the Brief once the Brief exists (§3).
    prompt_is_detailed: bool = False

    _norm = field_validator("safety", "route", "task_type", mode="before")(_lower)


class SafetyReview(BaseModel):
    """The safety reviewer's answer for a `review` request: allow or block, nothing in between."""

    safety: Literal["allow", "block"]
    safety_categories: list[str] = Field(default_factory=list, max_length=MAX_SAFETY_CATEGORIES)

    _norm = field_validator("safety", mode="before")(_lower)


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
    "rate_limited",  # refused locally before the call: no slot under a provider's rate limit
    "bad_request",  # HTTP 400: our request is wrong
    "gateway_error",  # any other gateway failure
    "bad_image",  # the provider returned bytes that are not a known image format
    "storage",  # the artifact store failed
    "internal",  # an unexpected bug
]

Verdict = Literal["allow", "block"]

# Retry levers. Create no longer retries (IMAGE-CREATE-OPTIMIZE v2.2); the type stays only
# because RunState's AttemptRecord.lever (always None now) is part of the stored-state contract.
Lever = Literal["reseed", "re_enhance", "switch_model", "raise_tier"]


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
    # Thumbnail sizes (long side, px) stored with the image; a failed one is simply absent.
    thumbnails: list[int] = Field(default_factory=list)


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


# -- the service output: a document, not an image (IMAGE-CREATE-OPTIMIZE §4.3) ------------------


class DocumentSize(BaseModel):
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class Layer(BaseModel):
    """One layer of a document. Create emits one `image` layer with role `base`; edit replaces
    a layer (version 2), merge holds several image layers, a poster adds text layers."""

    id: str
    type: Literal["image", "text"]
    image_id: str | None = None  # for image layers
    role: str = "base"


class Document(BaseModel):
    """A created image is a document with one layer. `scores` is the Critic's quality reading
    (information only: create never acts on it); `rank` is 1 for the best."""

    document_id: str
    version: int = Field(default=1, ge=1)
    size: DocumentSize
    layers: list[Layer] = Field(min_length=1)
    scores: dict[str, int] = Field(default_factory=dict)
    rank: int = Field(ge=1)
