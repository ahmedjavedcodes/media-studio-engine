"""Guardrail data types. See execution-steps/GUARDRAIL-AGENT-SPEC.md section 6."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Decision(StrEnum):
    APPROVE = "APPROVE"
    STANDARD_REJECT = "STANDARD_REJECT"
    INSTANT_REJECT = "INSTANT_REJECT"


class Category(StrEnum):
    SAFE = "SAFE"
    CSAM = "CSAM"
    NCII_UNDRESSING = "NCII_UNDRESSING"
    GORE_SELF_HARM = "GORE_SELF_HARM"
    TERRORISM_WEAPONS = "TERRORISM_WEAPONS"
    NSFW_EXPLICIT = "NSFW_EXPLICIT"
    CELEBRITY_LIKENESS = "CELEBRITY_LIKENESS"
    HATE_HARASSMENT = "HATE_HARASSMENT"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    PII_WATERMARK = "PII_WATERMARK"
    OTHER_UNSAFE = "OTHER_UNSAFE"


class Tier(StrEnum):
    CRITICAL = "CRITICAL"
    SEVERE = "SEVERE"
    STANDARD = "STANDARD"
    NONE = "NONE"


AccountAction = Literal["none", "warn", "cooldown_1m", "cooldown_2m", "cooldown_30m", "ban"]


class GuardInput(BaseModel):
    request_id: str
    user_id: str
    prompt: str
    images: list[bytes] = Field(default_factory=list)


class GuardVerdict(BaseModel):
    """One per model call."""

    source: Literal["jev", "vision"]
    decision: Decision
    category: Category
    confidence: float | None  # None = model gave no score
    reason: str  # internal log only
    scores: dict[str, float] = Field(default_factory=dict)
    needs_review: bool = False
    model: str
    latency_ms: int


class GuardDecision(BaseModel):
    """Final answer of the agent."""

    decision: Decision
    category: Category
    confidence: float | None
    verdicts: list[GuardVerdict]
    account_action: AccountAction = "none"
    strike_count: int | None = None  # instant rejects only: strikes in the last 30 days
    retry_after: int | None = None  # seconds until the user can try again, if a cooldown applies
    user_message: str | None = None
