"""Request / response types shared by the gateway and providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

ReasoningEffort = Literal["none", "low", "medium", "high"]


class LLMRequest(BaseModel):
    messages: list[dict[str, Any]]
    temperature: float | None = None
    max_tokens: int | None = None
    json_mode: bool = False
    # JSON Schema of the reply. A provider that supports structured output sends it natively;
    # the others get it appended to the system prompt (`inject_schema`).
    response_schema: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    reasoning_effort: ReasoningEffort | None = None
    include_reasoning: bool = False


class LLMResponse(BaseModel):
    text: str
    reasoning: str | None = None  # keep in traces only, never show to users
    usage: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float | None = None  # actual USD cost when the provider reports it (OpenRouter)


class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int | None = None
    seed: int | None = None
    input_images: list[bytes] = Field(default_factory=list)  # edit / merge references
    n: int = 1  # images per call (OpenRouter only; Cloudflare returns one)
    aspect_ratio: str | None = None  # OpenRouter; derived from width/height when omitted
    resolution: str | None = (
        None  # OpenRouter tier ('512','768','1K','1.5K','2K','4K'); derived when omitted
    )
    # OpenRouter quality setting (e.g. gpt-image-2: low/medium/high). None = the registry's
    # default_quality for that model; never left to the model's own default.
    quality: str | None = None
    # The Brief (IMAGE-CREATE-OPTIMIZE §4.2). When set, the gateway COMPILES it into `prompt` for
    # each chain entry it tries, so a fallback gets a prompt written for the model that answers.
    brief: dict[str, Any] | None = None


class ImageResponse(BaseModel):
    images: list[bytes]  # decoded image bytes
    cost_usd: float | None = None  # actual USD cost when the provider reports it (OpenRouter)
    # One per image, read from the response or the bytes; None = unrecognised. Never assume PNG.
    media_types: list[str | None] = Field(default_factory=list)


@dataclass(frozen=True)
class Route:
    """A provider's answer to "may this chain entry serve this image request?", decided BEFORE
    any call. `model` None = no (the entry is skipped, never called; `reason` names the limit).
    `model` differs from the entry's = served by a substitute (`reason` says why)."""

    model: str | None
    reason: str | None = None


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def is_placeholder(self) -> bool:
        return self.model.startswith("<")


@dataclass
class Attempt:
    """One chain entry that was tried or skipped, in order."""

    provider: str
    model: str
    outcome: str  # "success" | "skipped" | "failed"
    reason: str | None = None
    status: int | None = None
    code: int | str | None = None
    calls: int = 0


AttemptOutcome = Literal["ok", "skipped", "rate_limited", "timeout", "refused", "error"]


class AttemptInfo(BaseModel):
    """JSON-safe view of one gateway `Attempt` (for the developer trace)."""

    provider: str
    model: str
    outcome: AttemptOutcome
    reason: str | None = None


def attempt_infos(attempts: list[Attempt]) -> list[AttemptInfo]:
    """Map the gateway's attempts (success | skipped | failed) onto the trace's outcomes."""
    infos: list[AttemptInfo] = []
    for a in attempts:
        reason = (a.reason or "").lower()
        if a.outcome == "success":
            outcome: AttemptOutcome = "ok"
        elif a.outcome == "skipped":
            outcome = "skipped"
        elif a.status == 429 or any(k in reason for k in ("rate", "exhaust", "capacity", "429")):
            outcome = "rate_limited"
        elif any(k in reason for k in ("deadline", "timeout", "timed out")):
            outcome = "timeout"
        elif any(k in reason for k in ("refus", "nsfw", "content")):
            outcome = "refused"
        else:
            outcome = "error"
        detail = a.reason
        if a.status is not None or a.code is not None:
            detail = f"{a.reason or ''} (status={a.status}, code={a.code})".strip()
        infos.append(
            AttemptInfo(provider=a.provider, model=a.model, outcome=outcome, reason=detail)
        )
    return infos


@dataclass
class CallResult:
    output: LLMResponse | ImageResponse
    task: str
    provider: str
    model: str
    latency_ms: float
    is_fallback: bool
    neurons: float | None = None  # ACTUAL neurons reported by the provider (Cloudflare LLMs)
    neurons_est: float | None = None  # registry-price estimate (images, or LLM without usage)
    daily_neurons_total: float | None = None  # Cloudflare running total today (UTC)
    cost_usd: float | None = None  # ACTUAL USD cost reported by the provider (OpenRouter)
    daily_cost_usd_total: float | None = None  # running USD total today (UTC)
    attempts: list[Attempt] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Trace-safe summary (no prompt or image content)."""
        return {
            "task": self.task,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "is_fallback": self.is_fallback,
            "neurons": self.neurons,
            "neurons_est": self.neurons_est,
            "daily_neurons_total": self.daily_neurons_total,
            "cost_usd": self.cost_usd,
            "daily_cost_usd_total": self.daily_cost_usd_total,
            "attempts": [f"{a.provider}/{a.model}:{a.outcome}" for a in self.attempts],
        }
