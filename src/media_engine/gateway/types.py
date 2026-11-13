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


class ImageResponse(BaseModel):
    images: list[bytes]  # decoded image bytes
    cost_usd: float | None = None  # actual USD cost when the provider reports it (OpenRouter)


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
