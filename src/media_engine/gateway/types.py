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


class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int | None = None
    seed: int | None = None
    input_images: list[bytes] = Field(default_factory=list)  # edit / merge references


class ImageResponse(BaseModel):
    images: list[bytes]  # decoded image bytes


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
    neurons: float | None = None  # Cloudflare does not report neurons per call; None = unknown
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
            "attempts": [f"{a.provider}/{a.model}:{a.outcome}" for a in self.attempts],
        }
