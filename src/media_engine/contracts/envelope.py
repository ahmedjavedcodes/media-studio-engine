"""Delegation contracts. Every delegation passes Goal + StoppingCondition + AuthContext;
every agent returns an Envelope."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

Plan = Literal["free", "pro", "max", "ultra"]
ExecutionMode = Literal["auto", "manual"]
ModelDemand = Literal["cheap", "balanced", "best"]


class Status(StrEnum):
    DONE = "done"
    DONE_WITH_ISSUES = "done_with_issues"
    NEEDS_INPUT = "needs_input"
    ESCALATE = "escalate"
    REJECTED = "rejected"  # safety block: terminal
    CANCELLED = "cancelled"
    FAILED = "failed"


class Goal(BaseModel):
    task: str
    instructions: str


class StoppingCondition(BaseModel):
    success: str
    max_tokens: int
    max_retries: int
    deadline_ms: int


class AuthContext(BaseModel):
    """Who is asking and what they may use. Frozen: if code needs to change it, the design is
    wrong."""

    model_config = ConfigDict(frozen=True)

    plan: Plan
    execution_mode: ExecutionMode
    model_demand: ModelDemand


class Spend(BaseModel):
    tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    retries_used: int = 0
    neurons: float = 0.0  # Cloudflare free-tier units; cost_usd stays ~0 on the free path

    def plus(self, other: Spend) -> Spend:
        return Spend(
            tokens=self.tokens + other.tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            duration_ms=self.duration_ms + other.duration_ms,
            retries_used=self.retries_used + other.retries_used,
            neurons=self.neurons + other.neurons,
        )


class EnvelopeState(BaseModel):
    status: Status
    spend: Spend
    reason: str | None = None


# Images travel as image_id strings resolved through the ArtifactStore, never inline.
_DATA_URI = re.compile(r"^\s*data:[\w/+.-]+;base64,", re.IGNORECASE)
_BASE64_BLOB = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
# A pure-base64 string longer than this is treated as encoded binary. Prose never matches the
# pattern (it has spaces/punctuation), and ids/hashes are far shorter.
MAX_INLINE_BASE64_CHARS = 512


def _reject_binary(value: Any, path: str) -> None:
    if isinstance(value, bytes | bytearray | memoryview):
        # ValueError, not TypeError: pydantic only turns ValueError into a ValidationError.
        raise ValueError(f"output{path} carries raw bytes; store it and pass an image_id")  # noqa: TRY004
    if isinstance(value, str):
        if _DATA_URI.match(value) or (
            len(value) > MAX_INLINE_BASE64_CHARS and _BASE64_BLOB.match(value)
        ):
            raise ValueError(f"output{path} looks like base64 data; pass an image_id instead")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_binary(item, f"{path}[{key!r}]")
    elif isinstance(value, list | tuple | set | frozenset):
        for index, item in enumerate(value):
            _reject_binary(item, f"{path}[{index}]")


class Envelope(BaseModel):
    output: dict[str, Any]  # references and scalars ONLY, never image bytes
    summary: str  # <= 2 sentences
    state: EnvelopeState

    @field_validator("output")
    @classmethod
    def _no_image_bytes(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_binary(value, "")
        return value
