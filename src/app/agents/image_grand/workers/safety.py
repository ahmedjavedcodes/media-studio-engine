"""Per-image safety check. Spec v1.3 section 4 [5].

Each image is checked as soon as it is made, on its 256 px thumbnail, with the guardrail's vision
model. An unsafe image is deleted and never shown or charged. If the check itself fails (a
timeout or a bad answer) the image is dropped too: fail closed, per image. Nothing is released to
the user until it has passed this check.
"""

import base64
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.contracts.ports import VisionPort
from app.guardrails.retry import call_with_retry
from app.guardrails.schemas import Decision
from app.guardrails.vision_check import verdict_from_output

RETRY_DELAY_S = 0.5


class Safety(StrEnum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    FAILED = "failed"  # the check could not run. The image is dropped


@dataclass(frozen=True)
class SafetyResult:
    """What one check found, for the dev trace."""

    verdict: Safety
    started_at: float
    ended_at: float
    model: str
    p_unsafe: float | None = None
    neurons: float | None = None
    error: str | None = None  # the exception type when the check failed


async def check_image_detail(
    vision: VisionPort, thumb: bytes, prompt: str, *, timeout: float
) -> SafetyResult:
    """`thumb` is the 256 px JPEG. Never raises: any failure is `FAILED` (fail closed)."""
    url = "data:image/jpeg;base64," + base64.b64encode(thumb).decode("ascii")
    seen: dict[str, Any] = {}
    started = time.time()

    async def once(attempt_timeout: float) -> Decision:
        raw = await vision.classify(
            text=prompt or "(no text)", image_urls=[url], timeout=attempt_timeout
        )
        verdict = verdict_from_output(
            raw.text, raw.first_token_logprobs, model=vision.model, latency_ms=0
        )
        seen["p"] = verdict.scores.get("unsafe")
        if raw.usage:
            seen["neurons"] = seen.get("neurons", 0) + float(raw.usage.get("neurons") or 0)
        return verdict.decision

    error: str | None = None
    try:
        decision = await call_with_retry(
            once,
            timeout=timeout,
            retry_delay=RETRY_DELAY_S,
            deadline=time.monotonic() + 2 * timeout + RETRY_DELAY_S,
        )
        verdict_ = Safety.SAFE if decision is Decision.APPROVE else Safety.UNSAFE
    except Exception as exc:  # noqa: BLE001 - a failed check is never a pass
        verdict_, error = Safety.FAILED, type(exc).__name__
    return SafetyResult(
        verdict=verdict_,
        started_at=started,
        ended_at=time.time(),
        model=vision.model,
        p_unsafe=seen.get("p"),
        neurons=seen.get("neurons"),
        error=error,
    )


async def check_image(vision: VisionPort, thumb: bytes, prompt: str, *, timeout: float) -> Safety:
    """The verdict only. Never raises: any failure is `FAILED` (fail closed)."""
    return (await check_image_detail(vision, thumb, prompt, timeout=timeout)).verdict
