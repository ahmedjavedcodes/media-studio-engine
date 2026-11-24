"""Per-image safety check. Spec v1.3 section 4 [5].

Each image is checked as soon as it is made, on its 256 px thumbnail, with the guardrail's vision
model. An unsafe image is deleted and never shown or charged. If the check itself fails (a
timeout or a bad answer) the image is dropped too: fail closed, per image. Nothing is released to
the user until it has passed this check.
"""

import base64
import time
from enum import StrEnum

from app.contracts.ports import VisionPort
from app.guardrails.retry import call_with_retry
from app.guardrails.schemas import Decision
from app.guardrails.vision_check import verdict_from_output

RETRY_DELAY_S = 0.5


class Safety(StrEnum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    FAILED = "failed"  # the check could not run. The image is dropped


async def check_image(vision: VisionPort, thumb: bytes, prompt: str, *, timeout: float) -> Safety:
    """`thumb` is the 256 px JPEG. Never raises: any failure is `FAILED` (fail closed)."""
    url = "data:image/jpeg;base64," + base64.b64encode(thumb).decode("ascii")

    async def once(attempt_timeout: float) -> Decision:
        started = time.monotonic()
        raw = await vision.classify(
            text=prompt or "(no text)", image_urls=[url], timeout=attempt_timeout
        )
        verdict = verdict_from_output(
            raw.text,
            raw.first_token_logprobs,
            model=vision.model,
            latency_ms=int((time.monotonic() - started) * 1000),
        )
        return verdict.decision

    try:
        decision = await call_with_retry(
            once,
            timeout=timeout,
            retry_delay=RETRY_DELAY_S,
            deadline=time.monotonic() + 2 * timeout + RETRY_DELAY_S,
        )
    except Exception:  # noqa: BLE001 - a failed check is never a pass
        return Safety.FAILED
    return Safety.SAFE if decision is Decision.APPROVE else Safety.UNSAFE
