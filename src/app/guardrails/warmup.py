"""Warm the provider connections so the first user request is not cold. Never raises."""

import asyncio
import base64
import io

import structlog
from PIL import Image

from app.guardrails.agent import GuardDeps
from app.guardrails.text_check import load_questions

log = structlog.get_logger("guardrail")

WARMUP_TIMEOUT_S = 10.0


def _tiny_image_url() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


async def warm_up(deps: GuardDeps) -> dict[str, str]:
    """One small call to Jev and one to the vision model. Returns "ok" or the error per source."""
    csam = load_questions().questions["csam"]  # a single question keeps the call cheap
    state = {
        "context": "warm-up",
        "prompt_raw": "warm-up",
        "prompt_clean": "warm-up",
        "has_image": False,
    }

    async def jev() -> None:
        await deps.jev.ask(state=state, questions={"csam": csam}, timeout=WARMUP_TIMEOUT_S)

    async def vision() -> None:
        await deps.vision.classify(
            text="warm-up", image_urls=[_tiny_image_url()], timeout=WARMUP_TIMEOUT_S
        )

    results = await asyncio.gather(jev(), vision(), return_exceptions=True)
    out = {
        name: "ok" if not isinstance(r, BaseException) else type(r).__name__
        for name, r in zip(("jev", "vision"), results, strict=True)
    }
    log.info("guardrail_warmup", **out)
    return out
