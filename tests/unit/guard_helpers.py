"""Fakes and builders shared by the guardrail tests."""

import asyncio
import io
import math
from datetime import UTC, datetime
from typing import Any

from PIL import Image

from app.contracts.ports import VisionRaw
from app.core.config import Settings
from app.core.errors import ProviderCallFailed
from app.guardrails.agent import GuardDeps, Timing
from app.guardrails.text_check import QUESTION_CATEGORY
from app.local.fake_strikes import FakeStrikeLedger

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def jev_answers(**over: float) -> dict[str, Any]:
    base = {k: 0.01 for k in QUESTION_CATEGORY}
    base.update(over)
    return {k: {"type": "noul", "noul": v} for k, v in base.items()}


def png(size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "green").save(buf, format="PNG")
    return buf.getvalue()


def lp(p: float) -> list[tuple[str, float]]:
    return [("unsafe", math.log(p)), ("safe", math.log(1 - p))]


class FakeJev:
    model = "fake-jev"

    def __init__(
        self,
        answers: dict[str, Any] | None = None,
        *,
        delay: float = 0.0,
        errors: list[Exception] | None = None,
    ) -> None:
        self.answers = answers if answers is not None else jev_answers()
        self.delay = delay
        self.errors = list(errors or [])
        self.calls = 0
        self.states: list[dict[str, Any]] = []
        self.cancelled = False

    async def ask(
        self, *, state: dict[str, Any], questions: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        self.calls += 1
        self.states.append(state)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.errors:
            raise self.errors.pop(0)
        return self.answers


class FakeVision:
    model = "fake-vision"

    def __init__(
        self,
        text: str = "safe",
        p: float | None = 0.02,
        *,
        delay: float = 0.0,
        errors: list[Exception] | None = None,
    ) -> None:
        self.text = text
        self.p = p
        self.delay = delay
        self.errors = list(errors or [])
        self.calls: list[dict[str, Any]] = []
        self.cancelled = False

    async def classify(self, *, text: str, image_urls: list[str], timeout: float) -> VisionRaw:
        self.calls.append({"text": text, "image_urls": image_urls})
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.errors:
            raise self.errors.pop(0)
        return VisionRaw(self.text, lp(self.p) if self.p is not None else None)


def make_deps(
    jev: FakeJev | None = None,
    vision: FakeVision | None = None,
    ledger: FakeStrikeLedger | None = None,
    timing: Timing | None = None,
) -> GuardDeps:
    return GuardDeps(
        jev=jev or FakeJev(),
        vision=vision or FakeVision(),
        ledger=ledger or FakeStrikeLedger(now=lambda: T0),
        settings=Settings(_env_file=None),
        timing=timing or Timing(),
        clock=lambda: T0,
    )


def boom(retryable: bool = True) -> ProviderCallFailed:
    return ProviderCallFailed("boom", retryable=retryable)
