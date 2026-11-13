"""Neuron accounting for Cloudflare's shared 10,000-neurons/day free quota (resets 00:00 UTC)."""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from datetime import UTC, date, datetime

from media_engine.gateway.registry import ModelCard
from media_engine.gateway.types import ImageRequest

DAILY_FREE_NEURONS = 10_000
TILE = 512
MEGAPIXEL = 1024 * 1024


def _image_size(data: bytes) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            return img.size
    except (OSError, ValueError):  # unreadable reference: assume a full 1 MP input
        return 1024, 1024


def estimate_image_neurons(card: ModelCard, request: ImageRequest) -> float | None:
    """Estimated neurons for one image call, from the registry price. None if no price known."""
    price = card.neuron_pricing
    if price is None:
        return None
    out_tiles = math.ceil(request.width / TILE) * math.ceil(request.height / TILE)
    out_mp = max(1, math.ceil(request.width * request.height / MEGAPIXEL))
    steps = request.steps if request.steps is not None else (price.default_steps or 0)

    total = out_tiles * price.per_out_tile + steps * price.per_step
    if price.per_first_mp:
        total += price.per_first_mp + (out_mp - 1) * price.per_extra_mp
    for data in request.input_images:
        w, h = _image_size(data)
        total += math.ceil(w / TILE) * math.ceil(h / TILE) * price.per_in_tile
        total += max(1, math.ceil(w * h / MEGAPIXEL)) * price.per_in_mp
    return round(total, 2)


class DailyNeuronMeter:
    """Running Cloudflare neuron total for the current UTC day. Actual values win over estimates."""

    def __init__(self, now: Callable[[], datetime] | None = None) -> None:
        self._now = now or (lambda: datetime.now(UTC))
        self._day: date = self._now().astimezone(UTC).date()
        self.total = 0.0
        self.actual = 0.0
        self.estimated = 0.0

    def _roll(self) -> None:
        today = self._now().astimezone(UTC).date()
        if today != self._day:
            self._day = today
            self.total = self.actual = self.estimated = 0.0

    def add(self, *, actual: float | None, estimate: float | None) -> float:
        """Record one Cloudflare call; returns the new daily total."""
        self._roll()
        if actual is not None:
            self.actual += actual
            self.total += actual
        elif estimate is not None:
            self.estimated += estimate
            self.total += estimate
        return self.total

    @property
    def total_today(self) -> float:
        self._roll()
        return self.total

    @property
    def remaining(self) -> float:
        return max(0.0, DAILY_FREE_NEURONS - self.total_today)


class DailyCostMeter:
    """Running USD total (providers that report cost, i.e. OpenRouter) for the current UTC day."""

    def __init__(self, now: Callable[[], datetime] | None = None) -> None:
        self._now = now or (lambda: datetime.now(UTC))
        self._day: date = self._now().astimezone(UTC).date()
        self.total_usd = 0.0

    def _roll(self) -> None:
        today = self._now().astimezone(UTC).date()
        if today != self._day:
            self._day = today
            self.total_usd = 0.0

    def add(self, cost_usd: float | None) -> float:
        self._roll()
        if cost_usd is not None:
            self.total_usd += cost_usd
        return self.total_usd

    @property
    def total_today(self) -> float:
        self._roll()
        return self.total_usd
