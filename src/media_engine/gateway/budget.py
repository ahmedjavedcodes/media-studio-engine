"""Spend accounting: neurons for Cloudflare's shared 10,000-neurons/day free quota (resets
00:00 UTC), and normalised USD estimates for OpenRouter image calls.

`estimate_image_usd` IS ENFORCED (since M4): the gateway (router.image) checks it before every
paid OpenRouter call, against the per-call cap the caller passes (`max_cost_usd`). An unknown
price is "cannot afford". It was checked against a real `usage.cost`: gpt-image-2 `low`
estimates $0.006 against $0.00596 billed (tests/fixtures/openrouter/gpt-image-2.images.json).
It is an ESTIMATE: the other protections stay (ALLOW_PAID off by default, the plan guard).
tests/unit/gateway/test_budget_enforced.py pins where it is called from."""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Literal

from media_engine.contracts import Plan
from media_engine.gateway.image_registry import ImageModelSpec, image_registry
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


TaskType = Literal["create", "edit", "merge"]

# Wired into the gateway's pre-call check (router.image) and checked against a real usage.cost.
USD_ESTIMATE_ENFORCED = True

# Multiplier for an edit when no edit price was measured. 1.5 is the worst measured markup
# (FLUX.2 Pro: $0.030 create, $0.045 edit); Riverflow measured ~1.18, gpt-image-2 and Gemini 3
# Pro <= 1.07. Deliberately pessimistic: this number feeds a spend cap.
EDIT_MARKUP_DEFAULT = 1.5


def _listed_output_price(spec: ImageModelSpec) -> float | None:
    """The catalogue's listed per-image (or per-megapixel) output price. Listed is NOT billed:
    used only when nothing was measured. Token prices cannot be turned into an image price."""
    for entry in spec.listed_pricing:
        if entry.get("billable") == "output_image" and entry.get("unit") in {"image", "megapixel"}:
            cost = entry.get("cost_usd")
            if isinstance(cost, int | float) and not entry.get("variant"):
                return float(cost)
    return None


def _listed_input_price(spec: ImageModelSpec, unit: str) -> float:
    for entry in spec.listed_pricing:
        if entry.get("billable") == "input_image" and entry.get("unit") == unit:
            cost = entry.get("cost_usd")
            if isinstance(cost, int | float):
                return float(cost)
    return 0.0


def estimate_image_usd(
    spec: ImageModelSpec, request: ImageRequest, task_type: TaskType = "create"
) -> float | None:
    """ENFORCED by the gateway's pre-call check (see the module docstring).

    Estimated USD for one OpenRouter image call (all `request.n` images), normalised across
    billing units. None = no price is known: the caller must treat that as "cannot afford".

    Base price per image, in order: the measured price for the requested quality; the measured
    edit price for an edit; the measured price; the catalogue's listed price.
      - image:     flat per image
      - megapixel: base x output megapixels (rounded up, at least 1; base was measured at 1 MP)
      - token:     flat (token use depends on size and quality; base was measured at 1:1)
    An edit/merge with no measured edit price is base x EDIT_MARKUP_DEFAULT. Input references
    add the catalogue's listed per-image or per-megapixel input price (refs counted as 1 MP).
    """
    quality = request.quality or spec.default_quality
    editing = task_type != "create" or bool(request.input_images)
    base: float | None = None
    if quality is not None and quality in spec.price_by_quality:
        base = spec.price_by_quality[quality]
    elif quality is not None and spec.price_by_quality:
        base = max(spec.price_by_quality.values())  # unmeasured quality: assume the dearest
    if base is None and editing and spec.price_edit_usd is not None:
        base, editing = spec.price_edit_usd, False  # already the edit price: no markup
    if base is None:
        base = spec.price_measured_usd
    if base is None:
        base = _listed_output_price(spec)
    if base is None:
        return None

    per_image = base
    if spec.billing_unit == "megapixel":
        per_image *= max(1, math.ceil(request.width * request.height / MEGAPIXEL))
    if editing:
        per_image *= EDIT_MARKUP_DEFAULT

    refs = len(request.input_images)
    inputs = refs * (_listed_input_price(spec, "image") + _listed_input_price(spec, "megapixel"))
    return round(per_image * max(1, request.n) + inputs, 6)


def dearest_image_usd(plan: Plan) -> float | None:
    """The dearest image a plan's registry models can bill at default settings (None: none is
    priced). The entry point hands this to the supervisor for its step guard."""
    prices = [
        usd
        for spec in image_registry().for_plan(plan)
        if (usd := estimate_image_usd(spec, ImageRequest(prompt="x"))) is not None
    ]
    return max(prices) if prices else None


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
