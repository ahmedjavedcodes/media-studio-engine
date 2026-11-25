"""Credits for one run. A user pays credits, not dollars. The hold and the charge use this."""

from app.core.plans import TIER_CREDITS
from app.gateway.registry import ModelCard


def credits_per_image(card: ModelCard, *, resolution: str = "1K", n_refs: int = 0) -> int:
    """Credits for one image: the model's own price (or its tier's), plus a surcharge for each
    reference after the first, times the resolution multiplier (for example 4K on Nano Banana Pro).
    Rounded up, so a run is never under-charged."""
    base = card.credits if card.credits is not None else TIER_CREDITS[card.tier]
    extra_refs = max(0, n_refs - 1) * card.ref_surcharge
    multiplier = card.resolution_multipliers.get(resolution, 1.0)
    return int(-(-(base + extra_refs) * multiplier // 1))  # ceil


def credits_for_run(card: ModelCard, n: int, *, resolution: str = "1K", n_refs: int = 0) -> int:
    return credits_per_image(card, resolution=resolution, n_refs=n_refs) * n
