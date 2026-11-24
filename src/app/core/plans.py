"""Plans, model tiers and the limits that go with them. Spec section 3 (Plan limits).

A plan can use its own tier and every tier below it. The plan is chosen by hand in the UI during
the build phase. It will come from the user's account later.
"""

from enum import StrEnum
from typing import Final

from pydantic import BaseModel


class Plan(StrEnum):
    BASIC = "basic"
    PRO = "pro"
    ULTRA = "ultra"
    MAX = "max"


class Tier(StrEnum):
    BASIC = "basic"
    PRO = "pro"
    ULTRA = "ultra"
    MAX = "max"


TIER_ORDER: Final[tuple[Tier, ...]] = (Tier.BASIC, Tier.PRO, Tier.ULTRA, Tier.MAX)

# Tier upper bounds in USD per image at 1K (spec section 6). Max is everything above Ultra.
_TIER_BOUNDS: Final[tuple[tuple[float, Tier], ...]] = (
    (0.015, Tier.BASIC),
    (0.04, Tier.PRO),
    (0.09, Tier.ULTRA),
)

# Credits charged per delivered image, by model tier.
TIER_CREDITS: Final[dict[Tier, int]] = {Tier.BASIC: 1, Tier.PRO: 3, Tier.ULTRA: 6, Tier.MAX: 12}


class PlanLimits(BaseModel):
    plan: Plan
    max_n: int
    max_resolution: str  # what the plan is allowed to ask for
    tier: Tier  # highest model tier the plan may use


PLAN_LIMITS: Final[dict[Plan, PlanLimits]] = {
    Plan.BASIC: PlanLimits(plan=Plan.BASIC, max_n=2, max_resolution="1K", tier=Tier.BASIC),
    Plan.PRO: PlanLimits(plan=Plan.PRO, max_n=4, max_resolution="2K", tier=Tier.PRO),
    Plan.ULTRA: PlanLimits(plan=Plan.ULTRA, max_n=4, max_resolution="2K", tier=Tier.ULTRA),
    Plan.MAX: PlanLimits(plan=Plan.MAX, max_n=4, max_resolution="2K", tier=Tier.MAX),
}

RESOLUTIONS: Final[tuple[str, ...]] = ("1K", "2K")


def tier_for_cost(cost_per_image: float) -> Tier:
    """Auto tier from the normalized cost per image at 1K."""
    for bound, tier in _TIER_BOUNDS:
        if cost_per_image < bound:
            return tier
    return Tier.MAX


def tier_allowed(plan: Plan, tier: Tier) -> bool:
    return TIER_ORDER.index(tier) <= TIER_ORDER.index(PLAN_LIMITS[plan].tier)


def credits_for(tier: Tier, n: int) -> int:
    return TIER_CREDITS[tier] * n
