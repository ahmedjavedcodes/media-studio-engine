"""Model router (auto only). Spec section 4 [2]. Plain code, no LLM.

Runs BEFORE the enhancer and reads the raw prompt. It classifies the prompt into a category and
picks the best live model the plan allows. The same interface lets an LLM replace it later.
"""

import re
from typing import Final, Literal

from app.core.errors import PlanLimit
from app.core.plans import PLAN_LIMITS, TIER_ORDER, Plan
from app.gateway.registry import ModelCard, Registry

Category = Literal["product", "photo", "people", "text_in_image", "design", "general"]

# Category -> families to prefer, best first.
_PREFS: Final[dict[str, tuple[str, ...]]] = {
    "product": ("flux", "seedream", "leonardo"),
    "photo": ("flux", "seedream", "leonardo"),
    "people": ("gemini", "leonardo", "flux"),
    "text_in_image": ("qwen", "ideogram", "gpt_image", "leonardo"),
    "design": ("recraft", "leonardo"),
    "general": ("flux", "seedream", "leonardo"),
}

# Checked in this order. English plus a few Roman Urdu words. Anything unread falls to `general`.
_RULES: Final[tuple[tuple[Category, tuple[str, ...]], ...]] = (
    (
        "text_in_image",
        (
            "text",
            "typography",
            "lettering",
            "logo",
            "poster",
            "signboard",
            "headline",
            "slogan",
            "caption",
            "banner",
            "likha",
            "likhna",
        ),
    ),
    (
        "design",
        ("brand", "flat", "vector", "icon", "infographic", "palette", "pattern", "mockup"),
    ),
    (
        "people",
        (
            "person",
            "people",
            "man",
            "woman",
            "men",
            "women",
            "girl",
            "boy",
            "child",
            "kid",
            "portrait",
            "selfie",
            "family",
            "face",
            "crowd",
            "character",
            "couple",
            "ladki",
            "ladka",
        ),
    ),
    (
        "product",
        (
            "product",
            "packaging",
            "bottle",
            "ad",
            "advert",
            "advertisement",
            "commercial",
            "shoe",
            "perfume",
            "cup",
            "jar",
        ),
    ),
    (
        "photo",
        (
            "photo",
            "photograph",
            "landscape",
            "mountain",
            "beach",
            "city",
            "street",
            "sunset",
            "forest",
            "food",
            "dish",
            "interior",
            "building",
            "architecture",
            "animal",
            "car",
            "sky",
            "ocean",
        ),
    ),
)


def classify_prompt(prompt: str) -> Category:
    """Category of the raw prompt, by keyword rules. A quoted phrase means text in the image."""
    if '"' in prompt or "“" in prompt:
        return "text_in_image"
    words = set(re.findall(r"[a-z]+", prompt.lower()))
    for category, keywords in _RULES:
        if words & set(keywords):
            return category
    return "general"


def pick_model(
    registry: Registry,
    *,
    plan: Plan,
    category: str,
    aspect_ratio: str,
    enabled_providers: list[str],
) -> ModelCard:
    """The best live model the plan allows that supports the aspect ratio."""
    pool = [
        c
        for c in registry.allowed(plan, enabled_providers)
        if aspect_ratio in c.supported_aspect_ratios
    ]
    if not pool:
        raise PlanLimit("No model is available for this plan and aspect ratio.")
    prefs = _PREFS.get(category, _PREFS["general"])
    plan_tier = PLAN_LIMITS[plan].tier

    def rank(card: ModelCard) -> tuple[int, int, int, int, float]:
        # The plan's own tier first, then lower tiers (the highest of those first).
        tier_gap = TIER_ORDER.index(plan_tier) - TIER_ORDER.index(card.tier)
        suited = 0 if category in card.best_for else 1
        family = prefs.index(card.family) if card.family in prefs else len(prefs)
        return (tier_gap, suited, family, 0 if card.featured else 1, card.cost_per_image)

    return min(pool, key=rank)
