"""Model router (auto only). Spec section 4 [2]. Plain code, no LLM.

Runs BEFORE the enhancer and reads the raw prompt. It classifies the prompt into a category and
picks the best live model the plan allows. The same interface lets an LLM replace it later.
"""

import re
from dataclasses import dataclass
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
    n_refs: int = 0,
    transparent: bool = False,
) -> ModelCard:
    """The best live model the plan allows that supports the aspect ratio and the references."""
    pool = _usable_pool(registry, plan, aspect_ratio, enabled_providers, n_refs, transparent)
    if not pool:
        raise PlanLimit("No model is available for this plan and aspect ratio.")
    prefs = _PREFS.get(category, _PREFS["general"])
    plan_tier = PLAN_LIMITS[plan].prefer_tier  # Auto leans toward this tier. All allowed count

    def rank(card: ModelCard) -> tuple[int, int, int, int, float]:
        # The preferred tier first, then lower tiers (the highest of those first), then higher
        # tiers (the lowest of those first). A higher tier is allowed but never the first choice.
        gap = TIER_ORDER.index(plan_tier) - TIER_ORDER.index(card.tier)
        tier_gap = gap if gap >= 0 else 100 - gap
        suited = 0 if category in card.best_for else 1
        family = prefs.index(card.family) if card.family in prefs else len(prefs)
        return (tier_gap, suited, family, 0 if card.featured else 1, card.cost_per_image)

    return min(pool, key=rank)


def _usable_pool(
    registry: Registry,
    plan: Plan,
    aspect_ratio: str,
    enabled_providers: list[str],
    n_refs: int,
    transparent: bool,
) -> list[ModelCard]:
    return [
        c
        for c in registry.allowed(plan, enabled_providers)
        if c.auto  # Auto only chooses curated models. The rest are picked by hand
        and aspect_ratio in c.supported_aspect_ratios
        and c.max_refs >= n_refs
        and (not transparent or "transparent_bg" in c.capabilities)
    ]


# ---- auto routing rules (the Image Grand Agent) ----------------------------------------------
# Deterministic code, no LLM. Each rule is a list of model ids, best first. The first one that is
# live, allowed by the plan, on an enabled provider, and takes the aspect ratio and the number of
# reference images wins. If none qualifies, the category picker above chooses.

_LOGO = re.compile(
    r"\b(logo|logos|icon|icons|emblem|monogram|wordmark|vector|svg|scalable|brand ?mark)\b", re.I
)
_POSTER = re.compile(
    r"\b(poster|flyer|leaflet|brochure|packaging|banner|billboard|label|social (media )?ad|"
    r"ad layout|advert layout)\b",
    re.I,
)
_DENSE_TEXT = re.compile(
    r"(\"[^\"]{20,}\")|(\u201c[^\u201d]{20,}\u201d)|"
    r"\b(text overlay|headline|paragraph|caption|menu|infographic|slide)\b",
    re.I,
)
# Every candidate is in the curated list (picker + auto). A model outside it is never chosen here.
RULE_VECTOR = ["recraft-v4-1-vector"]
RULE_BRAND = ["recraft-v4-1"]
RULE_BRAND_WITH_STYLE = ["recraft-v4-1"]  # Recraft Styles is not on the list
RULE_MERGE = ["seedream-5-flash", "nano-banana-2-1"]  # both take up to 14 references
RULE_PRECISION = ["gpt-image-2-5-sunburst", "nano-banana-2-1"]
RULE_DRAFT_TEXT = ["flux-2-klein-4b", "recraft-v4-1-flash", "gpt-image-2-5-flare"]  # cheapest first
RULE_DRAFT_REFS = ["gpt-image-2-5-flare", "seedream-5-flash"]  # the cheap models that take images
RULE_DEFAULT = ["nano-banana-2-1", "seedream-5-pro"]


@dataclass(frozen=True)
class RouteRequest:
    prompt: str
    plan: Plan
    aspect_ratio: str
    enabled_providers: list[str]
    n_refs: int = 0
    brand_colors: tuple[str, ...] = ()
    draft: bool = False
    has_style_ref: bool = False
    transparent: bool = False


@dataclass(frozen=True)
class Routed:
    card: ModelCard
    rule: str  # which rule chose it, for the trace


def _rule_for(req: RouteRequest) -> tuple[str, list[str]]:
    """The first rule that applies, and its candidates. A draft is the user's own choice for
    speed and low cost, so it comes first."""
    if req.draft:
        return "draft", RULE_DRAFT_REFS if req.n_refs else RULE_DRAFT_TEXT
    if _LOGO.search(req.prompt):
        return "vector", RULE_VECTOR
    if req.brand_colors or _POSTER.search(req.prompt):
        return "brand", RULE_BRAND_WITH_STYLE if req.has_style_ref else RULE_BRAND
    if req.n_refs >= 2:
        return "merge", RULE_MERGE
    if req.n_refs == 1 or _DENSE_TEXT.search(req.prompt):
        return "precision", RULE_PRECISION
    return "default", RULE_DEFAULT


def route_auto(registry: Registry, req: RouteRequest) -> Routed:
    """Pick a model for `model="auto"`. Raises PlanLimit when nothing can run."""
    pool = _usable_pool(
        registry, req.plan, req.aspect_ratio, req.enabled_providers, req.n_refs, req.transparent
    )
    usable = {c.id: c for c in pool}
    rule, candidates = _rule_for(req)
    for model_id in candidates:
        if model_id in usable:
            return Routed(usable[model_id], rule)
    card = pick_model(
        registry,
        plan=req.plan,
        category=classify_prompt(req.prompt),
        aspect_ratio=req.aspect_ratio,
        enabled_providers=req.enabled_providers,
        n_refs=req.n_refs,
        transparent=req.transparent,
    )
    return Routed(card, f"{rule}->category")
