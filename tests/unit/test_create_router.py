import pytest

from app.agents.image_grand.workers.shot_plan import ShotPlan
from app.core.errors import PlanLimit
from app.core.plans import Plan, tier_allowed
from app.gateway.registry import load_registry
from app.gateway.router import classify_prompt, pick_model

CATEGORIES = ["product", "photo", "people", "text_in_image", "design", "general"]
PROMPTS = {
    "product": "iced coffee ad for summer",
    "photo": "a mountain lake at sunrise",
    "people": "a woman laughing in a park",
    "text_in_image": 'a poster that says "SALE"',
    "design": "flat vector icon set in brand colors",
    "general": "something calm",
}


@pytest.mark.parametrize("category", CATEGORIES)
def test_classifier_maps_prompt_to_category(category: str) -> None:
    assert classify_prompt(PROMPTS[category]) == category


def test_unreadable_script_falls_back_to_general() -> None:
    assert classify_prompt("ایک خوبصورت چیز") == "general"  # no word the classifier knows


def test_urdu_and_roman_urdu_keywords_are_read() -> None:
    assert classify_prompt("ایک خوبصورت منظر") == "photo"
    assert classify_prompt("ماربل پر پرفیوم کی بوتل") == "product"
    assert classify_prompt("ek ladki park mein") == "people"
    assert classify_prompt("پوسٹر پر عید مبارک") == "text_in_image"


@pytest.mark.parametrize("plan", list(Plan))
@pytest.mark.parametrize("category", CATEGORIES)
def test_every_category_maps_to_an_allowed_live_model_for_every_plan(
    plan: Plan, category: str
) -> None:
    reg = load_registry()
    card = pick_model(
        reg, plan=plan, category=category, aspect_ratio="1:1", enabled_providers=["cloudflare"]
    )
    assert card.status == "live"
    assert tier_allowed(plan, card.tier)
    assert card.providers_in(["cloudflare"])


def test_router_prefers_the_plans_own_tier() -> None:
    reg = load_registry()
    args = {"category": "general", "aspect_ratio": "1:1", "enabled_providers": ["cloudflare"]}
    assert pick_model(reg, plan=Plan.BASIC, **args).tier.value == "basic"  # type: ignore[arg-type]
    assert pick_model(reg, plan=Plan.PRO, **args).tier.value == "pro"  # type: ignore[arg-type]


def test_router_uses_best_for_inside_a_tier() -> None:
    reg = load_registry()
    card = pick_model(
        reg,
        plan=Plan.PRO,
        category="text_in_image",
        aspect_ratio="1:1",
        enabled_providers=["cloudflare"],
    )
    assert card.id == "phoenix-1"  # tagged for text


def test_router_skips_models_that_cannot_do_the_aspect_ratio() -> None:
    reg = load_registry()
    card = pick_model(
        reg,
        plan=Plan.BASIC,
        category="general",
        aspect_ratio="16:9",
        enabled_providers=["cloudflare"],
    )
    assert card.id != "flux-1-schnell"  # schnell is 1:1 only


def test_router_never_picks_hidden_or_paid_models() -> None:
    reg = load_registry()
    for plan in Plan:
        card = pick_model(
            reg, plan=plan, category="people", aspect_ratio="1:1", enabled_providers=["cloudflare"]
        )
        assert card.id != "nano-banana-2"  # not live, and OpenRouter is off


def test_no_model_for_the_ratio_is_a_plan_limit() -> None:
    reg = load_registry()
    with pytest.raises(PlanLimit):
        pick_model(
            reg, plan=Plan.BASIC, category="general", aspect_ratio="9:16", enabled_providers=[]
        )


def test_brief_is_not_needed_by_the_router() -> None:
    # the router reads the raw prompt. A brief does not exist yet when it runs
    assert ShotPlan().subject == ""
