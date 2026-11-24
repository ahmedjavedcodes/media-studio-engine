"""GET /v1/models and /v1/plans: what the model picker and the plan dropdown show."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.agents.image_grand.generation.create.deps import CreateDeps
from app.api.deps import get_create_deps
from app.core.plans import PLAN_LIMITS, TIER_CREDITS, Plan, tier_allowed
from app.tools.sizes import ASPECT_RATIOS

router = APIRouter(prefix="/v1")


@router.get("/plans")
async def list_plans() -> dict[str, Any]:
    return {
        "plans": [
            {
                "id": p.value,
                "max_n": lim.max_n,
                "max_resolution": lim.max_resolution,
                "tier": lim.tier.value,
            }
            for p, lim in PLAN_LIMITS.items()
        ],
        "aspect_ratios": list(ASPECT_RATIOS),
        # Every image is made at 1K for now. 2K comes from the upscale tool later.
        "resolutions": [{"id": "1K", "available": True}, {"id": "2K", "available": False}],
    }


@router.get("/models")
async def list_models(
    deps: Annotated[CreateDeps, Depends(get_create_deps)], plan: Plan = Plan.BASIC
) -> dict[str, Any]:
    """Models a user can pick. `locked` is true when the plan is too low. Featured models first,
    then by tier. No prices or provider names."""
    cards = deps.registry.live(deps.enabled_providers)
    models = [
        {
            "id": c.id,
            "name": c.name,
            "family": c.family,
            "tier": c.tier.value,
            "tag": c.tag,
            "featured": c.featured,
            "credits_per_image": TIER_CREDITS[c.tier],
            "locked": not tier_allowed(plan, c.tier),
            "aspect_ratios": c.supported_aspect_ratios,
        }
        for c in cards
    ]
    models.sort(key=lambda m: (not m["featured"], list(TIER_CREDITS).index(m["tier"]), m["name"]))  # type: ignore[arg-type]
    return {"models": models, "plan": plan.value}
