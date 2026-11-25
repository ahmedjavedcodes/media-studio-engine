"""GET /v1/models and /v1/plans: what the model picker and the plan dropdown show."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.agents.image_grand.generation.create.deps import CreateDeps
from app.api.deps import get_create_deps
from app.core.plans import PLAN_LIMITS, TIER_CREDITS, Plan, tier_allowed
from app.gateway.pricing import credits_per_image
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
    """Models for the picker. `locked` is true when the plan is too low, `available` is false for
    "coming soon" models. No prices or provider names."""
    usable = {c.id for c in deps.registry.live(deps.enabled_providers)}
    models = [
        {
            "id": c.id,
            "name": c.name,
            "family": c.family,
            "family_label": c.family_label,
            "tier": c.tier.value,
            # False = shown as "coming soon". Not run yet, or its provider is off. No provider
            # name is sent: the pipeline stays hidden.
            "available": c.id in usable,
            "tag": c.tag,
            "featured": c.featured,
            "credits_per_image": credits_per_image(c),
            "ref_surcharge": c.ref_surcharge,  # extra credits for each reference after the first
            "resolution_multipliers": c.resolution_multipliers,
            "max_refs": c.max_refs,
            "capabilities": c.capabilities,
            "locked": not tier_allowed(plan, c.tier),
            "aspect_ratios": c.supported_aspect_ratios,
        }
        for c in deps.registry.listed()
    ]
    # Usable models come first (featured, then by tier). "Coming soon" models follow, by tier.
    models.sort(
        key=lambda m: (
            not m["available"],
            not m["featured"],
            list(TIER_CREDITS).index(m["tier"]),  # type: ignore[arg-type]
            m["name"],
        )
    )
    return {"models": models, "plan": plan.value}
