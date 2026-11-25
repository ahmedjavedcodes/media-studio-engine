"""[1] Code checks. No LLM, no cost. Spec section 4 [1].

Plan limits, model and aspect ratio support. A failure here is a friendly error and nothing is
charged. Credits are held later, in the router step, once the real model (and so its tier) is known.
"""

from typing import Any

from app.agents.image_grand.generation.create.deps import CreateDeps
from app.agents.image_grand.generation.create.state import CreateState
from app.core.errors import BadInput, PlanLimit
from app.core.plans import PLAN_LIMITS, RESOLUTIONS, Plan, tier_allowed
from app.gateway.options import RequestExtras, options_for
from app.tools.sizes import ASPECT_RATIOS


async def run_code_checks(deps: CreateDeps, state: CreateState) -> dict[str, Any]:
    plan = Plan(state["plan"])
    limits = PLAN_LIMITS[plan]
    cfg = state["settings"]

    if cfg["n"] > limits.max_n:  # never lowered silently
        raise PlanLimit(
            f"Your {plan.value.title()} plan creates up to {limits.max_n} images at a time. "
            "Choose fewer, or upgrade your plan."
        )
    if cfg["resolution"] not in RESOLUTIONS:
        raise BadInput("Resolution must be 1K or 2K.")
    if cfg["resolution"] == "2K":  # never lowered silently. 2K comes from the upscale tool later
        if RESOLUTIONS.index("2K") > RESOLUTIONS.index(limits.max_resolution):
            raise PlanLimit("2K is not part of your plan. Upgrade to use it.")
        raise PlanLimit("2K is coming soon. Choose 1K for now.")
    if cfg["aspect_ratio"] not in ASPECT_RATIOS:
        raise BadInput("That aspect ratio is not supported.")

    if cfg["model"] != "auto":  # a manual pick is strict: never swapped for another model
        card = deps.registry.get(cfg["model"])
        usable = card is not None and card in deps.registry.live(deps.enabled_providers)
        if card is None or not usable:
            raise PlanLimit("That model is not available right now.")
        if not tier_allowed(plan, card.tier):
            raise PlanLimit(f"{card.name} needs a higher plan. Upgrade to use it.")
        if cfg["aspect_ratio"] not in card.supported_aspect_ratios:
            raise BadInput(f"{card.name} does not support the {cfg['aspect_ratio']} aspect ratio.")
        options_for(  # reference limit, quality guardrail, transparent background, colors
            card,
            RequestExtras(
                references=("x",) * state.get("n_refs", 0),
                brand_colors=tuple(cfg.get("brand_colors") or ()),
                transparent_bg=bool(cfg.get("transparent_bg")),
            ),
        )
    return {}
