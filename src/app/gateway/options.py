"""Build the per-call ImageOptions for a model. Code only: the LLM never writes these.

This is where the model's own rules are enforced: the quality guardrail, the reference limit,
transparent background support and brand colors. The same checks run again inside the adapter.
"""

import re
from dataclasses import dataclass

from app.contracts.ports import ImageOptions
from app.core.errors import BadInput
from app.core.plans import Plan
from app.gateway.registry import ModelCard

HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
MAX_BRAND_COLORS = 5


@dataclass(frozen=True)
class RequestExtras:
    """What the user asked for, beyond the prompt. All validated before it gets here."""

    references: tuple[str, ...] = ()
    brand_colors: tuple[str, ...] = ()
    transparent_bg: bool = False
    complexity: int | None = None
    plan: Plan | None = None  # picks the quality of a model that has a quality knob
    manual: bool = False  # the user picked this model by hand (not Auto)


def check_references(card: ModelCard, n_refs: int) -> None:
    """Pre-flight: a model takes a fixed number of reference images. Raises BadInput (400)."""
    if n_refs > card.max_refs:
        allowed = (
            "does not take reference images"
            if card.max_refs == 0
            else f"takes at most {card.max_refs}"
        )
        raise BadInput(
            f"{card.name} {allowed}. You attached {n_refs}. "
            "Remove some images or pick another model."
        )


def quality_for(card: ModelCard, plan: Plan | None, *, manual: bool) -> str | None:
    """Quality for one call. Models without `quality_by_plan` keep their own quality. For GPT Image:
    low on Basic, medium on Pro and up, and `high` only when a Max user picks the model by hand
    (an Auto choice never goes above medium, so Auto cannot run up the bill)."""
    if plan is None or not card.quality_by_plan:
        return card.quality
    quality = card.quality_by_plan.get(plan.value, card.quality)
    if quality == "high" and not (plan == Plan.MAX and manual):
        quality = "medium"
    return quality


def extra_params(card: ModelCard) -> dict[str, float | int | str]:
    """Optional parameters for this model: only those the registry says it takes AND has a value."""
    return {k: v for k, v in card.params.items() if k in card.supported_params}


def options_for(card: ModelCard, extras: RequestExtras | None = None) -> ImageOptions:
    """The options for one call to `card`. Raises BadInput when the request breaks a model rule."""
    extras = extras or RequestExtras()
    check_references(card, len(extras.references))
    quality = quality_for(card, extras.plan, manual=extras.manual)
    if quality is not None and quality in card.forbidden_qualities:
        raise BadInput(f"{card.name} cannot use {quality} quality.")  # cost guardrail
    if extras.transparent_bg and "transparent_bg" not in card.capabilities:
        raise BadInput(f"{card.name} cannot make a transparent background.")
    colors = tuple(c.upper() for c in extras.brand_colors)
    if any(not HEX_COLOR.match(c) for c in colors) or len(colors) > MAX_BRAND_COLORS:
        raise BadInput(f"Brand colors must be up to {MAX_BRAND_COLORS} hex colors like #1A2B3C.")
    use_colors = colors if "rgb_colors" in card.capabilities else ()  # a hint: other models skip it
    extra = extra_params(card)
    guidance, steps, style = extra.get("guidance"), extra.get("steps"), extra.get("style")
    return ImageOptions(
        references=extras.references,
        quality=quality,
        resolution="1K" if "1K" in card.resolutions else None,  # always an explicit size
        output_format=card.output_format,
        background="transparent" if extras.transparent_bg else None,
        brand_colors=use_colors,
        complexity=extras.complexity if "complexity_dial" in card.capabilities else None,
        endpoint=card.endpoint,
        guidance=float(guidance) if guidance is not None else None,
        steps=int(steps) if steps is not None else None,
        style=str(style) if style is not None else None,
    )
