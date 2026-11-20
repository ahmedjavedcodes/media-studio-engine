"""The plan catalogue: what the prompt bar may offer a user on a given plan.

Built from the image registry (gateway/models.yaml merged with the OpenRouter catalogue cache)
and the create config (harness.yaml, presets). Plain data, no model call, no network.

- Models: every model that is `available` (an endpoint serves it) and `offered` (deliberately
  in the catalogue). Plans are cumulative: a plan unlocks its own tier and every tier below.
  Higher-tier models are listed `locked`, with the plan that unlocks them, so the UI can show
  an upgrade hint. A model with prices per quality (GPT Image 2 low / high) is one option per
  priced quality, each with the lowest plan its quality needs. An option the card marks
  `batch_only_for` (GPT Image 2 high, ~123 s per image) is flagged `batch_only`: the picker
  must keep it off the interactive path.
- `default_model` is "auto": with no pick, the system decides.

UI locks are UX only. The GATEWAY enforces the plan on every call (ImageModelSpec.plan_block),
whatever the client sends. TODO(backend): serve this over HTTP; the dev shim
(scripts/dev_server.py, GET /catalog) is the only server here.

Reasoning stacks (R1-R4) are not listed yet: held until their models and prices are settled.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from media_engine.agents.image_generation.agent import OPERATIONS
from media_engine.contracts import Plan
from media_engine.gateway.image_registry import ImageModelSpec, ImageRegistry, image_registry
from media_engine.services.create.loader import CreateConfig, get_create_config

# Registry tier -> the plan that unlocks it ("entry" is the Basic/free plan's tier).
TIER_PLAN: dict[str, Plan] = {"entry": "free", "pro": "pro", "max": "max", "ultra": "ultra"}
PLAN_RANK: dict[str, int] = {"free": 0, "pro": 1, "max": 2, "ultra": 3}
RESOLUTIONS = ("1K", "2K", "4K")  # ascending; the plan's ceiling is the highest it may send
LIVE_OPERATIONS = frozenset({"create"})

OperationState = Literal["live", "coming_soon"]


class CatalogModel(BaseModel):
    id: str  # what the client sends back as its pick: the slug, plus "@quality" for a variant
    slug: str
    name: str
    plan: Plan  # the lowest plan that may use this option
    settings: dict[str, str]  # billed settings this option fixes, e.g. {"quality": "high"}
    price_usd: float | None  # measured per image (OpenRouter usage.cost); None = not measured
    locked: bool
    upgrade_to: Plan | None  # set when locked: the plan that unlocks it
    batch_only: bool  # too slow for an interactive run (e.g. GPT Image 2 high): batch use only


class Catalog(BaseModel):
    plan: Plan
    default_model: Literal["auto"] = "auto"
    models: list[CatalogModel]
    max_images: int
    aspect_ratios: list[str]
    default_aspect_ratio: str
    max_resolution: str
    resolutions: list[str]  # what the resolution picker may offer: 1K up to max_resolution
    modes: list[str]
    operations: dict[str, OperationState]


def _options(spec: ImageModelSpec, plan: Plan) -> list[CatalogModel]:
    """One option per priced quality (each with the plan its quality needs), else one option."""
    variants: list[tuple[dict[str, str], float | None, str]] = []
    if spec.price_by_quality:
        for quality, quality_price in spec.price_by_quality.items():
            variants.append(({"quality": quality}, quality_price, f"{spec.name} ({quality})"))
    else:
        variants.append(({}, spec.price_measured_usd, spec.name))

    options = []
    for settings, price, name in variants:
        needs = TIER_PLAN[spec.tier]
        for setting, value in settings.items():
            tier = spec.min_tier_for.get(setting, {}).get(value)
            if tier is not None and PLAN_RANK[TIER_PLAN[tier]] > PLAN_RANK[needs]:
                needs = TIER_PLAN[tier]
        locked = spec.plan_block(plan, settings) is not None
        suffix = "".join(f"@{v}" for v in settings.values())
        options.append(
            CatalogModel(
                id=f"{spec.slug}{suffix}",
                slug=spec.slug,
                name=name,
                plan=needs,
                settings=settings,
                price_usd=price,
                locked=locked,
                upgrade_to=needs if locked else None,
                batch_only=spec.is_batch_only(settings),
            )
        )
    return options


def _max_resolution(plan: Plan, specs: list[ImageModelSpec]) -> str:
    """The highest resolution tier the plan may send to at least one model it can use."""
    usable = [s for s in specs if s.allowed_for(plan)]
    for tier in reversed(RESOLUTIONS):
        if any(s.allowed_for(plan, {"resolution": tier}) for s in usable):
            return tier
    return RESOLUTIONS[0]


def catalog(
    plan: Plan, *, registry: ImageRegistry | None = None, cfg: CreateConfig | None = None
) -> Catalog:
    registry = registry or image_registry()
    cfg = cfg or get_create_config()
    specs = [s for s in registry.all() if s.available and s.offered]
    models = [option for spec in specs for option in _options(spec, plan)]
    models.sort(
        key=lambda m: (
            PLAN_RANK[m.plan],
            m.price_usd if m.price_usd is not None else float("inf"),
            m.name,
        )
    )
    ceiling = _max_resolution(plan, specs)
    return Catalog(
        plan=plan,
        models=models,
        max_images=cfg.policy.max_images_per_request[plan],
        aspect_ratios=list(cfg.presets.aspect_ratios),
        default_aspect_ratio=cfg.presets.defaults.aspect_ratio,
        max_resolution=ceiling,
        resolutions=list(RESOLUTIONS[: RESOLUTIONS.index(ceiling) + 1]),
        modes=["auto", "manual"],
        operations={op: "live" if op in LIVE_OPERATIONS else "coming_soon" for op in OPERATIONS},
    )


def resolve_pick(
    model_id: str, plan: Plan, *, registry: ImageRegistry | None = None
) -> CatalogModel:
    """The catalogue option `model_id` names, if this plan may pick it on an interactive run.
    Raises ValueError (with a reason for the user) when it is unknown, locked or batch only.
    Checked before any model call; the gateway enforces the plan again on the call itself."""
    options = {m.id: m for m in catalog(plan, registry=registry).models}
    option = options.get(model_id)
    if option is None:
        raise ValueError(f"model {model_id!r} is not in the catalogue")
    if option.locked:
        raise ValueError(f"model {model_id!r} needs the {option.upgrade_to} plan")
    if option.batch_only:
        raise ValueError(f"model {model_id!r} is batch only, not for an interactive run")
    return option


__all__ = ["Catalog", "CatalogModel", "catalog", "resolve_pick"]
