"""OpenRouter image model registry: models.yaml (curated) merged with the catalogue cache.

Two layers, merged at load time (IMAGE-CREATE-OPTIMIZE v2.1 §6.3):

- `models.yaml` is the source of truth for what we MEASURED or DECIDED: billed price, tier,
  prompt style, word budget, whether a seed is verified to reproduce an image.
- The catalogue cache (`openrouter_models.ModelCache`, refreshed MANUALLY by
  playground/sync_openrouter_models.py, never at startup or on a call) is the source of truth
  for what a model ACCEPTS: seed, n range, input references, aspect ratios, resolutions,
  quality values, mask.

When both state a capability and disagree, the catalogue wins and a warning is recorded. With
no cache file, the yaml values are used as they are. Loading never makes a network call.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from media_engine.contracts import Plan
from media_engine.core.logging import get_logger
from media_engine.gateway.openrouter_models import ImageModelInfo, ModelCache

log = get_logger("image_registry")

MODELS_PATH = Path(__file__).with_name("models.yaml")

Tier = Literal["entry", "pro", "max", "ultra"]
BillingUnit = Literal["image", "megapixel", "token"]
PromptStyle = Literal["nouns", "sentences", "tags"]

# Which plans may call a model of each tier. `entry` models are the PAID fallback for the free
# plan (used only once Cloudflare's free allocation is exhausted), so every plan may call them.
_TIER_RANK: dict[str, int] = {"entry": 0, "pro": 1, "max": 2, "ultra": 3}
_PLAN_RANK: dict[str, int] = {"free": 0, "pro": 1, "max": 2, "ultra": 3}


class ModelEntry(BaseModel):
    """One model as written in models.yaml."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slug: str
    name: str
    tier: Tier
    billing_unit: BillingUnit
    price_measured_usd: float | None
    price_edit_usd: float | None = None
    price_by_quality: dict[str, float] = Field(default_factory=dict)
    min_tier_for_quality: dict[str, Tier] = Field(default_factory=dict)
    quality_settings: list[str] = Field(default_factory=list)
    default_quality: str | None = None
    supports_seed: bool
    seed_verified: bool = False  # same seed reproduced the same image in a real test
    n_per_call: int = Field(ge=1)
    max_input_images: int = Field(ge=0)
    default_size: str | None = None  # documentation: size MUST still be sent on every call
    prompt_style: PromptStyle
    prompt_max_words: int = Field(gt=0)
    eval_default: bool = False
    available: bool = True
    notes: str | None = None


class ImageModelSpec(BaseModel):
    """A model as the gateway uses it: models.yaml merged with the catalogue (§6.3 fields)."""

    model_config = ConfigDict(frozen=True)

    slug: str
    name: str
    tier: Tier
    billing_unit: BillingUnit
    price_measured_usd: float | None
    price_edit_usd: float | None
    price_by_quality: dict[str, float]
    min_tier_for_quality: dict[str, Tier]
    supports_seed: bool
    seed_verified: bool
    n_per_call: int
    max_input_images: int
    supports_edit: bool
    supports_mask: bool
    supported_aspect_ratios: list[str]
    supported_sizes: list[str]  # catalogue `resolution` tiers; empty = the model has none
    default_size: str | None
    quality_settings: list[str]
    default_quality: str | None
    prompt_style: PromptStyle
    prompt_max_words: int
    eval_default: bool
    available: bool
    in_catalogue: bool  # False: no cache, or the cache does not list this slug
    listed_pricing: list[dict[str, Any]]  # catalogue /endpoints pricing (listed, not billed)

    def allowed_for(self, plan: Plan, quality: str | None = None) -> bool:
        """Whether `plan` may call this model (and at this quality setting)."""
        rank = _PLAN_RANK[plan]
        if rank < _TIER_RANK[self.tier]:
            return False
        needed = self.min_tier_for_quality.get(quality or "")
        return needed is None or rank >= _TIER_RANK[needed]


def _range_max(info: ImageModelInfo, name: str) -> int | None:
    descriptor = info.supported_parameters.get(name) or {}
    if descriptor.get("type") == "range" and isinstance(descriptor.get("max"), int):
        return int(descriptor["max"])
    return None


def _enum(info: ImageModelInfo, name: str) -> list[str]:
    descriptor = info.supported_parameters.get(name) or {}
    return [str(v) for v in descriptor.get("values") or []]


def merge(entry: ModelEntry, info: ImageModelInfo | None) -> tuple[ImageModelSpec, list[str]]:
    """Merge one yaml entry with its catalogue record. Returns the spec and any warnings."""
    warnings: list[str] = []
    seed, n_per_call, max_refs = entry.supports_seed, entry.n_per_call, entry.max_input_images
    quality = list(entry.quality_settings)
    aspect_ratios: list[str] = []
    sizes: list[str] = []
    mask = False

    if info is not None:

        def check(field: str, expected: object, actual: object) -> None:
            if expected != actual:
                warnings.append(
                    f"{entry.slug}: models.yaml {field}={expected!r} but the catalogue says "
                    f"{actual!r}; using the catalogue"
                )

        api_seed = "seed" in info.supported_parameters
        check("supports_seed", seed, api_seed)
        seed = api_seed

        api_n = _range_max(info, "n") or 1
        check("n_per_call", n_per_call, api_n)
        n_per_call = api_n

        api_refs = _range_max(info, "input_references")
        if api_refs is None:  # no parameter: refs only if the model takes image input at all
            api_refs = max_refs if info.accepts_image_input else 0
        check("max_input_images", max_refs, api_refs)
        max_refs = api_refs

        api_quality = _enum(info, "quality")
        if quality or api_quality:
            check("quality_settings", quality, api_quality)
        quality = api_quality

        aspect_ratios = _enum(info, "aspect_ratio")
        sizes = _enum(info, "resolution")
        mask = "mask" in info.supported_parameters
        if not info.supported_parameters:
            warnings.append(f"{entry.slug}: the catalogue lists no parameters for this model")

    default_quality = entry.default_quality
    if default_quality is not None and default_quality not in quality:
        warnings.append(
            f"{entry.slug}: default_quality {default_quality!r} is not one of {quality}; "
            "it will not be sent"
        )
        default_quality = None
    if entry.seed_verified and not seed:
        warnings.append(f"{entry.slug}: seed_verified but the model takes no seed")

    spec = ImageModelSpec(
        slug=entry.slug,
        name=entry.name,
        tier=entry.tier,
        billing_unit=entry.billing_unit,
        price_measured_usd=entry.price_measured_usd,
        price_edit_usd=entry.price_edit_usd,
        price_by_quality=dict(entry.price_by_quality),
        min_tier_for_quality=dict(entry.min_tier_for_quality),
        supports_seed=seed,
        seed_verified=entry.seed_verified and seed,
        n_per_call=n_per_call,
        max_input_images=max_refs,
        supports_edit=max_refs > 0,
        supports_mask=mask,
        supported_aspect_ratios=aspect_ratios,
        supported_sizes=sizes,
        default_size=entry.default_size,
        quality_settings=quality,
        default_quality=default_quality,
        prompt_style=entry.prompt_style,
        prompt_max_words=entry.prompt_max_words,
        eval_default=entry.eval_default,
        available=entry.available,
        in_catalogue=info is not None,
        listed_pricing=list(info.pricing) if info is not None else [],
    )
    return spec, warnings


class ImageRegistry:
    def __init__(self, specs: list[ImageModelSpec], warnings: list[str]) -> None:
        self._by_slug = {s.slug: s for s in specs}
        self.warnings = warnings

    @classmethod
    def load(cls, cache: ModelCache | None = None, path: Path = MODELS_PATH) -> ImageRegistry:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = [ModelEntry.model_validate(m) for m in raw.get("models", [])]
        slugs = [e.slug for e in entries]
        duplicates = sorted({s for s in slugs if slugs.count(s) > 1})
        if duplicates:
            raise ValueError(f"models.yaml lists these slugs more than once: {duplicates}")
        eval_defaults = [e.slug for e in entries if e.eval_default]
        if len(eval_defaults) > 1:
            raise ValueError(f"models.yaml has several eval_default models: {eval_defaults}")

        cache_loaded = cache is not None and cache.loaded
        specs: list[ImageModelSpec] = []
        warnings: list[str] = []
        for entry in entries:
            info = cache.get(entry.slug) if cache is not None and cache_loaded else None
            if cache_loaded and info is None:
                warnings.append(
                    f"{entry.slug}: not in the catalogue cache (synced {cache.synced_at})"  # type: ignore[union-attr]
                )
            spec, problems = merge(entry, info)
            specs.append(spec)
            warnings.extend(problems)
        for warning in warnings:
            log.warning("image_registry.mismatch", detail=warning)
        return cls(specs, warnings)

    def get(self, slug: str) -> ImageModelSpec | None:
        return self._by_slug.get(slug)

    def all(self) -> list[ImageModelSpec]:
        return list(self._by_slug.values())

    def for_plan(self, plan: Plan) -> list[ImageModelSpec]:
        return [s for s in self._by_slug.values() if s.available and s.allowed_for(plan)]

    def eval_default(self) -> ImageModelSpec | None:
        return next((s for s in self._by_slug.values() if s.eval_default), None)


@lru_cache(maxsize=1)
def image_registry() -> ImageRegistry:
    """Process-wide registry from models.yaml and the default catalogue cache file."""
    return ImageRegistry.load(ModelCache())
