"""Model registry: one model, many providers. Spec section 6.

Cards live in models.yaml. Costs there are estimates until the daily sync (gateway/sync.py) and
the billed cost after each call correct them. The tier is always computed from the cost.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from app.core.errors import BadInput
from app.core.plans import Plan, Tier, tier_allowed, tier_for_cost

Status = Literal["hidden", "testing", "live", "down"]

ALL_RATIOS = ["1:1", "4:5", "5:4", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9"]


TEXT_SKILLS = ("typography", "dense_text")  # a model that can be trusted with text in the image


class ProviderEntry(BaseModel):
    name: str
    slug: str
    cost: float  # USD per image at 1K


class ModelCard(BaseModel):
    id: str
    name: str
    family: str
    providers: list[ProviderEntry]
    tier: Tier = Tier.BASIC  # computed from the cheapest provider cost
    status: Status = "hidden"
    featured: bool = False
    tag: str = ""
    commercial_use: bool = True
    has_safety_filter: bool = False
    supports_seed: bool = False
    seed_honoured: bool | None = None  # None = never tested. True only after a repeat-seed test
    supports_negative_prompt: bool = False
    n_per_call: int = 1
    max_refs: int = 0  # reference images the model accepts (0 = text only)
    capabilities: list[str] = Field(
        default_factory=list
    )  # t2i, i2i, svg_vector, multi_ref_merge ...
    credits: int | None = None  # what a user pays per image. None = the tier's credits
    ref_surcharge: int = 0  # extra credits for each reference image after the first
    resolutions: list[str] = Field(default_factory=list)  # `resolution` values the provider takes
    resolution_multipliers: dict[str, float] = Field(default_factory=dict)  # credit multiplier
    endpoint: Literal["images", "chat"] = "images"  # which OpenRouter API the model uses
    quality: str | None = None  # sent to providers that have a quality knob (GPT Image)
    forbidden_qualities: list[str] = Field(default_factory=list)  # never sent (cost guardrail)
    output_format: str | None = None  # `svg` for vector models
    notes: str = ""
    picker: bool = False  # shown in the model picker. A hidden model can still be picked by id
    auto: bool = False  # the Auto router may choose it. False = only picked by hand (tests)
    supported_aspect_ratios: list[str] = Field(default_factory=lambda: list(ALL_RATIOS))
    max_resolution: str = "1K"
    prompt_max_words: int = 0
    prompt_guide: str = ""  # copied from the family at load time
    family_label: str = ""  # the family's display name, for the picker
    prompt_style: Literal["keywords", "sentences"] = "keywords"  # for COMPILE_MODE=code
    best_for: list[str] = Field(default_factory=list)  # router categories this model suits
    size_multiple: int = 32  # sides must be a multiple of this
    # The pixel sizes the model really makes ("WxH"). The code guard snaps to the nearest one for
    # the ratio. Empty = any size that is a multiple of `size_multiple`.
    native_sizes: list[str] = Field(default_factory=list)
    # Optional parameters the model really takes. A parameter is sent ONLY when it is listed here
    # AND has a value in `params`. (OpenRouter documents no negative prompt, style, guidance or
    # steps at top level, so the curated models list none of them.)
    supported_params: list[str] = Field(default_factory=list)  # guidance, steps, style
    params: dict[str, float | int | str] = Field(default_factory=dict)
    # Quality by plan for models with a quality knob (GPT Image): low on Basic, medium on Pro and
    # up. "high" is used only when a Max user PICKS the model by hand, never from Auto.
    quality_by_plan: dict[str, str] = Field(default_factory=dict)
    # Estimated USD per image at a quality other than the card's default (spend guard reserve).
    cost_by_quality: dict[str, float] = Field(default_factory=dict)

    @property
    def cost_per_image(self) -> float:
        return min(p.cost for p in self.providers)

    @property
    def handles_text(self) -> bool:
        """True when the model is trusted with text in the image (typography or dense_text)."""
        return any(c in self.capabilities for c in TEXT_SKILLS)

    def cost_for(self, entry: ProviderEntry, quality: str | None) -> float:
        """USD estimate for one image from `entry` at `quality`."""
        if quality is not None and quality != self.quality:
            return self.cost_by_quality.get(quality, entry.cost)
        return entry.cost

    @model_validator(mode="after")
    def _derive(self) -> "ModelCard":
        self.tier = tier_for_cost(self.cost_per_image)
        if self.quality is not None and self.quality in self.forbidden_qualities:
            raise ValueError(f"{self.id}: quality {self.quality!r} is forbidden for this model")
        return self

    def providers_in(self, enabled: list[str]) -> list[ProviderEntry]:
        """Providers this run may use, cheapest first. `enabled` is the chain's list."""
        usable = [p for p in self.providers if p.name in enabled]
        return sorted(usable, key=lambda p: p.cost)


class Registry:
    def __init__(
        self,
        cards: list[ModelCard],
        *,
        run_testing: bool = False,
        auto_rank: dict[str, dict[str, list[str]]] | None = None,
    ) -> None:
        self._cards = {c.id: c for c in cards}
        self._runnable = ("live", "testing") if run_testing else ("live",)
        # {category: {plan: [model ids, best first]}}, written by playground/eval.py
        self.auto_rank = auto_rank or {}

    def ranked(self, category: str, plan: Plan) -> list[str]:
        """Model ids the eval found best for this category and plan, best first. May be empty."""
        return list(self.auto_rank.get(category, {}).get(plan.value, []))

    def all(self) -> list[ModelCard]:
        return list(self._cards.values())

    def get(self, model_id: str) -> ModelCard | None:
        return self._cards.get(model_id)

    def live(self, enabled_providers: list[str]) -> list[ModelCard]:
        """Models a user can pick: status live (or testing, in dev when allowed) and at least
        one enabled provider."""
        return [
            c
            for c in self._cards.values()
            if c.status in self._runnable and c.providers_in(enabled_providers)
        ]

    def listed(self) -> list[ModelCard]:
        """Models the picker shows: the curated list (`picker: true`) that is live, or `testing`
        as "coming soon". A `testing` model cannot be run (the checks and the router only accept
        `live`). Everything else stays in the registry but is not shown."""
        return [c for c in self._cards.values() if c.picker and c.status in ("live", "testing")]

    def allowed(self, plan: Plan, enabled_providers: list[str]) -> list[ModelCard]:
        return [c for c in self.live(enabled_providers) if tier_allowed(plan, c.tier)]


def load_registry(path: Path | None = None, *, run_testing: bool = False) -> Registry:
    src = path or Path(__file__).with_name("models.yaml")
    data = yaml.safe_load(src.read_text("utf-8")) or {}
    raw = data.get("models") or []
    families = data.get("families") or {}
    if not isinstance(raw, list) or not isinstance(families, dict):
        raise BadInput("models.yaml: `models` must be a list and `families` a map")
    cards: list[ModelCard] = []
    for item in raw:
        fam = families.get(item.get("family"))
        if fam is None:
            raise BadInput(f"models.yaml: unknown family for model {item.get('id')!r}")
        merged = {
            "prompt_guide": str(fam.get("prompt_guide", "")).strip(),
            "family_label": str(fam.get("label", item.get("family", ""))),
            "prompt_style": fam.get("style", "keywords"),
            "prompt_max_words": fam.get("max_words", 60),
            **item,  # a model can override the family values
        }
        cards.append(ModelCard.model_validate(merged))
    auto_rank = data.get("auto_rank") or {}
    known = {c.id for c in cards}
    for by_plan in auto_rank.values():  # a stale id in the rank must fail loudly, not route
        for ids in by_plan.values():
            for model_id in ids:
                if model_id not in known:
                    raise BadInput(f"models.yaml: auto_rank names unknown model {model_id!r}")
    return Registry(cards, run_testing=run_testing, auto_rank=auto_rank)


@lru_cache
def get_registry() -> Registry:
    return load_registry()
