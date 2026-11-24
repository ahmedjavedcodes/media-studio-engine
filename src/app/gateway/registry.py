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
    max_input_images: int = 0
    supported_aspect_ratios: list[str] = Field(default_factory=lambda: list(ALL_RATIOS))
    max_resolution: str = "1K"
    prompt_max_words: int = 0
    prompt_guide: str = ""  # copied from the family at load time
    prompt_style: Literal["keywords", "sentences"] = "keywords"  # for COMPILE_MODE=code
    best_for: list[str] = Field(default_factory=list)  # router categories this model suits
    size_multiple: int = 32  # sides must be a multiple of this

    @property
    def cost_per_image(self) -> float:
        return min(p.cost for p in self.providers)

    @model_validator(mode="after")
    def _derive(self) -> "ModelCard":
        self.tier = tier_for_cost(self.cost_per_image)
        return self

    def providers_in(self, enabled: list[str]) -> list[ProviderEntry]:
        """Providers this run may use, cheapest first. `enabled` is the chain's list."""
        usable = [p for p in self.providers if p.name in enabled]
        return sorted(usable, key=lambda p: p.cost)


class Registry:
    def __init__(self, cards: list[ModelCard]) -> None:
        self._cards = {c.id: c for c in cards}

    def all(self) -> list[ModelCard]:
        return list(self._cards.values())

    def get(self, model_id: str) -> ModelCard | None:
        return self._cards.get(model_id)

    def live(self, enabled_providers: list[str]) -> list[ModelCard]:
        """Models a user can pick: status live and at least one enabled provider."""
        return [
            c
            for c in self._cards.values()
            if c.status == "live" and c.providers_in(enabled_providers)
        ]

    def allowed(self, plan: Plan, enabled_providers: list[str]) -> list[ModelCard]:
        return [c for c in self.live(enabled_providers) if tier_allowed(plan, c.tier)]


def load_registry(path: Path | None = None) -> Registry:
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
            "prompt_style": fam.get("style", "keywords"),
            "prompt_max_words": fam.get("max_words", 60),
            **item,  # a model can override the family values
        }
        cards.append(ModelCard.model_validate(merged))
    return Registry(cards)


@lru_cache
def get_registry() -> Registry:
    return load_registry()
