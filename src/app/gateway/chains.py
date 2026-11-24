"""Read gateway/chains.yaml. PROVIDER_MODE picks the chain."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from app.core.config import Settings


class LLMChoice(BaseModel):
    provider: str
    model: str


class Chain(BaseModel):
    image_providers: list[str]
    enhancer: LLMChoice
    critic: LLMChoice
    paid: set[str] = set()  # providers that cost money (from the `providers` map)


def load_chain(settings: Settings, path: Path | None = None) -> Chain:
    src = path or Path(__file__).with_name("chains.yaml")
    data: dict[str, Any] = yaml.safe_load(src.read_text("utf-8")) or {}
    chain = Chain.model_validate(data[settings.provider_mode])
    paid = {name for name, info in (data.get("providers") or {}).items() if info.get("paid")}
    # Paid providers join the chain only when the owner turns each one on.
    providers = [p for p in chain.image_providers if p not in paid]
    if settings.fal_images:
        providers = ["fal", *providers]
    if settings.openrouter_images:
        providers = ["openrouter", *providers]
    return chain.model_copy(update={"image_providers": providers, "paid": paid})
