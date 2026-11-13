"""Model cards: what each model can do. Providers read these instead of hard-coding quirks.

Unverified values are marked; run `playground/smoke_test.py` to confirm model IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["llm", "image"]
ApiFormat = Literal["json", "multipart"]


@dataclass(frozen=True)
class NeuronPricing:
    """Cloudflare image pricing in neurons. Source: developers.cloudflare.com/workers-ai/platform/pricing
    (checked 2026-11-13; $0.011 per 1,000 neurons). Re-check when Cloudflare changes prices."""

    per_out_tile: float = 0.0  # per 512x512 output tile
    per_step: float = 0.0  # per diffusion step (flat, not per tile)
    per_in_tile: float = 0.0  # per 512x512 input tile
    per_first_mp: float = 0.0  # per first megapixel of output
    per_extra_mp: float = 0.0  # per further output megapixel
    per_in_mp: float = 0.0  # per input megapixel
    default_steps: int | None = None


@dataclass(frozen=True)
class ModelCard:
    provider: str
    model: str
    kind: Kind
    free: bool = True
    supports_edit: bool = False  # accepts reference image(s) as input
    supports_merge: bool = False  # accepts several reference images
    supports_mask: bool = False
    api_format: ApiFormat = "json"  # Cloudflare FLUX.2 models take multipart form data
    max_side: int | None = None
    max_steps: int | None = None
    reasoning: bool = False
    gpt_oss: bool = False  # GPT-OSS uses include_reasoning, not reasoning_format
    neuron_pricing: NeuronPricing | None = None  # image models; LLMs report actual neurons


def _llm(provider: str, model: str, **kw: object) -> ModelCard:
    return ModelCard(provider, model, "llm", **kw)  # type: ignore[arg-type]


def _img(provider: str, model: str, **kw: object) -> ModelCard:
    return ModelCard(provider, model, "image", **kw)  # type: ignore[arg-type]


_CARDS: list[ModelCard] = [
    _llm("cloudflare", "@cf/openai/gpt-oss-120b", reasoning=True, gpt_oss=True),
    _llm("cloudflare", "@cf/openai/gpt-oss-20b", reasoning=True, gpt_oss=True),
    _llm("groq", "openai/gpt-oss-120b", reasoning=True, gpt_oss=True),
    _llm("groq", "openai/gpt-oss-20b", reasoning=True, gpt_oss=True),
    _llm("groq", "openai/gpt-oss-safeguard-20b", reasoning=True, gpt_oss=True),
    _llm("groq", "qwen/qwen3.8-27b", reasoning=True),
    _img(
        "cloudflare",
        "@cf/black-forest-labs/flux-1-schnell",
        max_steps=8,
        max_side=1024,
        neuron_pricing=NeuronPricing(per_out_tile=4.80, per_step=9.60, default_steps=4),
    ),
    _img(
        "cloudflare",
        "@cf/black-forest-labs/flux-2-klein-4b",
        api_format="multipart",
        supports_edit=True,
        supports_merge=True,
        neuron_pricing=NeuronPricing(per_out_tile=26.05, per_in_tile=5.37),
    ),
    _img(
        "cloudflare",
        "@cf/black-forest-labs/flux-2-klein-9b",
        api_format="multipart",
        supports_edit=True,
        supports_merge=True,
        neuron_pricing=NeuronPricing(per_first_mp=1363.64, per_extra_mp=181.82, per_in_mp=181.82),
    ),
]

_BY_KEY = {(c.provider, c.model): c for c in _CARDS}


def get_card(provider: str, model: str) -> ModelCard | None:
    return _BY_KEY.get((provider, model))


def card_or_default(provider: str, model: str, kind: Kind) -> ModelCard:
    return _BY_KEY.get((provider, model)) or ModelCard(provider, model, kind)
