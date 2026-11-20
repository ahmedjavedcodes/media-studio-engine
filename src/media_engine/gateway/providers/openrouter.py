"""OpenRouter (paid fallback): images via POST /api/v1/images, chat via /api/v1/chat/completions.

Image request/response shapes follow OpenRouter's API reference (checked 2026-11-13):
request `model`, `prompt`, `n`, `seed`, `aspect_ratio`, `resolution`, `quality`,
`input_references` (`[{"type": "image_url", "image_url": {"url": <data URL | https URL>}}]`);
response `data[].b64_json`, `usage.cost` (USD).

What a model accepts comes from THE image registry (`image_registry.image_registry()`:
models.yaml merged with the catalogue cache; one instance per process, shared with the model
cards and the gateway): a seed is sent only to a model that takes one, and a model with quality
settings always gets one explicitly (its registry default unless the request names one).

`route_image` is the gateway's pre-call guard: plan tier, price-multiplying settings, and `n`
(routed to a same-tier model that can return that many in one call, never silently reduced).
"""

from __future__ import annotations

import base64
import math
from typing import Any

import httpx

from media_engine.contracts import Plan
from media_engine.core.config import Settings
from media_engine.core.images import sniff_media_type
from media_engine.gateway.errors import ProviderError
from media_engine.gateway.image_registry import ImageModelSpec, ImageRegistry, image_registry
from media_engine.gateway.openrouter_models import validate_image_params
from media_engine.gateway.providers.base import (
    DEFAULT_TIMEOUT,
    IMAGE_TIMEOUT,
    Provider,
    inject_schema,
    parse_chat_completion,
)
from media_engine.gateway.types import (
    ImageRequest,
    ImageResponse,
    LLMRequest,
    LLMResponse,
    Route,
)

# Normalised resolution tiers from the API reference, as (max long side in px, tier).
_RESOLUTION_TIERS = [(512, "512"), (768, "768"), (1024, "1K"), (1536, "1.5K"), (2048, "2K")]
_ASPECT_RATIOS = [
    (1, 1), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4),
    (9, 16), (16, 9), (1, 3), (3, 1), (1, 4), (4, 1), (21, 9), (9, 21),
]  # fmt: skip


def derive_aspect_ratio(width: int, height: int) -> str:
    """Closest documented aspect ratio to width:height."""
    target = math.log(width / height)
    w, h = min(_ASPECT_RATIOS, key=lambda r: abs(math.log(r[0] / r[1]) - target))
    return f"{w}:{h}"


_TIER_ORDER = ["512", "768", "1K", "1.5K", "2K", "4K"]


def _ratio_value(text: str) -> float | None:
    w, sep, h = text.partition(":")
    try:
        return float(w) / float(h) if sep else None
    except (ValueError, ZeroDivisionError):
        return None


def closest_aspect_ratio(width: int, height: int, allowed: list[str] | None) -> str:
    """Closest ratio to width:height among `allowed` (a model's enum), else the generic list."""
    if not allowed:
        return derive_aspect_ratio(width, height)
    target = math.log(width / height)
    candidates = [(a, _ratio_value(a)) for a in allowed]
    usable = [(a, r) for a, r in candidates if r is not None and r > 0]
    if not usable:
        return derive_aspect_ratio(width, height)
    return min(usable, key=lambda ar: abs(math.log(ar[1]) - target))[0]


def closest_resolution(width: int, height: int, allowed: list[str] | None) -> str:
    """Derived tier, moved to the smallest supported tier at or above it (else the largest)."""
    wanted = derive_resolution(width, height)
    if not allowed or wanted in allowed:
        return wanted
    ordered = [t for t in _TIER_ORDER if t in allowed]
    if not ordered:
        return wanted
    for tier in ordered:
        if _TIER_ORDER.index(tier) >= _TIER_ORDER.index(wanted):
            return tier
    return ordered[-1]


def derive_resolution(width: int, height: int) -> str:
    side = max(width, height)
    for limit, tier in _RESOLUTION_TIERS:
        if side <= limit:
            return tier
    return "4K"


def encode_reference(data: bytes) -> dict[str, Any]:
    mime = sniff_media_type(data) or "image/png"
    url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    return {"type": "image_url", "image_url": {"url": url}}


class OpenRouterProvider(Provider):
    name = "openrouter"

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        super().__init__(client)
        self._key = settings.openrouter_api_key
        self._base = settings.openrouter_base_url.rstrip("/")

    @property
    def registry(self) -> ImageRegistry:
        """Always the process-wide registry: never a second copy that could drift."""
        return image_registry()

    def spec(self, model: str) -> ImageModelSpec | None:
        return self.registry.get(model)

    def accepts_seed(self, model: str) -> bool:
        """Registry first; else the catalogue cache; else unknown, so the seed is sent."""
        spec = self.spec(model)
        if spec is not None:
            return spec.supports_seed
        info = self.registry.info(model)
        if info is not None:
            return "seed" in info.supported_parameters
        return True

    def billed_settings(self, model: str, request: ImageRequest) -> dict[str, str]:
        """The price-relevant settings this request would SEND to `model` (defaults included):
        what the plan guard checks. Never the raw request: an unset quality is sent as the
        registry default, and that default is what gets billed."""
        payload = self.build_image_payload(model, request, encode_references=False)
        return {k: str(payload[k]) for k in ("quality", "resolution") if k in payload}

    def route_image(self, model: str, request: ImageRequest, plan: Plan) -> Route:
        """Pre-call guard (the gateway calls this before every OpenRouter image entry).

        1. A model missing from the registry is refused: its price and tier are unknown, so no
           plan may spend on it (fail closed).
        2. Plan guard: the model's tier, and every price-multiplying setting actually sent
           (`min_tier_for`, e.g. gpt-image-2 quality high = ultra only).
        3. `n`: (a) the model returns that many per call: use it; (b) else the cheapest
           available same-tier model that can, and that passes the plan guard; (c) else refuse,
           naming the limit. Never silently reduced.
        """
        spec = self.spec(model)
        if spec is None:
            return Route(None, f"{model} is not in the image registry (unknown price and tier)")
        blocked = spec.plan_block(plan, self.billed_settings(model, request))
        if blocked is not None:
            return Route(None, blocked)
        if request.n <= spec.n_per_call:
            return Route(model)
        for other in self.registry.same_tier(spec):
            if other.n_per_call < request.n:
                continue
            if other.plan_block(plan, self.billed_settings(other.slug, request)) is None:
                return Route(
                    other.slug,
                    f"n={request.n} > {model} per-call max {spec.n_per_call}; "
                    f"routed to {other.slug} (same {spec.tier} tier, max {other.n_per_call})",
                )
        return Route(
            None,
            f"n={request.n} exceeds the per-call max of {model} ({spec.n_per_call}) and of "
            f"every other {spec.tier}-tier model allowed for plan {plan}",
        )

    def is_configured(self) -> bool:
        return self._key is not None

    def _headers(self, model: str) -> dict[str, str]:
        return self._bearer(model, self._key, "OPENROUTER_API_KEY")

    def _error_from_response(self, model: str, response: httpx.Response) -> ProviderError:
        error = super()._error_from_response(model, response)
        # Moderation: 403 with error.metadata.reasons (OpenRouter errors reference). A 403
        # without reasons is something else (spending limit, permissions), not a refusal.
        if response.status_code == 403 and _moderation_reasons(response):
            error.refused = True
        return error

    # -- chat ---------------------------------------------------------------------------------

    async def chat(self, model: str, request: LLMRequest) -> LLMResponse:
        messages = request.messages
        if request.response_schema is not None:
            messages = inject_schema(messages, request.response_schema)
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if request.tools:
            payload["tools"] = request.tools
        if request.reasoning_effort and request.reasoning_effort != "none":
            payload["reasoning"] = {"effort": request.reasoning_effort}
        if request.include_reasoning:
            payload["include_reasoning"] = True
        response = await self._post(
            model,
            f"{self._base}/chat/completions",
            headers=self._headers(model),
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        data = response.json()
        parsed = parse_chat_completion(self.name, model, data)
        parsed.cost_usd = _cost(data)
        return parsed

    # -- images -------------------------------------------------------------------------------

    def build_image_payload(
        self, model: str, request: ImageRequest, *, encode_references: bool = True
    ) -> dict[str, Any]:
        """Build the request body. Explicit request fields are sent as given (and validated);
        values derived from width/height are adapted to what the model supports, and dropped
        when the model has no such parameter (e.g. gemini-2.5-flash-image has no `resolution`).
        """
        info = self.registry.info(model)
        supported = info.supported_parameters if info else None

        def allowed(name: str) -> list[str] | None:
            descriptor = (supported or {}).get(name) or {}
            values = descriptor.get("values")
            return [str(v) for v in values] if values else None

        payload: dict[str, Any] = {"model": model, "prompt": request.prompt}
        if request.aspect_ratio:
            payload["aspect_ratio"] = request.aspect_ratio
        elif supported is None or "aspect_ratio" in supported:
            payload["aspect_ratio"] = closest_aspect_ratio(
                request.width, request.height, allowed("aspect_ratio")
            )
        if request.resolution:
            payload["resolution"] = request.resolution
        elif supported is None or "resolution" in supported:
            payload["resolution"] = closest_resolution(
                request.width, request.height, allowed("resolution")
            )
        if request.n > 1:
            payload["n"] = request.n
        if request.seed is not None and self.accepts_seed(model):
            payload["seed"] = request.seed
        spec = self.spec(model)
        if request.quality is not None:
            payload["quality"] = request.quality  # explicit: sent as given, then validated
        elif spec is not None and spec.default_quality is not None:
            payload["quality"] = spec.default_quality
        if request.input_images and encode_references:
            payload["input_references"] = [encode_reference(b) for b in request.input_images]
        return payload

    def _validate(self, model: str, payload: dict[str, Any]) -> None:
        params = {
            k: v
            for k, v in payload.items()
            if k in {"aspect_ratio", "resolution", "n", "seed", "quality"}
        }
        n_refs = len(payload.get("input_references") or [])
        cache = self.registry.cache
        problems = validate_image_params(cache, model, params, n_refs) if cache else []
        if problems:
            # Raised as a 400 so the gateway applies the "400 never falls back" rule, without
            # sending (and paying for) a call that would fail.
            raise ProviderError(
                self.name,
                model,
                "rejected locally, not sent: " + "; ".join(problems),
                status=400,
                code="local_validation",
            )

    async def generate_image(self, model: str, request: ImageRequest) -> ImageResponse:
        payload = self.build_image_payload(model, request)
        self._validate(model, payload)
        response = await self._post(
            model,
            f"{self._base}/images",
            headers=self._headers(model),
            json=payload,
            timeout=IMAGE_TIMEOUT,
        )
        try:
            data = response.json()
            items = data["data"]
            images = [base64.b64decode(item["b64_json"]) for item in items]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(self.name, model, "malformed image response", status=502) from exc
        if not images:
            raise ProviderError(self.name, model, "response contained no images", status=502)
        media_types = [_media_type(item, image) for item, image in zip(items, images, strict=True)]
        return ImageResponse(images=images, cost_usd=_cost(data), media_types=media_types)


def _media_type(item: dict[str, Any], image: bytes) -> str | None:
    """The response's own media type when it gives one, else read from the bytes."""
    for key in ("media_type", "mime_type", "content_type"):
        value = item.get(key)
        if isinstance(value, str) and value.startswith("image/"):
            return value
    fmt = item.get("output_format")
    if isinstance(fmt, str) and fmt:
        return {"jpg": "image/jpeg", "svg": "image/svg+xml"}.get(fmt, f"image/{fmt}")
    return sniff_media_type(image)


def _moderation_reasons(response: httpx.Response) -> list[Any]:
    try:
        body = response.json()
    except ValueError:
        return []
    error = body.get("error") if isinstance(body, dict) else None
    metadata = error.get("metadata") if isinstance(error, dict) else None
    reasons = metadata.get("reasons") if isinstance(metadata, dict) else None
    return reasons if isinstance(reasons, list) else []


def _cost(data: dict[str, Any]) -> float | None:
    usage = data.get("usage")
    if isinstance(usage, dict):
        cost = usage.get("cost")
        if isinstance(cost, int | float):
            return float(cost)
    return None
