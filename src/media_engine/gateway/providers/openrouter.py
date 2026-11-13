"""OpenRouter (paid fallback): images via POST /api/v1/images, chat via /api/v1/chat/completions.

Image request/response shapes follow OpenRouter's API reference (checked 2026-11-13):
request `model`, `prompt`, `n`, `seed`, `aspect_ratio`, `resolution`, `input_references`
(`[{"type": "image_url", "image_url": {"url": <data URL | https URL>}}]`); response
`data[].b64_json`, `usage.cost` (USD).
"""

from __future__ import annotations

import base64
import math
from typing import Any

import httpx

from media_engine.core.config import Settings
from media_engine.gateway.errors import ProviderError
from media_engine.gateway.openrouter_models import ModelCache, validate_image_params
from media_engine.gateway.providers.base import (
    DEFAULT_TIMEOUT,
    IMAGE_TIMEOUT,
    Provider,
    parse_chat_completion,
)
from media_engine.gateway.types import ImageRequest, ImageResponse, LLMRequest, LLMResponse

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


def _mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def encode_reference(data: bytes) -> dict[str, Any]:
    url = f"data:{_mime(data)};base64,{base64.b64encode(data).decode()}"
    return {"type": "image_url", "image_url": {"url": url}}


class OpenRouterProvider(Provider):
    name = "openrouter"

    def __init__(
        self,
        client: httpx.AsyncClient,
        settings: Settings,
        cache: ModelCache | None = None,
    ) -> None:
        super().__init__(client)
        self._key = settings.openrouter_api_key
        self._base = settings.openrouter_base_url.rstrip("/")
        self.cache = cache or ModelCache()

    def is_configured(self) -> bool:
        return self._key is not None

    def _headers(self) -> dict[str, str]:
        assert self._key is not None
        return {"Authorization": f"Bearer {self._key.get_secret_value()}"}

    # -- chat ---------------------------------------------------------------------------------

    async def chat(self, model: str, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {"model": model, "messages": request.messages}
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
            headers=self._headers(),
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        data = response.json()
        parsed = parse_chat_completion(self.name, model, data)
        parsed.cost_usd = _cost(data)
        return parsed

    # -- images -------------------------------------------------------------------------------

    def build_image_payload(self, model: str, request: ImageRequest) -> dict[str, Any]:
        """Build the request body. Explicit request fields are sent as given (and validated);
        values derived from width/height are adapted to what the model supports, and dropped
        when the model has no such parameter (e.g. gemini-2.5-flash-image has no `resolution`).
        """
        info = self.cache.get(model)
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
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.input_images:
            payload["input_references"] = [encode_reference(b) for b in request.input_images]
        return payload

    def _validate(self, model: str, payload: dict[str, Any]) -> None:
        params = {
            k: v for k, v in payload.items() if k in {"aspect_ratio", "resolution", "n", "seed"}
        }
        n_refs = len(payload.get("input_references") or [])
        problems = validate_image_params(self.cache, model, params, n_refs)
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
            headers=self._headers(),
            json=payload,
            timeout=IMAGE_TIMEOUT,
        )
        try:
            data = response.json()
            images = [base64.b64decode(item["b64_json"]) for item in data["data"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(self.name, model, "malformed image response", status=502) from exc
        if not images:
            raise ProviderError(self.name, model, "response contained no images", status=502)
        return ImageResponse(images=images, cost_usd=_cost(data))


def _cost(data: dict[str, Any]) -> float | None:
    usage = data.get("usage")
    if isinstance(usage, dict):
        cost = usage.get("cost")
        if isinstance(cost, int | float):
            return float(cost)
    return None
