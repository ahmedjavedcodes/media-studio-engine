"""OpenRouter image models (paid, OFF until the owner sets OPENROUTER_IMAGES=true).

Two request formats, one adapter:

- `POST /api/v1/images` for most models (GPT Image, Seedream, Recraft, Muse, FLUX ...). Documented,
  with `input_references`, `resolution`, `quality`, `seed`, `background`, `output_format` and
  `usage.cost`. The reply is `data[].b64_json` plus a `media_type`.
- `POST /api/v1/chat/completions` with `modalities: [image, text]` for models marked
  `endpoint: chat` (the Gemini / Nano Banana family). It supports multi-turn editing. The image
  comes back as a data URL in `choices[0].message.images`.

Both formats were checked with real calls on 2026-11-25. Every call goes through the SpendGuard.
Options come from code (gateway/options.py). Nothing here reads model-written text as a parameter.
"""

import asyncio
import base64
import re
from typing import Any

import httpx
import structlog

from app.contracts.ports import GeneratedImage, ImageOptions
from app.core.errors import ProviderCallFailed
from app.gateway.budget import SpendGuard
from app.gateway.providers.http import post_json

log = structlog.get_logger("openrouter")

RETRY_PAUSE_S = 1.0
# GPT Image 2.6 uses about 4x the tokens at `low`, so it is never sent (cost guardrail).
_NO_LOW_QUALITY = re.compile(r"^openai/gpt-image-2\.6")


def _rgb(hex_color: str) -> list[int]:
    h = hex_color.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)]


def _check_quality(slug: str, quality: str | None) -> None:
    if quality == "low" and _NO_LOW_QUALITY.match(slug):
        raise ProviderCallFailed("low quality is not allowed for this model", retryable=False)


def build_images_body(
    slug: str,
    prompt: str,
    aspect_ratio: str,
    seed: int | None,
    options: ImageOptions,
) -> dict[str, Any]:
    """The body for POST /api/v1/images."""
    _check_quality(slug, options.quality)
    body: dict[str, Any] = {
        "model": slug,
        "prompt": prompt,
        "n": 1,
        "aspect_ratio": aspect_ratio,  # an explicit size: the ratio, plus `resolution` if taken
        "response_format": "b64_json",
    }
    if options.resolution:
        body["resolution"] = options.resolution
    if options.quality:
        body["quality"] = options.quality
    if seed is not None:
        body["seed"] = seed
    if options.output_format:
        body["output_format"] = options.output_format
    if options.background:
        body["background"] = options.background
    if options.references:  # omitted when there are none
        body["input_references"] = [
            {"type": "image_url", "image_url": {"url": url}} for url in options.references
        ]
    provider_options: dict[str, Any] = {}
    if options.brand_colors and slug.startswith("recraft/"):
        # No `rgb_colors` field exists. Recraft takes its palette through `provider.options`.
        provider_options["recraft"] = {
            "controls": {"colors": [{"rgb": _rgb(c)} for c in options.brand_colors]}
        }
    if options.complexity is not None and _NO_LOW_QUALITY.match(slug):
        provider_options["openai"] = {"complexity": options.complexity}
    if provider_options:
        body["provider"] = {"options": provider_options}
    return body


def build_chat_body(
    slug: str, prompt: str, aspect_ratio: str, options: ImageOptions
) -> dict[str, Any]:
    """The body for POST /api/v1/chat/completions (Gemini). Single turn, or a conversation."""
    if options.references:
        content: Any = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": url}} for url in options.references
        ]
    else:
        content = prompt
    image_config: dict[str, Any] = {"aspect_ratio": aspect_ratio}
    if options.resolution:
        image_config["image_size"] = options.resolution
    return {
        "model": slug,
        "messages": [*options.history, {"role": "user", "content": content}],
        "modalities": ["image", "text"],
        "image_config": image_config,
    }


def parse_images_reply(data: dict[str, Any]) -> tuple[bytes, str | None]:
    """(file bytes, media type) from an /images reply. SVG comes back as base64 text too."""
    try:
        item = data["data"][0]
        raw = base64.b64decode(item["b64_json"], validate=True)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ProviderCallFailed("openrouter returned no image", retryable=False) from exc
    if not raw:
        raise ProviderCallFailed("openrouter returned an empty image", retryable=False)
    media = item.get("media_type")
    return raw, media if isinstance(media, str) else None


def parse_chat_reply(data: dict[str, Any]) -> tuple[bytes, str | None]:
    """(file bytes, media type) from a chat reply: `choices[0].message.images[0]` as a data URL."""
    try:
        url = data["choices"][0]["message"]["images"][0]["image_url"]["url"]
        head, payload = url.split(",", 1)
        media = head.removeprefix("data:").split(";")[0] or None
        raw = base64.b64decode(payload, validate=True)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise ProviderCallFailed("openrouter returned no image", retryable=False) from exc
    if not raw:
        raise ProviderCallFailed("openrouter returned an empty image", retryable=False)
    return raw, media


def billed_cost(data: dict[str, Any]) -> float | None:
    usage = data.get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    return float(cost) if isinstance(cost, int | float) else None


class OpenRouterImages:
    name = "openrouter"

    def __init__(
        self, client: httpx.AsyncClient, *, api_key: str, base_url: str, guard: SpendGuard
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base = base_url.rstrip("/")
        self._guard = guard

    async def generate(
        self,
        *,
        slug: str,
        family: str,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        aspect_ratio: str,
        seed: int | None,
        timeout: float,
        estimated_cost_usd: float = 0.0,
        options: ImageOptions | None = None,
    ) -> GeneratedImage:
        if not self._api_key:
            raise ProviderCallFailed("openrouter key is not set", retryable=False)
        opts = options or ImageOptions()
        chat = opts.endpoint == "chat"
        path = "chat/completions" if chat else "images"
        body = (
            build_chat_body(slug, prompt, aspect_ratio, opts)
            if chat
            else build_images_body(slug, prompt, aspect_ratio, seed, opts)
        )

        async def call() -> dict[str, Any]:
            return await self._post(path, body, timeout)

        # The guard reserves the estimate first, so parallel variations cannot pass the cap.
        data = await self._guard.run(estimated_cost_usd, model=slug, call=call, billed=billed_cost)
        raw, media = parse_chat_reply(data) if chat else parse_images_reply(data)
        return GeneratedImage(data=raw, cost_usd=billed_cost(data), media_type=media)

    async def _post(self, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        """One retry for a 429 or 5xx (OpenRouter bills nothing for those). A timeout is never
        retried: the first request may still be running and be billed."""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = f"{self._base}/{path}"
        try:
            return await post_json(self._client, url, headers=headers, json=body, timeout=timeout)
        except ProviderCallFailed as exc:
            if not exc.retryable or "timed out" in str(exc):
                raise
            log.warning("openrouter_retry", path=path, error=str(exc))
        await asyncio.sleep(RETRY_PAUSE_S)
        return await post_json(self._client, url, headers=headers, json=body, timeout=timeout)
