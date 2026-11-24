"""OpenRouter image models (paid, off until the owner asks).

Images come back from /chat/completions with `modalities: [image, text]`, as base64 data URLs in
`message.images`. UNVERIFIED until the owner turns OpenRouter on and we make a first call. Every
call is checked against SPEND_CAP_USD first, and the billed `usage.cost` is recorded after.
"""

import base64
from typing import Any

import httpx

from app.contracts.ports import GeneratedImage
from app.core.errors import ProviderCallFailed
from app.gateway.budget import SpendGuard
from app.gateway.providers.http import post_json


class OpenRouterImages:
    name = "openrouter"

    def __init__(
        self, client: httpx.AsyncClient, *, api_key: str, base_url: str, guard: SpendGuard
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/chat/completions"
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
    ) -> GeneratedImage:
        if not self._api_key:
            raise ProviderCallFailed("openrouter key is not set", retryable=False)
        body: dict[str, Any] = {
            "model": slug,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image", "text"],
            # An explicit size. Some models default to 2K and bill double.
            "image_config": {"aspect_ratio": aspect_ratio, "image_size": "1K"},
        }
        if seed is not None:
            body["seed"] = seed

        async def call() -> dict[str, Any]:
            return await post_json(
                self._client,
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=body,
                timeout=timeout,
            )

        def billed(data: dict[str, Any]) -> float | None:
            usage = data.get("usage")
            cost = usage.get("cost") if isinstance(usage, dict) else None
            return float(cost) if isinstance(cost, int | float) else None

        # The guard reserves the estimate first, so parallel variations cannot pass the cap.
        data = await self._guard.run(estimated_cost_usd, model=slug, call=call, billed=billed)
        return GeneratedImage(data=_first_image(data), cost_usd=billed(data))


def _first_image(data: dict[str, Any]) -> bytes:
    try:
        url = data["choices"][0]["message"]["images"][0]["image_url"]["url"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderCallFailed("openrouter returned no image", retryable=False) from exc
    if not isinstance(url, str) or "," not in url:
        raise ProviderCallFailed("openrouter returned a bad image", retryable=False)
    try:
        return base64.b64decode(url.split(",", 1)[1])
    except ValueError as exc:
        raise ProviderCallFailed("openrouter returned a bad image", retryable=False) from exc
