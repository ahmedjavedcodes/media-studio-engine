"""fal.ai image models (paid, OFF until the owner sets FAL_IMAGES=true).

`POST https://fal.run/<endpoint_id>` with `Authorization: Key <FAL_KEY>`. The reply lists image
URLs, which we download. The size is always explicit (`image_size` as {width, height}). Every call
goes through the SpendGuard, so SPEND_CAP_USD applies.

UNVERIFIED until the owner turns fal on: written from fal's public docs and mock-tested only.
The reply has no cost field, so a call is charged at the registry estimate.
"""

from typing import Any
from urllib.parse import urlparse

import httpx

from app.contracts.ports import GeneratedImage, ImageOptions
from app.core.errors import ProviderCallFailed
from app.gateway.budget import SpendGuard
from app.gateway.providers.http import post_json

FAL_RUN_URL = "https://fal.run"
# Image files are fetched from the URL fal returns. Only fal's own hosts, only over https.
_IMAGE_HOSTS = ("fal.media", "fal.ai", "fal.run")


def _allowed_image_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return parsed.scheme == "https" and any(
        host == h or host.endswith("." + h) for h in _IMAGE_HOSTS
    )


class FalImages:
    name = "fal"

    def __init__(self, client: httpx.AsyncClient, *, api_key: str, guard: SpendGuard) -> None:
        self._client = client
        self._api_key = api_key
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
            raise ProviderCallFailed("fal key is not set", retryable=False)
        body: dict[str, Any] = {
            "prompt": prompt,
            "image_size": {"width": width, "height": height},  # always explicit
            "num_images": 1,
            "output_format": "jpeg",
            "enable_safety_checker": True,
        }
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        if seed is not None:
            body["seed"] = seed

        async def call() -> bytes:
            data = await post_json(
                self._client,
                f"{FAL_RUN_URL}/{slug}",
                headers={"Authorization": f"Key {self._api_key}"},
                json=body,
                timeout=timeout,
            )
            return await self._download(_first_image_url(data), timeout)

        # The guard reserves the estimate first. fal reports no cost, so the estimate is charged.
        image = await self._guard.run(estimated_cost_usd, model=slug, call=call)
        return GeneratedImage(data=image, cost_usd=None)

    async def _download(self, url: str, timeout: float) -> bytes:
        if not _allowed_image_url(url):
            raise ProviderCallFailed("fal returned an unexpected image address", retryable=False)
        try:
            resp = await self._client.get(url, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise ProviderCallFailed("fal image download timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderCallFailed("fal image download failed", retryable=True) from exc
        if resp.status_code >= 400:
            raise ProviderCallFailed(
                f"fal image download returned {resp.status_code}",
                retryable=resp.status_code >= 500 or resp.status_code == 429,
            )
        return resp.content


def _first_image_url(data: dict[str, Any]) -> str:
    try:
        url = data["images"][0]["url"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderCallFailed("fal returned no image", retryable=False) from exc
    if not isinstance(url, str) or not url:
        raise ProviderCallFailed("fal returned a bad image address", retryable=False)
    return url
