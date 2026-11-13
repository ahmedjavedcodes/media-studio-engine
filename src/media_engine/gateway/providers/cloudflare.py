"""Cloudflare Workers AI: LLMs (OpenAI-compatible) and images (`/ai/run/@cf/...`)."""

from __future__ import annotations

import base64
from typing import Any

import httpx

from media_engine.core.config import Settings
from media_engine.gateway.errors import ProviderError
from media_engine.gateway.providers.base import (
    DEFAULT_TIMEOUT,
    IMAGE_TIMEOUT,
    Provider,
    parse_chat_completion,
)
from media_engine.gateway.registry import card_or_default
from media_engine.gateway.types import ImageRequest, ImageResponse, LLMRequest, LLMResponse

API_ROOT = "https://api.cloudflare.com/client/v4/accounts"


class CloudflareProvider(Provider):
    name = "cloudflare"

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        super().__init__(client)
        self._account_id = settings.cloudflare_account_id
        self._token = settings.cloudflare_api_token

    def is_configured(self) -> bool:
        return bool(self._account_id and self._token)

    def _headers(self) -> dict[str, str]:
        assert self._token is not None
        return {"Authorization": f"Bearer {self._token.get_secret_value()}"}

    async def chat(self, model: str, request: LLMRequest) -> LLMResponse:
        url = f"{API_ROOT}/{self._account_id}/ai/v1/chat/completions"
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
            payload["reasoning_effort"] = request.reasoning_effort
        response = await self._post(
            model, url, headers=self._headers(), json=payload, timeout=DEFAULT_TIMEOUT
        )
        return parse_chat_completion(self.name, model, response.json())

    async def generate_image(self, model: str, request: ImageRequest) -> ImageResponse:
        card = card_or_default(self.name, model, "image")
        url = f"{API_ROOT}/{self._account_id}/ai/run/{model}"
        if card.api_format == "multipart":
            # FLUX.2 models take multipart form data; reference images are input_image_0..N.
            fields: dict[str, Any] = {"prompt": request.prompt}
            fields["width"] = str(request.width)
            fields["height"] = str(request.height)
            if request.steps is not None:
                fields["steps"] = str(request.steps)
            if request.seed is not None:
                fields["seed"] = str(request.seed)
            files = {
                f"input_image_{i}": (f"input_{i}.png", data, "image/png")
                for i, data in enumerate(request.input_images)
            }
            response = await self._post(
                model,
                url,
                headers=self._headers(),
                data=fields,
                files=files or None,
                timeout=IMAGE_TIMEOUT,
            )
        else:
            body: dict[str, Any] = {"prompt": request.prompt}
            steps = request.steps
            if steps is not None and card.max_steps is not None:
                steps = min(steps, card.max_steps)
            if steps is not None:
                body["steps"] = steps
            if request.seed is not None:
                body["seed"] = request.seed
            response = await self._post(
                model, url, headers=self._headers(), json=body, timeout=IMAGE_TIMEOUT
            )
        return self._parse_image(model, response)

    def _parse_image(self, model: str, response: httpx.Response) -> ImageResponse:
        try:
            result = response.json()["result"]
            encoded = result["image"]
            return ImageResponse(images=[base64.b64decode(encoded)])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(self.name, model, "malformed image response", status=502) from exc
