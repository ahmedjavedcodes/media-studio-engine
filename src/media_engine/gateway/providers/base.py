from __future__ import annotations

import json
from abc import ABC
from typing import Any

import httpx
from pydantic import SecretStr

from media_engine.gateway.errors import ProviderError
from media_engine.gateway.types import ImageRequest, ImageResponse, LLMRequest, LLMResponse

DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
IMAGE_TIMEOUT = httpx.Timeout(180.0, connect=10.0)


class Provider(ABC):
    """One provider. Raises `ProviderError` on any failure; never returns error values."""

    name: str

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    def is_configured(self) -> bool:
        raise NotImplementedError

    async def chat(self, model: str, request: LLMRequest) -> LLMResponse:
        raise ProviderError(self.name, model, "chat not supported by this provider", status=400)

    async def generate_image(self, model: str, request: ImageRequest) -> ImageResponse:
        raise ProviderError(self.name, model, "images not supported by this provider", status=400)

    # -- shared helpers ---------------------------------------------------------------------

    def _bearer(self, model: str, secret: SecretStr | None, env_name: str) -> dict[str, str]:
        """Auth header. The gateway skips unconfigured providers, so a missing key here is our
        bug: raised as a 400 (never falls back), not an assert that `python -O` would strip."""
        if secret is None:
            raise ProviderError(
                self.name, model, f"{env_name} is not set; provider not configured", status=400
            )
        return {"Authorization": f"Bearer {secret.get_secret_value()}"}

    async def _post(self, model: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.post(url, **kwargs)
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, model, f"timeout: {exc!r}", timeout=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                self.name, model, f"transport error: {exc!r}", timeout=True
            ) from exc
        if response.status_code >= 400:
            raise self._error_from_response(model, response)
        return response

    def _error_from_response(self, model: str, response: httpx.Response) -> ProviderError:
        code, message = parse_error_body(response)
        return ProviderError(
            self.name,
            model,
            message or f"HTTP {response.status_code}",
            status=response.status_code,
            code=code,
            retry_after=parse_retry_after(response),
        )


def parse_retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def parse_error_body(response: httpx.Response) -> tuple[int | str | None, str | None]:
    """Extract (code, message) from Cloudflare (`errors: [...]`) or OpenAI-style (`error: {...}`)."""
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return None, response.text[:300] or None
    if not isinstance(body, dict):
        return None, None
    entry: Any = None
    errors = body.get("errors")
    if isinstance(errors, list) and errors:
        entry = errors[0]
    elif isinstance(body.get("error"), dict):
        entry = body["error"]
    elif isinstance(body.get("error"), str):
        return None, body["error"]
    if not isinstance(entry, dict):
        return None, None
    code = entry.get("code")
    if isinstance(code, str) and code.isdigit():
        code = int(code)
    message = entry.get("message")
    return code, str(message) if message is not None else None


def parse_chat_completion(provider: str, model: str, data: dict[str, Any]) -> LLMResponse:
    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(
            provider, model, "malformed chat completion response", status=502
        ) from exc
    if choice.get("finish_reason") == "content_filter":
        # OpenAI-compatible signal that the provider's filter withheld the reply. The call was
        # still billed, so keep its usage for metering.
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        cost = usage.get("cost") if usage else None
        raise ProviderError(
            provider,
            model,
            "reply withheld by content filter",
            status=200,
            refused=True,
            usage=usage,
            cost_usd=float(cost) if isinstance(cost, int | float) else None,
        )
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    return LLMResponse(
        text=message.get("content") or "",
        reasoning=reasoning,
        usage=data.get("usage") or {},
    )
