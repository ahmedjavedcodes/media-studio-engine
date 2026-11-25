"""Groq: a reasoning model for the enhancer (LLMPort). PAID, per token. Only when GROQ_LLM=true.

Groq speaks the OpenAI chat format. gpt-oss models take `reasoning_effort` and answer in a JSON
schema. Reasoning tokens count against the completion limit, so we leave room for them.
"""

import asyncio
from typing import Any

import httpx
import structlog
from pydantic import BaseModel

from app.contracts.ports import VisionRaw
from app.core.errors import ProviderCallFailed
from app.gateway.providers.cloudflare_create import extract_text, parse_schema
from app.gateway.providers.http import post_json

log = structlog.get_logger("provider")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
REASONING_HEADROOM = 1024  # tokens the model may think with, on top of the answer
MAX_RETRY_WAIT_S = 60.0  # a 429 that asks us to wait longer than this is not retried
# USD per million tokens (input, output). Used only for the dev trace estimate.
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}


def estimate_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


async def post_with_retry(
    client: httpx.AsyncClient, key: str, body: dict[str, Any], timeout: float, model: str
) -> dict[str, Any]:
    """One retry for a 429 that says how long to wait (Retry-After). Groq's limit is tokens per
    minute, and the Art Director sends a long system prompt, so two calls close together can hit
    it. A request refused with 429 is not billed."""
    for attempt in range(2):
        try:
            return await post_json(
                client,
                GROQ_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=body,
                timeout=timeout,
            )
        except ProviderCallFailed as exc:
            wait = exc.retry_after
            if attempt == 1 or wait is None or wait > MAX_RETRY_WAIT_S or "429" not in str(exc):
                raise
            log.warning("groq_rate_limited_waiting", wait_s=wait, model=model)
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")  # the loop returns or raises


class GroqLLM:
    """A text model that answers in a Pydantic schema (LLMPort)."""

    provider = "groq"

    def __init__(self, client: httpx.AsyncClient, *, api_key: str, model: str) -> None:
        self._client = client
        self._key = api_key
        self.model = model

    async def complete_json[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        images: list[str] | None = None,
        max_tokens: int,
        temperature: float,
        timeout: float,
        reasoning_effort: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> T:
        content: str | list[dict[str, Any]] = user
        if images:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": u}} for u in images
            ]
        reasoning = "gpt-oss" in self.model
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": max_tokens + (REASONING_HEADROOM if reasoning else 0),
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            },
        }
        if reasoning:
            body["reasoning_effort"] = reasoning_effort or "low"
            body["include_reasoning"] = False  # the thinking is never shown or logged
        result = await self._post(body, timeout)
        self._add_usage(usage, result)
        return parse_schema(extract_text(result), schema)

    async def _post(self, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        return await post_with_retry(self._client, self._key, body, timeout, self.model)

    def _add_usage(self, sink: dict[str, Any] | None, result: dict[str, Any]) -> None:
        raw = result.get("usage")
        if sink is None or not isinstance(raw, dict):
            return
        pt = int(raw.get("prompt_tokens") or 0)
        ct = int(raw.get("completion_tokens") or 0)
        sink["prompt_tokens"] = sink.get("prompt_tokens", 0) + pt
        sink["completion_tokens"] = sink.get("completion_tokens", 0) + ct
        sink["cost_usd"] = sink.get("cost_usd", 0.0) + estimate_usd(self.model, pt, ct)
        log.info("provider_usage", provider="groq", model=self.model, tokens_in=pt, tokens_out=ct)


class GroqVision:
    """The per-image safety check on Groq (VisionPort): a vision model told to answer like Llama
    Guard. Groq gives no logprobs, so the verdict has no score. Used only as the fallback when
    Cloudflare fails. Not a safety guarantee, the same as the Cloudflare stand-in."""

    provider = "groq"

    def __init__(
        self, client: httpx.AsyncClient, *, api_key: str, model: str, system_prompt: str
    ) -> None:
        self._client = client
        self._key = api_key
        self._system = system_prompt
        self.model = model

    async def classify(self, *, text: str, image_urls: list[str], timeout: float) -> VisionRaw:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"<user_text>\n{text}\n</user_text>"}
        ]
        content += [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
        result = await post_with_retry(
            self._client,
            self._key,
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self._system},
                    {"role": "user", "content": content},
                ],
                "temperature": 0,
                "max_completion_tokens": 20,
            },
            timeout,
            self.model,
        )
        text_out = extract_text(result)
        raw = result.get("usage")
        raw = raw if isinstance(raw, dict) else {}
        usage = {k: raw[k] for k in ("prompt_tokens", "completion_tokens") if k in raw}
        return VisionRaw(str(text_out), None, usage or None)
