"""Cloudflare Workers AI (build mode). Jev and the vision stand-in.

UNVERIFIED: the request/response shape of `typesafe/jev` on Cloudflare is assumed to match
TypeSafe's own API (`state`, `questions` in; `answers` out). Check it on the first real call.
"""

from typing import Any

import httpx
import structlog

from app.contracts.ports import VisionRaw
from app.core.errors import ProviderCallFailed
from app.gateway.providers.http import parse_chat_completion, post_json

log = structlog.get_logger("provider")

CF_JEV_MODEL = "typesafe/jev"
CF_VISION_MODEL = "@cf/meta/llama-4-scout-17b-16e-instruct"


class CloudflareClient:
    def __init__(self, client: httpx.AsyncClient, *, account_id: str, api_token: str) -> None:
        self._client = client
        self._account_id = account_id
        self._token = api_token

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def token(self) -> str:
        return self._token

    async def run(self, model: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}/ai/run/{model}"
        data = await post_json(
            self._client,
            url,
            headers={"Authorization": f"Bearer {self._token}"},
            json=body,
            timeout=timeout,
        )
        if data.get("success") is False:
            raise ProviderCallFailed("cloudflare reported failure", retryable=False)
        result = data.get("result", data)
        if not isinstance(result, dict):
            raise ProviderCallFailed("cloudflare returned unexpected result", retryable=False)
        usage = result.get("usage")
        if isinstance(usage, dict):  # cost tracking: neurons count against the free daily limit
            log.info(
                "provider_usage", provider="cloudflare", model=model, neurons=usage.get("neurons")
            )
        return result


class CloudflareJev:
    model = CF_JEV_MODEL

    def __init__(self, cf: CloudflareClient) -> None:
        self._cf = cf

    async def ask(
        self, *, state: dict[str, Any], questions: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        result = await self._cf.run(self.model, {"state": state, "questions": questions}, timeout)
        answers = result.get("answers")
        if not isinstance(answers, dict):
            raise ProviderCallFailed("jev response has no answers", retryable=False)
        return answers


class CloudflareVisionStandIn:
    """Llama 4 Scout told to answer like Llama Guard. Not a safety guarantee (spec section 5)."""

    model = CF_VISION_MODEL

    def __init__(self, cf: CloudflareClient, *, system_prompt: str) -> None:
        self._cf = cf
        self._system = system_prompt

    async def classify(self, *, text: str, image_urls: list[str], timeout: float) -> VisionRaw:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"<user_text>\n{text}\n</user_text>"}
        ]
        content += [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
        result = await self._cf.run(
            self.model,
            {
                "messages": [
                    {"role": "system", "content": self._system},
                    {"role": "user", "content": content},
                ],
                "temperature": 0,
                "max_tokens": 10,
                "logprobs": True,
                "top_logprobs": 5,
            },
            timeout,
        )
        text_out, logprobs = parse_chat_completion(result)
        raw_usage = result.get("usage")
        usage = (
            {
                k: raw_usage[k]
                for k in ("prompt_tokens", "completion_tokens", "neurons")
                if isinstance(raw_usage.get(k), int | float)
            }
            if isinstance(raw_usage, dict)
            else None
        )
        return VisionRaw(text_out, logprobs, usage)
