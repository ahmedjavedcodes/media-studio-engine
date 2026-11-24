"""OpenRouter (test mode). Llama Guard 4 12B for images + cleaned text."""

from typing import Any

import httpx

from app.contracts.ports import VisionRaw
from app.gateway.providers.http import parse_chat_completion, post_json

LLAMA_GUARD_MODEL = "meta-llama/llama-guard-4-12b"


class OpenRouterLlamaGuard:
    model = LLAMA_GUARD_MODEL

    def __init__(self, client: httpx.AsyncClient, *, api_key: str, base_url: str) -> None:
        self._client = client
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/chat/completions"

    async def classify(self, *, text: str, image_urls: list[str], timeout: float) -> VisionRaw:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        content += [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
        body = await post_json(
            self._client,
            self._url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": 10,
                "logprobs": True,
                "top_logprobs": 5,
            },
            timeout=timeout,
        )
        text_out, logprobs = parse_chat_completion(body)
        return VisionRaw(text_out, logprobs)
