"""Groq: free-tier reasoning LLMs (OpenAI-compatible). No images."""

from __future__ import annotations

from typing import Any

import httpx

from media_engine.core.config import Settings
from media_engine.gateway.providers.base import DEFAULT_TIMEOUT, Provider, parse_chat_completion
from media_engine.gateway.registry import card_or_default
from media_engine.gateway.types import LLMRequest, LLMResponse

BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(Provider):
    name = "groq"

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        super().__init__(client)
        self._key = settings.groq_api_key

    def is_configured(self) -> bool:
        return self._key is not None

    async def chat(self, model: str, request: LLMRequest) -> LLMResponse:
        assert self._key is not None
        card = card_or_default(self.name, model, "llm")
        payload: dict[str, Any] = {"model": model, "messages": request.messages}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_completion_tokens"] = request.max_tokens
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if request.tools:
            payload["tools"] = request.tools
        if request.reasoning_effort:
            payload["reasoning_effort"] = request.reasoning_effort
        if card.gpt_oss:
            # GPT-OSS: reasoning_format is unsupported; use include_reasoning.
            payload["include_reasoning"] = request.include_reasoning
        else:
            # Others: "parsed" splits reasoning out of content. Never "raw" with tools / JSON mode.
            payload["reasoning_format"] = "parsed" if request.include_reasoning else "hidden"
        response = await self._post(
            model,
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {self._key.get_secret_value()}"},
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        return parse_chat_completion(self.name, model, response.json())
