"""Groq: free-tier reasoning LLMs (OpenAI-compatible). No images."""

from __future__ import annotations

from typing import Any

import httpx

from media_engine.core.config import Settings
from media_engine.gateway.providers.base import (
    DEFAULT_TIMEOUT,
    Provider,
    inject_schema,
    parse_chat_completion,
)
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
        card = card_or_default(self.name, model, "llm")
        messages = request.messages
        schema = request.response_schema
        if schema is not None and not card.json_schema:
            messages = inject_schema(messages, schema)
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_completion_tokens"] = request.max_tokens
        if schema is not None and card.json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": schema},
            }
        elif request.json_mode:
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
            headers=self._bearer(model, self._key, "GROQ_API_KEY"),
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        return parse_chat_completion(self.name, model, response.json())
