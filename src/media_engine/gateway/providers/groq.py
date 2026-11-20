"""Groq: free-tier reasoning LLMs (OpenAI-compatible). No images."""

from __future__ import annotations

from typing import Any

import httpx

from media_engine.core.config import Settings
from media_engine.core.logging import get_logger
from media_engine.gateway.errors import ProviderError
from media_engine.gateway.providers.base import (
    DEFAULT_TIMEOUT,
    Provider,
    inject_schema,
    parse_chat_completion,
)
from media_engine.gateway.registry import card_or_default
from media_engine.gateway.types import LLMRequest, LLMResponse

BASE_URL = "https://api.groq.com/openai/v1"
log = get_logger("gateway.groq")


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
        if schema is not None and (not card.json_schema or card.schema_in_prompt):
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
        try:
            response = await self._post(
                model,
                f"{BASE_URL}/chat/completions",
                headers=self._bearer(model, self._key, "GROQ_API_KEY"),
                json=payload,
                timeout=DEFAULT_TIMEOUT,
            )
        except ProviderError as err:
            # Groq's json_schema mode answers a reply it could not validate with HTTP 400
            # `json_validate_failed` (measured: qwen emitted `{~~ ...`). That is the MODEL's
            # bad output, not our bad request: hand the broken text back as an ordinary reply
            # so the agent's repair path runs, instead of failing the step with a 400.
            if err.status == 400 and err.code == "json_validate_failed":
                log.warning("groq.json_validate_failed", model=model)
                return LLMResponse(text=getattr(err, "failed_generation", "") or "", usage={})
            raise
        return parse_chat_completion(self.name, model, response.json())

    def _error_from_response(self, model: str, response: httpx.Response) -> ProviderError:
        error = super()._error_from_response(model, response)
        try:
            detail = response.json().get("error", {})
            text = detail.get("failed_generation") if isinstance(detail, dict) else None
        except ValueError:
            text = None
        error.failed_generation = text if isinstance(text, str) else ""  # type: ignore[attr-defined]
        return error
