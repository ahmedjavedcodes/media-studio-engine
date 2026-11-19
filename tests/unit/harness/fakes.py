"""Test doubles. No test may make a real model call."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from media_engine.gateway.errors import GatewayError
from media_engine.gateway.types import CallResult, LLMRequest, LLMResponse


class ScriptedChat:
    """A ChatBackend that replays scripted replies in order. A reply is either a string (the
    model's text) or an exception instance to raise."""

    def __init__(
        self,
        replies: Iterable[str | Exception],
        *,
        prompt_tokens: int = 60,
        completion_tokens: int = 40,
        neurons: float | None = 2.5,
        provider: str = "cloudflare",
        model: str = "@cf/openai/gpt-oss-20b",
    ) -> None:
        self.replies = list(replies)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.neurons = neurons
        self.provider = provider
        self.model = model
        self.requests: list[tuple[str, LLMRequest]] = []

    async def chat(self, task: str, request: LLMRequest) -> CallResult:
        self.requests.append((task, request))
        if not self.replies:
            raise AssertionError("ScriptedChat ran out of replies (unexpected extra LLM call)")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return CallResult(
            output=LLMResponse(
                text=reply,
                usage={
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.prompt_tokens + self.completion_tokens,
                },
            ),
            task=task,
            provider=self.provider,
            model=self.model,
            latency_ms=1.0,
            is_fallback=False,
            neurons=self.neurons,
        )


class AllDown(GatewayError):
    pass


class RaisingSink:
    """An EventSink that raises on every call."""

    def __init__(self) -> None:
        self.attempts = 0

    def emit(self, run_id: str, event: dict[str, Any]) -> None:
        self.attempts += 1
        raise RuntimeError("event sink down")


def messages_of(backend: ScriptedChat, index: int) -> list[dict[str, Any]]:
    return backend.requests[index][1].messages
