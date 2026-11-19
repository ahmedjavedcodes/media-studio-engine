"""Test doubles. No test may make a real model call."""

from __future__ import annotations

import io
from collections.abc import Iterable
from typing import Any

from PIL import Image

from media_engine.gateway.errors import GatewayError
from media_engine.gateway.types import (
    CallResult,
    ImageRequest,
    ImageResponse,
    LLMRequest,
    LLMResponse,
)


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


def image_bytes(width: int = 64, height: int = 64, fmt: str = "PNG", color: str = "red") -> bytes:
    """Real, decodable image bytes (the Creator and Critic sniff the format)."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, fmt)
    return buf.getvalue()


class ScriptedImages:
    """An ImageBackend (what ModelGateway calls) that replays scripted results in order: bytes
    -> one image, an exception instance -> raised. Records every request and deadline."""

    def __init__(
        self,
        replies: Iterable[bytes | Exception],
        *,
        provider: str = "cloudflare",
        model: str = "@cf/black-forest-labs/flux-1-schnell",
        neurons_est: float | None = 57.6,
        cost_usd: float | None = None,
    ) -> None:
        self.replies = list(replies)
        self.provider = provider
        self.model = model
        self.neurons_est = neurons_est
        self.cost_usd = cost_usd
        self.requests: list[tuple[str, ImageRequest, float | None]] = []

    async def image(
        self, task: str, request: ImageRequest, *, deadline_s: float | None = None
    ) -> CallResult:
        self.requests.append((task, request, deadline_s))
        if not self.replies:
            raise AssertionError("ScriptedImages ran out of replies (unexpected extra image call)")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return CallResult(
            output=ImageResponse(images=[reply], cost_usd=self.cost_usd),
            task=task,
            provider=self.provider,
            model=self.model,
            latency_ms=1.0,
            is_fallback=False,
            neurons_est=self.neurons_est,
            cost_usd=self.cost_usd,
        )
