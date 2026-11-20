"""Test doubles. No test may make a real model call."""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from typing import Any, Self

from PIL import Image

from media_engine.gateway.errors import GatewayError
from media_engine.gateway.llm_gateway import ChainResolver, LLMGateway, LLMResult
from media_engine.gateway.types import (
    CallResult,
    ImageRequest,
    ImageResponse,
    LLMRequest,
    LLMResponse,
)


class Review(str):
    """A scripted verdict on ONE image (what the per-image Critic used to reply). The Critic now
    reviews every image in one call, so ScriptedChat folds the next N Reviews into one batch
    reply, one per image sent: `[enhanced(), verdict(9), verdict(7)]` still reads "image 1 got
    9, image 2 got 7"."""

    fields: dict[str, Any]

    def __new__(cls, fields: dict[str, Any]) -> Self:
        obj = super().__new__(cls, json.dumps(fields))
        obj.fields = fields
        return obj


def verdict(score: int = 9, **overrides: Any) -> Review:
    return Review(
        {
            "quality_score": score,
            "safety_verdict": "allow",
            "safety_reason": None,
            "critique": "Matches the request; the lamps are slightly blown out.",
            **overrides,
        }
    )


def batch_reply(reviews: list[Review]) -> str:
    """One Critic reply for `reviews` (image 1, 2, ...). Ranks follow the scores."""
    allowed = sorted(
        (i for i, r in enumerate(reviews) if r.fields["safety_verdict"] != "block"),
        key=lambda i: (-reviews[i].fields["quality_score"], i),
    )
    images = []
    for i, review in enumerate(reviews):
        f = review.fields
        score = f["quality_score"]
        images.append(
            {
                "index": i + 1,
                "prompt_adherence": score,
                "craft": score,
                "visual_interest": score,
                "technical": score,
                "safety_verdict": f["safety_verdict"],
                "safety_reason": f["safety_reason"],
                "rank": allowed.index(i) + 1 if i in allowed else None,
                "critique": f["critique"],
            }
        )
    return json.dumps({"images": images})


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
        self.agents: list[str] = []  # filled by AgentRecordingLLMGateway: who made each call

    async def chat(self, task: str, request: LLMRequest) -> CallResult:
        self.requests.append((task, request))
        if not self.replies:
            raise AssertionError("ScriptedChat ran out of replies (unexpected extra LLM call)")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, Review):
            count = _image_count(request)
            reviews = [reply]
            while len(reviews) < count and self.replies and isinstance(self.replies[0], Review):
                reviews.append(self.replies.pop(0))
            reply = batch_reply(reviews)
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


def _image_count(request: LLMRequest) -> int:
    content = request.messages[-1]["content"]
    if not isinstance(content, list):
        return 1
    return sum(1 for part in content if part.get("type") == "image_url") or 1


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
        self.start_at: list[int] = []  # the chain offset of each call
        self.plans: list[str | None] = []  # the plan each call was made for (gateway-enforced)

    async def image(
        self,
        task: str,
        request: ImageRequest,
        *,
        deadline_s: float | None = None,
        start_at: int = 0,
        plan: str | None = None,
        max_cost_usd: float | None = None,
    ) -> CallResult:
        self.requests.append((task, request, deadline_s))
        self.start_at.append(start_at)
        self.plans.append(plan)
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


class AgentRecordingLLMGateway(LLMGateway):
    """An LLMGateway that records, on the ScriptedChat behind it, which AGENT made each call:
    by the name `complete` is called with, never by prompt text (prompts get rewritten)."""

    def __init__(self, backend: ScriptedChat, resolve_chain: ChainResolver) -> None:
        super().__init__(backend, resolve_chain)
        self._recorder = backend

    async def complete(self, agent: str, *args: Any, **kwargs: Any) -> LLMResult:
        self._recorder.agents.append(agent)
        return await super().complete(agent, *args, **kwargs)
