"""Image generation for the Image Creator. Picks the gateway chain by plan tier (agents.yaml,
`image_creator.models`), then delegates to the fallback-chain Gateway, which handles failover,
the circuit breaker and the deadline. The caller never learns that a failover happened, apart
from `is_fallback` for logging.

No LLM. Stateless and shared across runs, like LLMGateway: per-run accounting travels in the
returned `ImageResult`.

Raises (all `GatewayError`): `ContentRefused` (a real decision for the Image Grand Agent),
`NoProviderAvailable`, `DeadlineExceeded`, `BadRequestError`.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Protocol

from pydantic import BaseModel, Field

from media_engine.contracts import Plan
from media_engine.gateway.registry import card_or_default
from media_engine.gateway.types import (
    AttemptInfo,
    CallResult,
    ImageRequest,
    ImageResponse,
    attempt_infos,
)

IMAGE_CREATOR = "image_creator"

PlanChainResolver = Callable[[str, Plan], str]  # (agent, plan) -> chain; same as LLMGateway's


class ImageBackend(Protocol):
    """What ModelGateway needs from `media_engine.gateway.router.Gateway`."""

    def image(
        self,
        task: str,
        request: ImageRequest,
        *,
        deadline_s: float | None = None,
        start_at: int = 0,
        plan: Plan | None = None,
    ) -> Awaitable[CallResult]: ...


class ImageResult(BaseModel):
    """Gateway-level result. Holds bytes, so it never goes into an Envelope: the Image Creator
    stores the bytes in the ArtifactStore and passes on only the image_id."""

    images: list[bytes] = Field(min_length=1)
    provider: str
    model: str
    chain: str
    seed: int  # the seed sent
    seed_honoured: bool | None  # from the model card; None = not verified for this model
    cost_usd: float
    neurons: float  # actual if reported, else the registry estimate (Cloudflare images)
    duration_ms: int
    is_fallback: bool
    attempts: list[AttemptInfo] = Field(default_factory=list)  # developer trace only
    steps: int | None = None  # steps sent (None: the provider's default)
    # What the PROVIDER said each image is (response field, else its own byte sniff); None =
    # not reported. The Image Creator still reads the type from the bytes before storing.
    media_types: list[str | None] = Field(default_factory=list)


class ModelGateway:
    def __init__(
        self,
        backend: ImageBackend,
        resolve_chain: PlanChainResolver,
        *,
        agent: str = IMAGE_CREATOR,
    ) -> None:
        self._backend = backend
        self._resolve_chain = resolve_chain
        self._agent = agent

    async def generate(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        tier: Plan,
        seed: int,
        deadline_ms: int,
        steps: int | None = None,
        chain_offset: int = 0,
    ) -> ImageResult:
        """`tier` is a plan tier, not necessarily the user's plan: the `raise_tier` retry
        lever passes the next tier up. `chain_offset` skips that many leading chain entries:
        the `switch_model` lever starts at the next model. `deadline_ms` comes from the
        step's StoppingCondition and bounds the whole chain walk."""
        chain = self._resolve_chain(self._agent, tier)
        request = ImageRequest(prompt=prompt, width=width, height=height, seed=seed, steps=steps)
        started = time.perf_counter()
        # The plan travels to the gateway, which enforces it (model tier, paid settings).
        result = await self._backend.image(
            chain, request, deadline_s=deadline_ms / 1000, start_at=chain_offset, plan=tier
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        output = result.output
        if not isinstance(output, ImageResponse):  # pragma: no cover - image always returns images
            raise TypeError(f"image returned {type(output).__name__}, expected ImageResponse")
        neurons = result.neurons if result.neurons is not None else (result.neurons_est or 0.0)
        return ImageResult(
            images=output.images,
            provider=result.provider,
            model=result.model,
            chain=chain,
            seed=seed,
            seed_honoured=card_or_default(result.provider, result.model, "image").honours_seed,
            cost_usd=result.cost_usd or 0.0,
            neurons=neurons,
            duration_ms=duration_ms,
            is_fallback=result.is_fallback,
            attempts=attempt_infos(result.attempts),
            steps=steps,
            media_types=list(output.media_types),
        )


__all__ = ["IMAGE_CREATOR", "ImageBackend", "ImageResult", "ModelGateway"]
