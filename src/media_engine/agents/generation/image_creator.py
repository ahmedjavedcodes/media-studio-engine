"""Image Creator. NO LLM: it formats a request, calls the model gateway, stores the image and
returns its id. It is never given a model, and never learns whether a failover happened.

inputs:
    prompt:       str   the enhanced prompt (required)
    width/height: int   requested size (required). Not every model honours it: the result's
                        width/height are read from the image that came back.
    seed:         int   (required). The supervisor picks seeds, so N branches get distinct ones.
    tier:         Plan  chain tier (optional, default auth.plan). The raise_tier lever passes
                        the next tier up.
    chain_offset: int   leading chain entries to skip (optional, default 0). The switch_model
                        lever starts at the next model.

output (envelope): GenerationResult fields (image_id, provider, model, seed, seed_honoured,
    cost_usd, neurons, width, height, mime). Never image bytes.

Cancellation is not this class's job: the supervisor's step guard checks it immediately before
the call, and an in-flight cancel aborts the task (a put in progress is finished and orphaned).

Returns DONE, or FAILED with
`output["failure"]` set (content_refused, no_provider, deadline, bad_request, gateway_error,
bad_image, storage, invalid_input). A provider refusal is FAILED, not REJECTED: the Image
Grand Agent decides what it means (Cloudflare's NSFW filter has false positives).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from media_engine.agents.base import Agent, AgentConfig, describe_validation_error, gateway_failure
from media_engine.contracts import (
    AuthContext,
    Envelope,
    Goal,
    Plan,
    Spend,
    Status,
    StoppingCondition,
)
from media_engine.core.images import NotAnImage, inspect_image
from media_engine.core.logging import get_logger
from media_engine.gateway.errors import ContentRefused, GatewayError
from media_engine.gateway.model_gateway import ModelGateway
from media_engine.ports.artifact_store import ArtifactStore
from media_engine.ports.event_sink import EventSink
from media_engine.schemas import GenerationResult

SUMMARY = "Created an image."

log = get_logger("agents.image_creator")


class CreatorInputs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    seed: int = Field(ge=0)
    tier: Plan | None = None
    chain_offset: int = Field(default=0, ge=0)


class ImageCreator(Agent):
    def __init__(
        self,
        config: AgentConfig,
        events: EventSink,
        *,
        images: ModelGateway,
        artifacts: ArtifactStore,
    ) -> None:
        if config.prompt_file is not None or config.output_model is not None:
            raise ValueError("The Image Creator makes no LLM call: give it no prompt or output")
        super().__init__(config, None, events)  # no LLM gateway, by design
        self._images = images
        self._artifacts = artifacts

    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope:
        try:
            args = CreatorInputs.model_validate(inputs)
        except ValidationError as err:
            return self.failed(
                "invalid inputs:\n" + describe_validation_error(err), Spend(), "invalid_input"
            )
        started = time.perf_counter()
        try:
            result = await self._images.generate(
                args.prompt,
                width=args.width,
                height=args.height,
                tier=args.tier or auth.plan,
                seed=args.seed,
                deadline_ms=stop.deadline_ms,
                chain_offset=args.chain_offset,
            )
        except GatewayError as err:
            spend = Spend(duration_ms=_elapsed_ms(started))
            if isinstance(err, ContentRefused):  # billed refusals, as the Agent base charges them
                spend = spend.plus(Spend(neurons=err.neurons or 0.0, cost_usd=err.cost_usd or 0.0))
            log.warning("creator.gateway_error", error=type(err).__name__)
            return self.failed(
                f"{type(err).__name__}: {str(err)[:300]}", spend, gateway_failure(err)
            )

        # From here the image exists and was paid for: every outcome reports its spend.
        spend = Spend(
            cost_usd=result.cost_usd, neurons=result.neurons, duration_ms=_elapsed_ms(started)
        )
        if len(result.images) > 1:
            log.info("creator.extra_images_ignored", count=len(result.images) - 1)
        data = result.images[0]
        try:
            info = inspect_image(data)
        except NotAnImage as err:
            log.warning("creator.bad_image", provider=result.provider, model=result.model)
            return self.failed(str(err), spend, "bad_image")
        put = asyncio.ensure_future(self._artifacts.put(data, info.mime))
        try:
            # Shielded: a cancel must not abandon a put half-way, or the image is stored but its
            # id is lost and it is never swept.
            image_id = await asyncio.shield(put)
        except asyncio.CancelledError:
            await self._orphan_after_cancel(put)
            raise
        except Exception as err:  # the store is a backend port: any failure is "storage"
            log.exception("creator.storage_failed")
            return self.failed(f"artifact store failed: {type(err).__name__}", spend, "storage")

        generated = GenerationResult(
            image_id=image_id,
            provider=result.provider,
            model=result.model,
            seed=result.seed,
            seed_honoured=result.seed_honoured,
            cost_usd=result.cost_usd,
            neurons=result.neurons,
            width=info.width,
            height=info.height,
            mime=info.mime,
        )
        if (info.width, info.height) != (args.width, args.height):
            log.info(
                "creator.size_differs",
                requested=f"{args.width}x{args.height}",
                got=f"{info.width}x{info.height}",
                model=result.model,
            )
        return self.envelope(Status.DONE, generated.model_dump(), SUMMARY, spend)

    async def _orphan_after_cancel(self, put: asyncio.Future[str]) -> None:
        """Cancelled while storing: let the put finish, then mark what it stored as orphaned
        ("cancelled") so the sweep deletes it. The caller re-raises the cancel."""
        try:
            image_id = await put
        except Exception:  # noqa: BLE001 - the put failed: nothing was stored, nothing to orphan
            return
        try:
            await self._artifacts.mark_orphaned(image_id, "cancelled")
        except Exception:
            log.exception("creator.mark_orphaned_failed")


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
