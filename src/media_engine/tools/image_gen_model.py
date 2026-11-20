"""image-gen-model: a TOOL, not a reasoning agent (config key `image_creator`). NO LLM: it
formats a request, calls the model gateway, stores the image and returns its id. It is never
given a model, and never learns whether a failover happened.

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
from media_engine.core.images import NotAnImage, downscale, inspect_image
from media_engine.core.logging import get_logger
from media_engine.gateway.errors import ContentRefused, GatewayError
from media_engine.gateway.model_gateway import ImageResult, ModelGateway
from media_engine.gateway.registry import card_or_default
from media_engine.gateway.types import attempt_infos
from media_engine.harness import trace as tracing
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
    # The Brief: the gateway compiles it into the prompt for each model it tries (§5.2).
    brief: dict[str, Any] | None = None
    # What this one call may cost in USD (the gateway checks its estimate before a paid call).
    max_cost_usd: float | None = Field(default=None, ge=0)


class CreatorParams(BaseModel):
    """`params` block for image_creator in agents.yaml."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thumbnail_sizes: list[int] = Field(default_factory=list)  # long side, px
    thumbnail_quality: int = Field(default=85, ge=1, le=95)


class ImageGenModel(Agent):
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
        self.params = CreatorParams.model_validate(config.model_policy.params)

    async def _thumbnails(self, image_id: str, data: bytes) -> list[int]:
        """Best effort: a thumbnail that fails is logged and skipped, never fails the image
        (the original is what was paid for). Returns the sizes stored."""
        made: list[int] = []
        for size in self.params.thumbnail_sizes:
            try:
                thumb, mime = await asyncio.to_thread(
                    downscale, data, size, self.params.thumbnail_quality
                )
                await self._artifacts.put_thumbnail(image_id, size, thumb, mime)
                made.append(size)
            except Exception:
                log.warning("creator.thumbnail_failed", size=size, exc_info=True)
        return made

    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope:
        try:
            args = CreatorInputs.model_validate(inputs)
        except ValidationError as err:
            return self.failed(
                "invalid inputs:\n" + describe_validation_error(err), Spend(), "invalid_input"
            )
        started_at, started = tracing.now(), time.perf_counter()
        tier = args.tier or auth.plan
        try:
            result = await self._images.generate(
                args.prompt,
                width=args.width,
                height=args.height,
                tier=tier,
                seed=args.seed,
                deadline_ms=stop.deadline_ms,
                chain_offset=args.chain_offset,
                brief=args.brief,
                max_cost_usd=args.max_cost_usd,
            )
        except asyncio.CancelledError:
            self._trace(args, tier, started_at, started, failure="aborted")
            raise
        except GatewayError as err:
            self._trace(args, tier, started_at, started, failure=gateway_failure(err), err=err)
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
            self._trace(
                args, tier, started_at, started, result=result, size=len(data), failure="bad_image"
            )
            return self.failed(str(err), spend, "bad_image")
        claimed = result.media_types[0] if result.media_types else None
        if claimed is not None and claimed != info.mime:
            # The stored type always comes from the bytes (never a hard-coded extension, never
            # the provider's word alone); a disagreement is worth knowing about.
            log.warning(
                "creator.media_type_mismatch",
                provider=result.provider,
                model=result.model,
                claimed=claimed,
                detected=info.mime,
            )
        stored: list[int] = []

        async def store() -> str:
            # The image and its thumbnails in ONE shielded task: a cancel can never lose the id
            # of something already stored (it is orphaned and swept instead).
            new_id = await self._artifacts.put(data, info.mime)
            stored.extend(await self._thumbnails(new_id, data))
            return new_id

        put = asyncio.ensure_future(store())
        try:
            # Shielded: a cancel must not abandon a put half-way, or the image is stored but its
            # id is lost and it is never swept.
            image_id = await asyncio.shield(put)
        except asyncio.CancelledError:
            await self._orphan_after_cancel(put)
            raise
        except Exception as err:  # the store is a backend port: any failure is "storage"
            log.exception("creator.storage_failed")
            self._trace(
                args,
                tier,
                started_at,
                started,
                result=result,
                info=info,
                size=len(data),
                failure="storage",
            )
            return self.failed(f"artifact store failed: {type(err).__name__}", spend, "storage")

        self._trace(args, tier, started_at, started, result=result, info=info, size=len(data))
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
            thumbnails=stored,
        )
        if (info.width, info.height) != (args.width, args.height):
            log.info(
                "creator.size_differs",
                requested=f"{args.width}x{args.height}",
                got=f"{info.width}x{info.height}",
                model=result.model,
            )
        return self.envelope(Status.DONE, generated.model_dump(), SUMMARY, spend)

    def _trace(
        self,
        args: CreatorInputs,
        tier: Plan,
        started_at: str,
        started: float,
        *,
        result: ImageResult | None = None,
        info: Any = None,
        size: int | None = None,
        failure: str | None = None,
        err: GatewayError | None = None,
    ) -> None:
        """Developer trace: one record for this image call. No model call, no image bytes."""
        scope = tracing.current()
        if scope is None:
            return
        provider = model = chain = None
        attempts: list[Any] = []
        neurons = cost = 0.0
        fallback = False
        seed_sent, honoured, steps = True, None, None
        if result is not None:
            provider, model, chain = result.provider, result.model, result.chain
            attempts, fallback = result.attempts, result.is_fallback
            neurons, cost, honoured, steps = (
                result.neurons,
                result.cost_usd,
                result.seed_honoured,
                result.steps,
            )
            seed_sent = card_or_default(result.provider, result.model, "image").accepts_seed
        elif err is not None:
            attempts = attempt_infos(getattr(err, "attempts", []) or [])
            neurons = float(getattr(err, "neurons", 0.0) or 0.0)
            cost = float(getattr(err, "cost_usd", 0.0) or 0.0)
        scope.trace.add_call(
            ts=started_at,
            step=scope.step,
            role=self.name,
            attempt=scope.attempt,
            branch=scope.branch,
            kind="image",
            chain=chain,
            provider=provider,
            model=model,
            was_fallback=fallback,
            attempts=attempts,
            neurons=neurons,
            cost_usd=cost,
            duration_ms=_elapsed_ms(started),
            failure=failure,
            request_text=tracing.redact(args.prompt) or "",
            reply_text=None if info is None else f"{info.mime}, {size} bytes " + tracing.OMITTED,
            image=tracing.ImageDetail(
                width_requested=args.width,
                height_requested=args.height,
                width=None if info is None else info.width,
                height=None if info is None else info.height,
                seed=args.seed,
                seed_sent=seed_sent,
                seed_honoured=honoured,
                mime=None if info is None else info.mime,
                bytes=size,
                steps=steps,
            ),
        )

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
