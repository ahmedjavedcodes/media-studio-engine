"""Generate one image: provider call, read the real file type, save it with a 256 px thumbnail."""

import hashlib
import time
from dataclasses import dataclass

from app.contracts.ports import ImageOptions
from app.core.tracing import Span
from app.gateway.compiler import CompiledPrompt
from app.gateway.image_gateway import ImageGateway
from app.gateway.registry import ModelCard
from app.local.output_store import OutputStore
from app.tools.imaging import make_thumbnail, sniff


def seed_for(run_id: str, index: int) -> int:
    """Seed = hash of (run_id, image_index). Different per image, the same on a replay."""
    digest = hashlib.sha256(f"{run_id}:{index}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31)


@dataclass(frozen=True)
class SavedImage:
    index: int
    file: str
    thumb: str
    mime: str
    width: int
    height: int
    seed: int
    provider: str
    cost_usd: float | None
    latency_ms: int
    fallback: bool
    seed_sent: bool


async def generate_one(
    gateway: ImageGateway,
    store: OutputStore,
    card: ModelCard,
    compiled: CompiledPrompt,
    *,
    run_id: str,
    index: int,
    timeout: float,
    span: Span | None = None,
    options: ImageOptions | None = None,
) -> SavedImage:
    seed = seed_for(run_id, index)
    started = time.monotonic()
    result = await gateway.generate(
        card,
        prompt=compiled.prompt,
        negative_prompt=compiled.negative_prompt,
        width=compiled.width,
        height=compiled.height,
        aspect_ratio=compiled.aspect_ratio,
        seed=seed,
        timeout=timeout,
        options=options,
    )
    image, provider = result.image, result.provider
    latency_ms = int((time.monotonic() - started) * 1000)
    info = sniff(image.data, image.media_type)  # the type comes from the bytes. Never assume PNG
    name = f"img_{index + 1}.{info.ext}"
    thumb = f"thumb_{index + 1}.jpg"
    store.write_bytes(run_id, name, image.data)
    store.write_bytes(run_id, thumb, make_thumbnail(image.data))
    saved = SavedImage(
        index=index,
        file=name,
        thumb=thumb,
        mime=info.mime,
        width=info.width,
        height=info.height,
        seed=seed,
        provider=provider,
        cost_usd=image.cost_usd,
        latency_ms=latency_ms,
        fallback=len(result.tried) > 1,
        seed_sent=result.seed_sent,
    )
    if span is not None:  # ids and numbers only. Never the image bytes
        entry = next((e for e in card.providers if e.name == provider), None)
        opts = options or ImageOptions()
        span.log(
            model=card.id,
            model_name=card.name,
            slug=entry.slug if entry else None,
            cost_usd=image.cost_usd,  # what the provider billed (None if it did not say)
            cost_estimate_usd=entry.cost if entry else card.cost_per_image,
            bytes=len(image.data),
            media_type=info.mime,
            options={
                "quality": opts.quality,
                "resolution": opts.resolution,
                "output_format": opts.output_format,
                "endpoint": opts.endpoint,
                "references": len(opts.references),
                "brand_colors": len(opts.brand_colors),
            },
            prompt_words=len(compiled.prompt.split()),
            image_id=f"img_{index + 1}",
            thumb_url=f"thumb_{index + 1}.jpg",
            seed_sent=result.seed_sent,
            seed=seed if result.seed_sent else None,
            seed_honoured=card.seed_honoured,
            size=f"{info.width}x{info.height}",
            requested_size=f"{compiled.width}x{compiled.height}",
            latency_ms=latency_ms,
            usage_cost=image.cost_usd,
            provider=provider,
            providers_tried=result.tried,
            provider_fallback=len(result.tried) > 1,
        )
    return saved
