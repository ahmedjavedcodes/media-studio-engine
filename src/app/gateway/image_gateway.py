"""The one door to image models. Picks providers for a model and falls back between them.

Fallback happens only on 429, 5xx or timeout (retryable), and when the spend cap stops a paid
provider but a FREE provider can still serve the model. Never on a content refusal or any other
4xx, and never from one paid provider to another after the cap. A paid provider that fails 3 times
in a row is skipped for 60 seconds, so a 503 storm cannot use up the test budget.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import structlog

from app.contracts.ports import GeneratedImage, ImageOptions, ImageProviderPort
from app.core.errors import ProviderCallFailed, ProviderDown, SpendCapReached
from app.gateway.registry import ModelCard

log = structlog.get_logger("image_gateway")

BREAKER_FAILURES = 3
BREAKER_PAUSE_S = 60.0


@dataclass(frozen=True)
class GatewayResult:
    image: GeneratedImage
    provider: str  # the provider that made the image
    tried: list[str]  # every provider tried, in order. More than one means a fallback
    seed_sent: bool


class ImageGateway:
    def __init__(
        self,
        providers: dict[str, ImageProviderPort],
        enabled: list[str],
        paid: set[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._providers = providers
        self.enabled = [name for name in enabled if name in providers]
        self._paid = paid or set()
        self._clock = clock
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def _breaker_open(self, name: str) -> bool:
        return name in self._paid and self._open_until.get(name, 0.0) > self._clock()

    def _record(self, name: str, ok: bool) -> None:
        if name not in self._paid:  # a free provider is never switched off by the breaker
            return
        if ok:
            self._failures[name] = 0
            return
        self._failures[name] = self._failures.get(name, 0) + 1
        if self._failures[name] >= BREAKER_FAILURES:
            self._open_until[name] = self._clock() + BREAKER_PAUSE_S
            self._failures[name] = 0
            log.warning("provider_breaker_open", provider=name, pause_s=BREAKER_PAUSE_S)

    async def generate(
        self,
        card: ModelCard,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        aspect_ratio: str,
        seed: int | None,
        timeout: float = 90.0,
        options: ImageOptions | None = None,
    ) -> GatewayResult:
        """Returns the image, the provider that made it and the providers tried."""
        last: ProviderCallFailed | None = None
        tried: list[str] = []
        entries = card.providers_in(self.enabled)
        for position, entry in enumerate(entries):
            if self._breaker_open(entry.name):
                log.info("provider_skipped_breaker", provider=entry.name)
                continue
            provider = self._providers[entry.name]
            tried.append(entry.name)
            try:
                image = await provider.generate(
                    slug=entry.slug,
                    family=card.family,
                    prompt=prompt,
                    negative_prompt=negative_prompt if card.supports_negative_prompt else "",
                    width=width,
                    height=height,
                    aspect_ratio=aspect_ratio,
                    seed=seed if card.supports_seed else None,
                    timeout=timeout,
                    estimated_cost_usd=card.cost_for(entry, options.quality if options else None),
                    options=options,
                )
            except SpendCapReached:
                # Fall back only to a free provider. Never to another paid one.
                free_next = any(e.name not in self._paid for e in entries[position + 1 :])
                if not free_next:
                    raise
                log.warning("spend_cap_falling_back_to_free", provider=entry.name, model=card.id)
                continue
            except ProviderCallFailed as exc:
                self._record(entry.name, ok=False)
                if not exc.retryable:
                    raise
                log.warning("provider_failed_falling_back", provider=entry.name, model=card.id)
                last = exc
                continue
            self._record(entry.name, ok=True)
            return GatewayResult(image, entry.name, tried, seed_sent=card.supports_seed)
        if last is not None:
            raise last
        raise ProviderDown("no provider is available for this model")
