"""A scripted fake provider for gateway failover / breaker tests. No HTTP at all."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field

from media_engine.gateway.errors import ProviderError
from media_engine.gateway.providers.base import Provider
from media_engine.gateway.types import ImageRequest, ImageResponse, LLMRequest, LLMResponse

HANG = "hang"  # script entry: block until cancelled (a provider that never answers)


@dataclass
class Gated:
    """Script entry: block until the test sets `release`, then return / raise `then`."""

    then: bytes | str | Exception = b"img"
    release: asyncio.Event = field(default_factory=asyncio.Event)
    started: asyncio.Event = field(default_factory=asyncio.Event)


class FakeProvider(Provider):
    """Replays a script, one entry per call: bytes -> image, str -> chat text, an exception ->
    raised, HANG -> never returns. When the script runs out it repeats `default`."""

    def __init__(
        self,
        name: str,
        script: Iterable[bytes | str | Exception | Gated] = (),
        *,
        default: bytes | str | Exception = b"img",
    ) -> None:
        self.name = name
        self.script = list(script)
        self.default = default
        self.calls = 0
        self.cancelled = 0
        self.requests: list[ImageRequest | LLMRequest] = []

    def is_configured(self) -> bool:
        return True

    async def _next(self, request: ImageRequest | LLMRequest) -> bytes | str:
        self.calls += 1
        self.requests.append(request)
        entry = self.script.pop(0) if self.script else self.default
        if isinstance(entry, Gated):
            entry.started.set()
            try:
                await entry.release.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            entry = entry.then
        if entry == HANG:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        if isinstance(entry, Exception):
            raise entry
        return entry

    async def generate_image(self, model: str, request: ImageRequest) -> ImageResponse:
        entry = await self._next(request)
        return ImageResponse(images=[entry if isinstance(entry, bytes) else entry.encode()])

    async def chat(self, model: str, request: LLMRequest) -> LLMResponse:
        entry = await self._next(request)
        return LLMResponse(text=entry.decode() if isinstance(entry, bytes) else entry)


def down(name: str, status: int = 503) -> ProviderError:
    return ProviderError(name, "m", f"HTTP {status}", status=status)


def timeout(name: str) -> ProviderError:
    return ProviderError(name, "m", "timeout", timeout=True)


def refused(name: str) -> ProviderError:
    return ProviderError(name, "m", "nsfw", status=400, refused=True)


def rate_limited(name: str) -> ProviderError:
    return ProviderError(name, "m", "slow down", status=429)


def bad_request(name: str) -> ProviderError:
    return ProviderError(name, "m", "bad param", status=400)


class Clock:
    """Manual monotonic clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
