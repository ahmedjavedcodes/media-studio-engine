"""Per-provider circuit breaker. Works alongside `fallback.py`, not instead of it.

`fallback.decide` reads what one error *means* (daily quota gone, model needs a paid plan, our
bug, ...) and `AvailabilityState` remembers those verdicts. The breaker answers a different
question: "has this provider stopped answering at all?" Free-tier-first makes failover the
normal path, so without it every request would pay a full timeout before falling back.

States: closed (calls pass) -> open after `failure_threshold` consecutive FAILURE verdicts
(calls skip) -> half-open once `cooldown_seconds` pass (exactly one probe call passes) ->
closed when the probe gets any answer, open again when the probe fails.

Probe ownership is per call: `allow()` returns a `Permit`, and only the call holding the probe
permit can free the probe slot or decide the half-open outcome. Reports from calls that started
before the breaker opened are stale evidence and do not move an open / half-open breaker.

Recovery is probed lazily: the first request after the cooldown is the probe. The harness is a
library with no background loop, so there is nothing to run a timer-driven health check.
Requests that arrive while a probe is in flight can `wait_for_probe()` instead of failing.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import yaml
from pydantic import BaseModel, ConfigDict, Field

from media_engine.core.logging import get_logger
from media_engine.gateway.errors import ProviderError

PROVIDERS_PATH = Path(__file__).with_name("providers.yaml")

log = get_logger("gateway.breaker")


class BreakerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    failure_threshold: int = Field(gt=0)
    cooldown_seconds: float = Field(gt=0)


class BreakerPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default: BreakerSettings
    providers: dict[str, BreakerSettings] = Field(default_factory=dict)

    def for_provider(self, provider: str) -> BreakerSettings:
        return self.providers.get(provider, self.default)


def load_breaker_policy(path: Path = PROVIDERS_PATH) -> BreakerPolicy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return BreakerPolicy.model_validate(raw.get("breaker") or {})


class Verdict(StrEnum):
    SUCCESS = "success"  # the call returned a result
    FAILURE = "failure"  # the provider did not answer: timeout, transport error, 5xx
    NEUTRAL = "neutral"  # the provider answered with an error: 429, 4xx, quota, refusal


def verdict_for(error: ProviderError) -> Verdict:
    """FAILURE only when the provider failed to answer. Any other error is an answer, so it says
    nothing against the provider's health (NEUTRAL), but it is no success either."""
    if error.refused:
        return Verdict.NEUTRAL
    if error.timeout or (error.status is not None and error.status >= 500):
        return Verdict.FAILURE
    return Verdict.NEUTRAL


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class Permit(NamedTuple):
    allowed: bool
    probe: bool  # this call holds the half-open probe slot


DENIED = Permit(allowed=False, probe=False)
PASS = Permit(allowed=True, probe=False)
PROBE = Permit(allowed=True, probe=True)


class CircuitBreaker:
    def __init__(
        self,
        provider: str,
        settings: BreakerSettings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self._clock = clock
        self.state = BreakerState.CLOSED
        self.consecutive_failures = 0
        self._opened_at: float | None = None
        self._probe_done: asyncio.Event | None = None  # set while a probe is in flight

    @property
    def probe_in_flight(self) -> bool:
        return self._probe_done is not None

    def allow(self) -> Permit:
        """May a call go to this provider now? A `PROBE` permit claims the single half-open
        slot, so call this last, right before the call, and always hand the permit back through
        `report()` or `release()`."""
        if self.state is BreakerState.CLOSED:
            return PASS
        if self.state is BreakerState.OPEN:
            if self._opened_at is None:
                raise RuntimeError(f"breaker for {self.provider} is open without an opened_at")
            if self._clock() - self._opened_at < self.settings.cooldown_seconds:
                return DENIED
            self._transition(BreakerState.HALF_OPEN)
        if self._probe_done is not None:
            return DENIED  # one probe at a time
        self._probe_done = asyncio.Event()
        return PROBE

    def report(self, verdict: Verdict, permit: Permit) -> None:
        if permit.probe:
            self._end_probe()
            if verdict is Verdict.FAILURE:
                self.consecutive_failures += 1
                self._open()
            else:  # any answer, even an error one, proves the provider is back
                self.consecutive_failures = 0
                self._close()
            return
        if self.state is not BreakerState.CLOSED:
            return  # stale: this call started before the breaker opened; the probe decides
        if verdict is Verdict.SUCCESS:
            self.consecutive_failures = 0
        elif verdict is Verdict.FAILURE:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.settings.failure_threshold:
                self._open()
        # NEUTRAL: the count is left as it is

    def release(self, permit: Permit) -> None:
        """The call ended without telling us anything about the provider (cancelled by the
        caller, or never sent). If it held the probe slot, free it for the next request."""
        if permit.probe:
            self._end_probe()

    async def wait_for_probe(self) -> None:
        """Return once the in-flight probe (if any) has reported back."""
        if self._probe_done is not None:
            await self._probe_done.wait()

    @property
    def skip_reason(self) -> str:
        return f"circuit_{self.state.value}"

    def _end_probe(self) -> None:
        if self._probe_done is not None:
            self._probe_done.set()
            self._probe_done = None

    def _open(self) -> None:
        self._opened_at = self._clock()
        self._transition(BreakerState.OPEN)

    def _close(self) -> None:
        self._opened_at = None
        if self.state is not BreakerState.CLOSED:
            self._transition(BreakerState.CLOSED)

    def _transition(self, new: BreakerState) -> None:
        old, self.state = self.state, new
        if old is new:
            return
        level = "warning" if new is BreakerState.OPEN else "info"
        getattr(log, level)(
            "breaker.transition",
            provider=self.provider,
            from_state=old.value,
            to_state=new.value,
            consecutive_failures=self.consecutive_failures,
            cooldown_seconds=self.settings.cooldown_seconds,
        )


class BreakerBoard:
    """One breaker per provider, created on first use. In-process, like `AvailabilityState`."""

    def __init__(
        self,
        policy: BreakerPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy or load_breaker_policy()
        self._clock = clock
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, provider: str) -> CircuitBreaker:
        breaker = self._breakers.get(provider)
        if breaker is None:
            breaker = CircuitBreaker(provider, self.policy.for_provider(provider), self._clock)
            self._breakers[provider] = breaker
        return breaker


__all__ = [
    "BreakerBoard",
    "BreakerPolicy",
    "BreakerSettings",
    "BreakerState",
    "CircuitBreaker",
    "Permit",
    "Verdict",
    "load_breaker_policy",
    "verdict_for",
]
