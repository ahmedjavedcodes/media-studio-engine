"""When to retry, switch provider, or fail. Rules come from CLAUDE.md.

`decide()` is a pure function; `AvailabilityState` remembers exhaustion, cooldowns and
disabled models between calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

from media_engine.gateway.errors import ProviderError

CF_DAILY_EXHAUSTED = 3036
CF_TEMP_CAPACITY = 3040
CF_PAID_ONLY = 5035

MAX_ATTEMPTS = 2  # one call + one retry
RETRY_AFTER_MAX_SECONDS = 5.0  # "a few seconds"
CAPACITY_BACKOFF_SECONDS = 1.0
SERVER_ERROR_BACKOFF_SECONDS = 0.5


class Action(Enum):
    RETRY_SAME = "retry_same"
    NEXT = "next"
    RAISE = "raise"


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    delay: float = 0.0
    exhausted_until: datetime | None = None  # provider-wide
    cooldown_seconds: float | None = None  # this provider+model only
    disable_model: bool = False


def next_utc_midnight(now: datetime) -> datetime:
    now = now.astimezone(UTC)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def _retry_once_or_next(attempt: int, reason: str, delay: float) -> Decision:
    if attempt < MAX_ATTEMPTS:
        return Decision(Action.RETRY_SAME, reason, delay=delay)
    return Decision(Action.NEXT, f"{reason}_retry_failed")


def decide(error: ProviderError, attempt: int, now: datetime) -> Decision:
    """Decide what to do after `attempt` (1-based) consecutive failures on one model."""
    if error.provider == "cloudflare":
        if error.code == CF_DAILY_EXHAUSTED:
            return Decision(
                Action.NEXT,
                "cloudflare_daily_neurons_exhausted",
                exhausted_until=next_utc_midnight(now),
            )
        if error.code == CF_TEMP_CAPACITY:
            return _retry_once_or_next(
                attempt, "cloudflare_temporary_capacity", CAPACITY_BACKOFF_SECONDS
            )
        if error.code == CF_PAID_ONLY:
            return Decision(Action.NEXT, "cloudflare_model_needs_paid_plan", disable_model=True)

    if error.status == 400:
        return Decision(Action.RAISE, "bad_request")

    if error.status == 429:
        retry_after = error.retry_after
        if (
            retry_after is not None
            and retry_after <= RETRY_AFTER_MAX_SECONDS
            and attempt < MAX_ATTEMPTS
        ):
            return Decision(Action.RETRY_SAME, "rate_limited_short", delay=retry_after)
        return Decision(Action.NEXT, "rate_limited", cooldown_seconds=retry_after)

    if error.timeout or (error.status is not None and error.status >= 500):
        return _retry_once_or_next(attempt, "server_error_or_timeout", SERVER_ERROR_BACKOFF_SECONDS)

    # 401/403/404/422 etc.: config or model problem, not transient. Move on and log loudly.
    return Decision(Action.NEXT, f"unexpected_status_{error.status}")


class AvailabilityState:
    """In-process memory of what is temporarily or permanently unusable."""

    def __init__(self, now: Callable[[], datetime] | None = None) -> None:
        self._now = now or (lambda: datetime.now(UTC))
        self._exhausted_until: dict[str, datetime] = {}
        self._cooldown_until: dict[tuple[str, str], datetime] = {}
        self._disabled: set[tuple[str, str]] = set()

    def now(self) -> datetime:
        return self._now()

    def unavailable_reason(self, provider: str, model: str) -> str | None:
        now = self._now()
        until = self._exhausted_until.get(provider)
        if until is not None:
            if now < until:
                return f"provider_exhausted_until_{until.isoformat()}"
            del self._exhausted_until[provider]
        if (provider, model) in self._disabled:
            return "model_disabled_paid_plan_only"
        cool = self._cooldown_until.get((provider, model))
        if cool is not None:
            if now < cool:
                return f"model_cooling_down_until_{cool.isoformat()}"
            del self._cooldown_until[(provider, model)]
        return None

    def apply(self, provider: str, model: str, decision: Decision) -> None:
        if decision.exhausted_until is not None:
            self._exhausted_until[provider] = decision.exhausted_until
        if decision.cooldown_seconds:
            self._cooldown_until[(provider, model)] = self._now() + timedelta(
                seconds=decision.cooldown_seconds
            )
        if decision.disable_model:
            self._disabled.add((provider, model))
