from __future__ import annotations

from media_engine.gateway.types import Attempt


class GatewayError(Exception):
    """Base class for gateway errors."""


class ProviderError(GatewayError):
    """A single provider call failed. Raised by providers; interpreted by `fallback.decide`."""

    def __init__(
        self,
        provider: str,
        model: str,
        message: str,
        *,
        status: int | None = None,
        code: int | str | None = None,
        retry_after: float | None = None,
        timeout: bool = False,
        refused: bool = False,
    ) -> None:
        super().__init__(f"{provider}/{model}: {message}")
        self.provider = provider
        self.model = model
        self.message = message
        self.status = status
        self.code = code
        self.retry_after = retry_after
        self.timeout = timeout
        self.refused = refused  # the provider declined the content (safety filter / moderation)


class BadRequestError(GatewayError):
    """HTTP 400: our request is wrong. Never falls back; fix the call or the registry."""

    def __init__(self, error: ProviderError) -> None:
        super().__init__(
            f"Bad request to {error.provider}/{error.model} "
            f"(status={error.status}, code={error.code}): {error.message}"
        )
        self.provider = error.provider
        self.model = error.model
        self.status = error.status
        self.code = error.code
        self.message = error.message


class ContentRefused(GatewayError):
    """A provider's safety filter declined the request. Stops the chain: another provider would
    only be a way round the filter. The caller (the Image Grand Agent) decides what happens next.
    Not a provider-health failure, so it never trips the circuit breaker."""

    def __init__(self, error: ProviderError, task: str) -> None:
        super().__init__(
            f"Content refused by {error.provider}/{error.model} for task '{task}' "
            f"(status={error.status}, code={error.code}): {error.message}"
        )
        self.provider = error.provider
        self.model = error.model
        self.status = error.status
        self.code = error.code
        self.message = error.message
        self.task = task


def _trail(attempts: list[Attempt]) -> str:
    return ", ".join(f"{a.provider}/{a.model}={a.outcome}({a.reason})" for a in attempts)


class AllProvidersFailed(GatewayError):
    def __init__(self, task: str, attempts: list[Attempt]) -> None:
        super().__init__(
            f"All providers failed for task '{task}': {_trail(attempts) or 'no usable entry'}"
        )
        self.task = task
        self.attempts = attempts


# Spec name (IMAGE-CREATE.md §9). Same class: existing callers catch AllProvidersFailed.
NoProviderAvailable = AllProvidersFailed


class DeadlineExceeded(GatewayError):
    """The caller's deadline ran out while walking the chain. The in-flight call was cancelled
    (httpx aborts the request), and the remaining entries were never tried."""

    def __init__(self, task: str, deadline_s: float, attempts: list[Attempt]) -> None:
        super().__init__(
            f"Deadline of {deadline_s:.1f}s exceeded for task '{task}': "
            f"{_trail(attempts) or 'no attempt'}"
        )
        self.task = task
        self.deadline_s = deadline_s
        self.attempts = attempts
