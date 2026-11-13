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
    ) -> None:
        super().__init__(f"{provider}/{model}: {message}")
        self.provider = provider
        self.model = model
        self.message = message
        self.status = status
        self.code = code
        self.retry_after = retry_after
        self.timeout = timeout


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


class AllProvidersFailed(GatewayError):
    def __init__(self, task: str, attempts: list[Attempt]) -> None:
        trail = ", ".join(f"{a.provider}/{a.model}={a.outcome}({a.reason})" for a in attempts)
        super().__init__(f"All providers failed for task '{task}': {trail or 'no usable entry'}")
        self.task = task
        self.attempts = attempts
