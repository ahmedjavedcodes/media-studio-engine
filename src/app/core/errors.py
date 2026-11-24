"""Engine errors. The frontend shows a clear message for each.

Errors that reach the HTTP layer carry `http_status` and a JSON `body()`.
"""

from typing import Any


class EngineError(Exception):
    code = "engine_error"
    http_status = 500

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.code)
        self.message = message or self.code

    def body(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message}

    def headers(self) -> dict[str, str]:
        return {}


class ContentRefused(EngineError):
    code = "content_refused"


class OutOfCredits(EngineError):
    code = "out_of_credits"


class ProviderDown(EngineError):
    code = "provider_down"


# Guardrail errors (see execution-steps/GUARDRAIL-AGENT-SPEC.md section 11)


class BadInput(EngineError):
    code = "bad_input"
    http_status = 400


class AccountRestricted(EngineError):
    code = "account_restricted"
    http_status = 403

    def __init__(self, message: str = "Your account is restricted.") -> None:
        super().__init__(message)


class Cooldown(EngineError):
    code = "cooldown"
    http_status = 429

    def __init__(self, retry_after: int) -> None:
        super().__init__("Too many blocked requests. Please wait before trying again.")
        self.retry_after = max(1, retry_after)

    def body(self) -> dict[str, Any]:
        return {**super().body(), "retry_after": self.retry_after}

    def headers(self) -> dict[str, str]:
        return {"Retry-After": str(self.retry_after)}


class GuardrailUnavailable(EngineError):
    code = "guardrail_unavailable"
    http_status = 503

    def __init__(self) -> None:
        super().__init__("Safety verification timed out. Please try again.")


class ContentBlocked(EngineError):
    """A guardrail reject. `payload` is the safe body from guardrails.messages."""

    code = "content_blocked"
    http_status = 422

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(str(payload.get("message", "")))
        self.payload = payload

    def body(self) -> dict[str, Any]:
        return self.payload


class ProviderCallFailed(ProviderDown):
    """One model call failed. `retryable` is True for timeouts, 429 and 5xx."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class InvalidModelOutput(EngineError):
    """A model answered, but not in a form we can trust. Guardrails fail closed on this."""

    code = "invalid_model_output"
