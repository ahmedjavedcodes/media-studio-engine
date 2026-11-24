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
    http_status = 402


class ProviderDown(EngineError):
    code = "provider_down"
    http_status = 503


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

    def __init__(self, details: list[dict[str, str]] | None = None) -> None:
        super().__init__("Safety verification timed out. Please try again.")
        # Internal only (logs, evals). Never part of body().
        self.details = details or []


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


# Image create errors (execution-steps/Image Create Agent/TXT2IMG-AGENT-SPEC.md)


class PlanLimit(EngineError):
    """The request asks for more than the plan allows. Nothing is charged."""

    code = "plan_limit"
    http_status = 400


class SpendCapReached(EngineError):
    """SPEND_CAP_USD is used up. Stop and report. Never switch to another paid provider."""

    code = "spend_cap_reached"
    http_status = 503

    def __init__(self) -> None:
        super().__init__(
            "Image generation is paused right now. You were not charged. Please try again later."
        )


class OutputBlocked(EngineError):
    """Every generated image failed the output safety check. Nothing is charged."""

    code = "output_blocked"
    http_status = 422

    def __init__(self) -> None:
        super().__init__(
            "We couldn't deliver these images because they didn't pass our safety check. "
            "You were not charged. Try changing your request."
        )
