"""The one place where an internal error becomes text a user may see. Spec section 5.

Only errors we wrote ourselves, with a user-safe message, pass through. Provider and model errors
("provider returned 500", "the model answered in the wrong format") are logged and replaced by a
fixed message. Never send an exception message to the client by any other route.
"""

import structlog

from app.core.errors import (
    BadInput,
    ContentRefused,
    OutOfCredits,
    OutputBlocked,
    PlanLimit,
    ProviderCallFailed,
    ProviderDown,
    SpendCapReached,
)

log = structlog.get_logger("user_errors")

# Exact types whose messages we authored for users. A subclass is NOT trusted (ProviderCallFailed
# is a ProviderDown, but its message comes from a provider).
_TRUSTED: tuple[type[Exception], ...] = (
    PlanLimit,
    BadInput,
    OutOfCredits,
    ContentRefused,
    OutputBlocked,
    SpendCapReached,
    ProviderDown,
)

PROVIDER_DOWN_MESSAGE = (
    "We couldn't create your image right now. You were not charged. Please try again."
)
INTERNAL_MESSAGE = "Something went wrong. You were not charged."


def to_user_error(exc: BaseException) -> dict[str, str]:
    """{code, message} that is safe to send. The real message goes to the log only."""
    if type(exc) in _TRUSTED:
        return {"code": getattr(exc, "code", "internal_error"), "message": str(exc)}
    log.error("run_failed", error=type(exc).__name__, detail=str(exc)[:300])
    if isinstance(exc, ProviderCallFailed):
        return {"code": "provider_down", "message": PROVIDER_DOWN_MESSAGE}
    return {"code": "internal_error", "message": INTERNAL_MESSAGE}
