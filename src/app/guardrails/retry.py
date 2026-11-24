"""One call with a deadline and at most one retry. Spec section 9.

Retry only on timeout, 429 and 5xx. Never on 400 or on an answer we cannot trust.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable

from app.core.errors import ProviderCallFailed
from app.guardrails.policy import MIN_RETRY_WINDOW_S


async def _attempt[T](fn: Callable[[float], Awaitable[T]], timeout: float) -> T:
    try:
        async with asyncio.timeout(timeout):
            return await fn(timeout)
    except TimeoutError as exc:
        raise ProviderCallFailed("check timed out", retryable=True) from exc


async def call_with_retry[T](
    fn: Callable[[float], Awaitable[T]],
    *,
    timeout: float,
    retry_delay: float | None,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """`fn` gets the attempt timeout. `deadline` is an absolute `clock()` time.

    `retry_delay=None` means no retry.
    """
    first = min(timeout, deadline - clock())
    if first <= 0:
        raise ProviderCallFailed("no time left for check", retryable=False)
    try:
        return await _attempt(fn, first)
    except ProviderCallFailed as exc:
        if not exc.retryable or retry_delay is None:
            raise
        # Skip the retry if less than MIN_RETRY_WINDOW_S would be left after the wait.
        if deadline - clock() - retry_delay < MIN_RETRY_WINDOW_S:
            raise
    await asyncio.sleep(retry_delay)
    remaining = deadline - clock()
    if remaining < MIN_RETRY_WINDOW_S:
        raise ProviderCallFailed("no time left for retry", retryable=False)
    return await _attempt(fn, min(timeout, remaining))
