"""Process-level sliding-window limiter for the safety reviewer.

Groq allows gpt-oss-safeguard-20b 3 requests per minute, and the `safety_review` chain has no
fallback. Sending a 4th request in a minute only buys a 429 from Groq. This limiter keeps the
process under the provider's limit: a review waits a few seconds for a free slot, and if none
frees up in time it gives up WITHOUT calling the model. The caller then fails the run closed
with a retryable failure kind (never an allow).

One instance per process (`review_limiter`), shared by every run, like the gateway. It is
per-process, not global: several worker processes each get their own budget (TODO(backend):
a shared limiter, e.g. in Redis, when the backend runs more than one worker).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from functools import lru_cache


class SlidingWindowLimiter:
    def __init__(
        self,
        max_calls: int,
        period_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_calls < 1 or period_s <= 0:
            raise ValueError("max_calls must be >= 1 and period_s > 0")
        self.max_calls = max_calls
        self.period_s = period_s
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        while self._calls and now - self._calls[0] >= self.period_s:
            self._calls.popleft()

    async def acquire(self, wait_s: float) -> bool:
        """Take a slot, waiting up to `wait_s` seconds for one. False: none freed in time (the
        caller must not make the call). The slot is counted when taken, before the call."""
        deadline = self._clock() + max(0.0, wait_s)
        while True:
            async with self._lock:
                now = self._clock()
                self._prune(now)
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return True
                frees_at = self._calls[0] + self.period_s
            if frees_at > deadline:
                return False
            await self._sleep(max(0.0, frees_at - self._clock()))


@lru_cache(maxsize=4)
def review_limiter(max_calls: int, period_s: float) -> SlidingWindowLimiter:
    """THE process-wide limiter for a given limit (one per configuration, normally one)."""
    return SlidingWindowLimiter(max_calls, period_s)


__all__ = ["SlidingWindowLimiter", "review_limiter"]
