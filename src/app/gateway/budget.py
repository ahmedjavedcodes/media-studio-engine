"""Spend guard for paid calls (test mode). Stops at SPEND_CAP_USD. Spec section 8, CLAUDE.md.

Parallel variations all ask before any of them is billed, so the guard RESERVES the estimated
cost under a lock before each call. Then it settles with the billed cost, or keeps the estimate
when the provider reports none, so the cap cannot be passed by the calls already in flight.
"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import structlog

from app.core.errors import ProviderCallFailed, SpendCapReached

log = structlog.get_logger("budget")


class SpendGuard:
    def __init__(self, cap_usd: float, ledger: Path | None = None) -> None:
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self.reserved_usd = 0.0
        self._ledger = ledger
        self._lock = asyncio.Lock()

    async def reserve(self, estimate_usd: float) -> float:
        """Hold `estimate_usd` for one call. Raises SpendCapReached if it would pass the cap."""
        estimate = max(estimate_usd, 0.0)
        async with self._lock:
            if self.spent_usd + self.reserved_usd + estimate > self.cap_usd:
                log.error("spend_cap_reached", spent=self.spent_usd, cap=self.cap_usd)
                raise SpendCapReached()
            self.reserved_usd += estimate
        return estimate

    async def settle(self, reservation: float, billed_usd: float | None, *, model: str) -> None:
        """The call finished. Charge the billed cost, or the estimate if none was reported."""
        cost = billed_usd if billed_usd is not None else reservation
        async with self._lock:
            self.reserved_usd -= reservation
            self.spent_usd += cost
        log.info("paid_call", model=model, cost_usd=cost, total_usd=round(self.spent_usd, 5))
        if self._ledger is not None:
            self._ledger.parent.mkdir(parents=True, exist_ok=True)
            line = {"t": int(time.time()), "model": model, "cost_usd": cost}
            with self._ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")

    async def run[T](
        self,
        estimate_usd: float,
        *,
        model: str,
        call: Callable[[], Awaitable[T]],
        billed: Callable[[T], float | None] = lambda _: None,
    ) -> T:
        """One paid call, with the cap enforced. Reserve, call, then settle with the billed cost
        (or the estimate if none is reported). Only a request the provider surely refused (a 4xx)
        releases its reservation. A timeout, 429, 5xx, transport error or cancel may still have
        been billed, so those keep the estimate."""
        reservation = await self.reserve(estimate_usd)
        try:
            result = await call()
        except ProviderCallFailed as exc:
            if exc.retryable:
                await self.settle(reservation, None, model=model)
            else:
                await self.release(reservation)
            raise
        except BaseException:  # cancelled mid-request: it may have been sent
            await self.settle(reservation, None, model=model)
            raise
        await self.settle(reservation, billed(result), model=model)
        return result

    async def release(self, reservation: float) -> None:
        """The call failed before it could be billed."""
        async with self._lock:
            self.reserved_usd -= reservation
