import asyncio
import time

import pytest

from app.core.errors import InvalidModelOutput, ProviderCallFailed
from app.guardrails.retry import call_with_retry


class Flaky:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.timeouts: list[float] = []

    async def __call__(self, timeout: float) -> str:
        self.timeouts.append(timeout)
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        if out == "slow":
            await asyncio.sleep(10)
        return str(out)


def run(fn: Flaky, *, timeout: float = 0.5, delay: float = 0.01, budget: float = 2.0) -> object:
    return call_with_retry(
        fn, timeout=timeout, retry_delay=delay, deadline=time.monotonic() + budget
    )


async def test_success_first_try() -> None:
    fn = Flaky("ok")
    assert await run(fn) == "ok"
    assert len(fn.timeouts) == 1


async def test_retries_once_on_retryable_error() -> None:
    fn = Flaky(ProviderCallFailed("429", retryable=True), "ok")
    assert await run(fn) == "ok"
    assert len(fn.timeouts) == 2


async def test_gives_up_after_one_retry() -> None:
    fn = Flaky(ProviderCallFailed("a", retryable=True), ProviderCallFailed("b", retryable=True))
    with pytest.raises(ProviderCallFailed):
        await run(fn)
    assert len(fn.timeouts) == 2


async def test_no_retry_on_400_or_bad_output() -> None:
    fn = Flaky(ProviderCallFailed("400", retryable=False), "ok")
    with pytest.raises(ProviderCallFailed):
        await run(fn)
    assert len(fn.timeouts) == 1
    fn2 = Flaky(InvalidModelOutput("junk"), "ok")
    with pytest.raises(InvalidModelOutput):
        await run(fn2)
    assert len(fn2.timeouts) == 1


async def test_timeout_is_retried() -> None:
    fn = Flaky("slow", "ok")
    assert await run(fn, timeout=0.05) == "ok"
    assert len(fn.timeouts) == 2


async def test_attempt_timeout_is_capped_by_deadline() -> None:
    fn = Flaky("ok")
    await run(fn, timeout=0.5, budget=0.2)
    assert fn.timeouts[0] <= 0.2


async def test_retry_skipped_when_under_100ms_left() -> None:
    fn = Flaky(ProviderCallFailed("x", retryable=True), "ok")
    with pytest.raises(ProviderCallFailed):
        await run(fn, delay=0.15, budget=0.2)
    assert len(fn.timeouts) == 1


async def test_deadline_already_passed() -> None:
    fn = Flaky("ok")
    with pytest.raises(ProviderCallFailed):
        await run(fn, budget=-1)
    assert fn.timeouts == []
