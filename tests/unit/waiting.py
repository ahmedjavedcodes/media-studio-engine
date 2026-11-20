"""Bounded waits for tests.

A test that waits on an `asyncio.Event` which never fires (a broken fake, a crashed branch)
hangs the whole suite with no signal. `wait_for_event` turns that into a failed test."""

from __future__ import annotations

import asyncio

DEFAULT_SECONDS = 5.0


async def wait_for_event(event: asyncio.Event, seconds: float = DEFAULT_SECONDS) -> None:
    try:
        await asyncio.wait_for(event.wait(), seconds)
    except TimeoutError:
        raise AssertionError(f"the awaited event never fired within {seconds}s") from None
