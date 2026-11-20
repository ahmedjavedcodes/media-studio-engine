import asyncio
import time

import pytest

from tests.unit.waiting import wait_for_event


async def test_an_event_that_never_fires_fails_the_test_fast_instead_of_hanging():
    start = time.perf_counter()
    with pytest.raises(AssertionError, match="never fired"):
        await wait_for_event(asyncio.Event(), seconds=0.2)
    assert time.perf_counter() - start < 1
