"""M4: model gateway with a fake provider. Failover, the breaker tripping and recovering,
ContentRefused stopping the chain, and the deadline. No HTTP: FakeProvider replays a script."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from media_engine.contracts import Plan
from media_engine.core.config import Settings
from media_engine.gateway.breaker import BreakerBoard, BreakerPolicy, BreakerSettings, BreakerState
from media_engine.gateway.errors import (
    AllProvidersFailed,
    BadRequestError,
    ContentRefused,
    DeadlineExceeded,
    NoProviderAvailable,
)
from media_engine.gateway.fallback import AvailabilityState
from media_engine.gateway.llm_gateway import chain_resolver
from media_engine.gateway.model_gateway import ModelGateway
from media_engine.gateway.router import Gateway
from media_engine.gateway.types import ImageRequest, LLMRequest, ModelRef
from media_engine.services.create.loader import load_create_config

from .conftest import NOW
from .fakes import (
    HANG,
    Clock,
    FakeProvider,
    Gated,
    bad_request,
    down,
    rate_limited,
    refused,
)

THRESHOLD = 2
COOLDOWN = 30.0
REQ = ImageRequest(prompt="a cozy coffee shop")
CHAT = LLMRequest(messages=[{"role": "user", "content": "hi"}])


async def no_sleep(_: float) -> None:
    return None


@pytest.fixture
def clock() -> Clock:
    return Clock()


def build(alpha: FakeProvider, beta: FakeProvider, clock: Clock) -> tuple[Gateway, BreakerBoard]:
    board = BreakerBoard(
        BreakerPolicy(
            default=BreakerSettings(failure_threshold=THRESHOLD, cooldown_seconds=COOLDOWN)
        ),
        clock=clock,
    )
    gw = Gateway(
        Settings(_env_file=None),
        chains={
            # The real task name, so the image_creator policy in agents.yaml resolves to it.
            "image_create": [ModelRef("alpha", "img-a"), ModelRef("beta", "img-b")],
            "fast_llm": [ModelRef("alpha", "llm-a"), ModelRef("beta", "llm-b")],
        },
        providers={"alpha": alpha, "beta": beta},
        state=AvailabilityState(now=lambda: NOW),
        breakers=board,
        sleep=no_sleep,
    )
    return gw, board


def model_gateway(gw: Gateway) -> ModelGateway:
    return ModelGateway(gw, chain_resolver(load_create_config().agents))


async def generate(mg: ModelGateway, tier: Plan = "free", deadline_ms: int = 60_000):
    return await mg.generate(
        "a cozy coffee shop", width=768, height=512, tier=tier, seed=7, deadline_ms=deadline_ms
    )


# -- failover ---------------------------------------------------------------------------------


async def test_failover_is_invisible_to_the_image_creator(clock):
    alpha = FakeProvider("alpha", default=down("alpha"))
    beta = FakeProvider("beta", default=b"beta-img")
    gw, _ = build(alpha, beta, clock)

    result = await generate(model_gateway(gw))

    assert result.images == [b"beta-img"]
    assert (result.provider, result.model, result.chain) == ("beta", "img-b", "image_create")
    assert result.is_fallback is True
    assert result.seed == 7
    sent = beta.requests[0]
    assert isinstance(sent, ImageRequest)
    assert (sent.width, sent.height, sent.seed) == (768, 512, 7)
    assert alpha.calls == 2  # 5xx: the fallback rules' one retry, then next


async def test_chain_offset_starts_the_walk_at_the_next_model(clock):
    # The switch_model retry lever: skip the first chain entry, even though it is healthy.
    alpha = FakeProvider("alpha", default=b"alpha-img")
    beta = FakeProvider("beta", default=b"beta-img")
    gw, _ = build(alpha, beta, clock)

    result = await model_gateway(gw).generate(
        "a cozy coffee shop",
        width=768,
        height=512,
        tier="free",
        seed=7,
        deadline_ms=60_000,
        chain_offset=1,
    )

    assert (result.provider, result.model) == ("beta", "img-b")
    assert alpha.calls == 0


async def test_chain_offset_past_the_end_of_the_chain_is_no_provider(clock):
    alpha = FakeProvider("alpha", default=b"alpha-img")
    beta = FakeProvider("beta", default=b"beta-img")
    gw, _ = build(alpha, beta, clock)

    with pytest.raises(AllProvidersFailed):
        await model_gateway(gw).generate(
            "x", width=64, height=64, tier="free", seed=1, deadline_ms=1000, chain_offset=2
        )
    assert alpha.calls == beta.calls == 0


async def test_first_provider_success_is_not_a_fallback(clock):
    gw, board = build(FakeProvider("alpha"), FakeProvider("beta"), clock)

    result = await generate(model_gateway(gw))

    assert result.provider == "alpha" and result.is_fallback is False
    assert board.get("alpha").state is BreakerState.CLOSED


async def test_all_providers_down_raises_no_provider_available(clock):
    gw, _ = build(
        FakeProvider("alpha", default=down("alpha")),
        FakeProvider("beta", default=down("beta", 502)),
        clock,
    )

    with pytest.raises(NoProviderAvailable) as exc:
        await generate(model_gateway(gw))

    assert isinstance(exc.value, AllProvidersFailed)  # same class, spec name
    assert [(a.provider, a.outcome) for a in exc.value.attempts] == [
        ("alpha", "failed"),
        ("beta", "failed"),
    ]


# -- breaker trips and recovers -----------------------------------------------------------------


async def trip_alpha(gw: Gateway) -> None:
    for _ in range(THRESHOLD):
        with contextlib.suppress(AllProvidersFailed):  # when beta is scripted down too
            await gw.image("image_create", REQ)
    assert gw.breakers.get("alpha").state is BreakerState.OPEN


async def test_breaker_trips_after_threshold_and_stops_calling_the_provider(clock):
    alpha = FakeProvider("alpha", default=down("alpha"))
    beta = FakeProvider("beta")
    gw, board = build(alpha, beta, clock)

    await trip_alpha(gw)
    assert board.get("alpha").state is BreakerState.OPEN
    calls_when_tripped = alpha.calls

    result = await gw.image("image_create", REQ)

    assert result.provider == "beta"
    assert alpha.calls == calls_when_tripped  # no timeout paid: alpha is not called at all
    assert (result.attempts[0].outcome, result.attempts[0].reason) == ("skipped", "circuit_open")


async def test_breaker_recovers_after_cooldown_via_one_probe(clock):
    # Down for the two tripping calls (2 calls each, retry included), then healthy again.
    alpha = FakeProvider("alpha", [down("alpha")] * (2 * THRESHOLD), default=b"alpha-back")
    gw, board = build(alpha, FakeProvider("beta"), clock)
    await trip_alpha(gw)

    clock.advance(COOLDOWN - 1)
    assert (await gw.image("image_create", REQ)).provider == "beta"  # still cooling down

    clock.advance(1)
    result = await gw.image("image_create", REQ)  # this request is the probe

    assert result.provider == "alpha" and result.output.images == [b"alpha-back"]
    assert board.get("alpha").state is BreakerState.CLOSED
    assert (await gw.image("image_create", REQ)).provider == "alpha"


async def test_failed_probe_reopens_the_breaker(clock):
    alpha = FakeProvider("alpha", default=down("alpha"))
    gw, board = build(alpha, FakeProvider("beta"), clock)
    await trip_alpha(gw)
    clock.advance(COOLDOWN)

    before = alpha.calls
    assert (await gw.image("image_create", REQ)).provider == "beta"  # probe failed, fell over
    assert alpha.calls == before + 2  # the probe gets the normal retry, no more
    assert board.get("alpha").state is BreakerState.OPEN

    probe_calls = alpha.calls
    await gw.image("image_create", REQ)
    assert alpha.calls == probe_calls  # skipped again for a fresh cooldown


async def test_breaker_is_per_provider_and_shared_across_tasks(clock):
    alpha = FakeProvider("alpha", default=down("alpha"))
    beta = FakeProvider("beta", default=b"ok")
    gw, _ = build(alpha, beta, clock)
    await trip_alpha(gw)  # tripped by image calls...
    before = alpha.calls

    result = await gw.chat("fast_llm", CHAT)  # ...so chat skips alpha too

    assert result.provider == "beta"
    assert alpha.calls == before


@pytest.mark.parametrize("error", [rate_limited("alpha"), bad_request("alpha")])
async def test_rate_limits_and_bad_requests_do_not_trip_the_breaker(clock, error):
    alpha = FakeProvider("alpha", default=error)
    gw, board = build(alpha, FakeProvider("beta"), clock)

    for _ in range(THRESHOLD + 2):
        try:
            await gw.image("image_create", REQ)
        except BadRequestError:
            pass  # 400 still never falls back (our bug), as before

    assert board.get("alpha").state is BreakerState.CLOSED
    assert board.get("alpha").consecutive_failures == 0
    assert alpha.calls == THRESHOLD + 2  # called every time


async def test_a_success_between_failures_resets_the_count(clock):
    alpha = FakeProvider(
        "alpha", [down("alpha"), down("alpha"), b"ok", down("alpha"), down("alpha")]
    )
    gw, board = build(alpha, FakeProvider("beta"), clock)

    for _ in range(3):  # fail(2 calls), succeed, fail(2 calls)
        await gw.image("image_create", REQ)

    assert board.get("alpha").state is BreakerState.CLOSED
    assert board.get("alpha").consecutive_failures == 1


# -- ContentRefused -----------------------------------------------------------------------------


async def test_content_refused_propagates_and_stops_the_chain(clock):
    alpha = FakeProvider("alpha", default=refused("alpha"))
    beta = FakeProvider("beta")
    gw, _ = build(alpha, beta, clock)

    with pytest.raises(ContentRefused) as exc:
        await generate(model_gateway(gw))

    assert exc.value.provider == "alpha" and exc.value.task == "image_create"
    assert alpha.calls == 1  # no retry
    assert beta.calls == 0  # never routed round the filter


async def test_content_refusals_never_trip_the_breaker(clock):
    alpha = FakeProvider("alpha", [refused("alpha")] * (THRESHOLD + 3), default=b"fine")
    gw, board = build(alpha, FakeProvider("beta"), clock)

    for _ in range(THRESHOLD + 3):
        with pytest.raises(ContentRefused):
            await gw.image("image_create", REQ)

    assert board.get("alpha").state is BreakerState.CLOSED
    assert board.get("alpha").consecutive_failures == 0
    assert (await gw.image("image_create", REQ)).provider == "alpha"


async def test_refused_probe_counts_as_the_provider_being_up(clock):
    alpha = FakeProvider("alpha", [down("alpha")] * (2 * THRESHOLD), default=refused("alpha"))
    gw, board = build(alpha, FakeProvider("beta"), clock)
    await trip_alpha(gw)
    clock.advance(COOLDOWN)

    with pytest.raises(ContentRefused):
        await gw.image("image_create", REQ)

    assert board.get("alpha").state is BreakerState.CLOSED


# -- deadline (real time: asyncio.timeout runs on the loop clock) -------------------------------


async def test_deadline_cancels_a_hung_call_and_counts_as_a_failure():
    alpha = FakeProvider("alpha", default=HANG)
    beta = FakeProvider("beta")
    gw, board = build(alpha, beta, Clock())

    with pytest.raises(DeadlineExceeded) as exc:
        await generate(model_gateway(gw), deadline_ms=50)

    assert alpha.calls == 1 and alpha.cancelled == 1  # in-flight call aborted, not retried
    assert beta.calls == 0  # no time left for it
    assert exc.value.attempts[-1].reason == "deadline"
    assert board.get("alpha").consecutive_failures == 1


async def test_deadline_is_shared_across_the_chain():
    alpha = FakeProvider("alpha", default=down("alpha"))  # fails fast
    beta = FakeProvider("beta", default=HANG)
    gw, _ = build(alpha, beta, Clock())

    with pytest.raises(DeadlineExceeded) as exc:
        await gw.image("image_create", REQ, deadline_s=0.05)

    assert [(a.provider, a.outcome, a.reason) for a in exc.value.attempts] == [
        ("alpha", "failed", "server_error_or_timeout_retry_failed"),
        ("beta", "failed", "deadline"),
    ]
    assert beta.cancelled == 1


async def test_outside_cancellation_frees_the_half_open_probe(clock):
    alpha = FakeProvider("alpha", [down("alpha")] * (2 * THRESHOLD), default=HANG)
    gw, board = build(alpha, FakeProvider("beta"), clock)
    await trip_alpha(gw)
    clock.advance(COOLDOWN)

    task = asyncio.create_task(gw.image("image_create", REQ))  # the probe hangs
    await asyncio.sleep(0.01)
    assert alpha.calls == 2 * THRESHOLD + 1
    task.cancel()  # e.g. the run was cancelled
    with pytest.raises(asyncio.CancelledError):
        await task

    breaker = board.get("alpha")
    assert breaker.state is BreakerState.HALF_OPEN  # no verdict either way
    assert breaker.allow().probe  # the slot is free for the next probe


# -- neutral answers between failures -----------------------------------------------------------


async def test_interleaved_503_and_429_still_trips_the_breaker(clock):
    # 503 (2 calls: retry), then 429 (1 call, no retry-after: next), repeated. The 429s are
    # neutral, so they must not reset the count of 503 failures.
    pattern = [down("alpha"), down("alpha"), rate_limited("alpha")]
    alpha = FakeProvider("alpha", pattern * THRESHOLD)
    gw, board = build(alpha, FakeProvider("beta"), clock)

    for _ in range(2 * THRESHOLD - 1):  # fail, neutral, fail, ... ending on a failure
        await gw.image("image_create", REQ)

    assert board.get("alpha").state is BreakerState.OPEN


# -- concurrency in the half-open window --------------------------------------------------------


async def half_open(gw: Gateway, clock: Clock) -> None:
    await trip_alpha(gw)
    clock.advance(COOLDOWN)


async def test_concurrent_half_open_requests_make_exactly_one_provider_call(clock):
    gate = Gated(then=b"probe-ok")
    alpha = FakeProvider("alpha", [down("alpha")] * (2 * THRESHOLD) + [gate])
    beta = FakeProvider("beta", default=b"beta")
    gw, _ = build(alpha, beta, clock)
    await half_open(gw, clock)
    before = alpha.calls

    probe = asyncio.create_task(gw.image("image_create", REQ))
    await gate.started.wait()
    other = await gw.image("image_create", REQ)  # beta works, so no reason to wait

    assert other.provider == "beta"
    assert (other.attempts[0].reason) == "circuit_half_open"
    gate.release.set()
    assert (await probe).provider == "alpha"
    assert alpha.calls == before + 1


async def test_a_stale_call_cancelled_mid_probe_does_not_open_a_second_probe(clock):
    stale_gate = Gated()  # starts while closed, still running after the cooldown
    probe_gate = Gated(then=b"probe-ok")
    alpha = FakeProvider("alpha", [stale_gate, *[down("alpha")] * (2 * THRESHOLD), probe_gate])
    gw, board = build(alpha, FakeProvider("beta"), clock)

    stale = asyncio.create_task(gw.image("image_create", REQ))
    await stale_gate.started.wait()
    await half_open(gw, clock)
    probe = asyncio.create_task(gw.image("image_create", REQ))
    await probe_gate.started.wait()
    calls_with_probe_in_flight = alpha.calls

    stale.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stale

    assert board.get("alpha").probe_in_flight  # the stale call did not free it
    third = await gw.image("image_create", REQ)
    assert third.provider == "beta"  # skipped alpha: no second probe
    assert alpha.calls == calls_with_probe_in_flight
    probe_gate.release.set()
    assert (await probe).provider == "alpha"


# -- probe window with nothing else available ----------------------------------------------------
# On the free path Cloudflare is the only image provider. Decision: a request that finds its
# provider mid-probe and has nowhere else to go waits for the probe's verdict (within its
# deadline), then retries only the entries it skipped for the probe.


async def test_m8_shape_waiters_get_the_probe_verdict_instead_of_failing(clock):
    gate = Gated(then=b"back")
    alpha = FakeProvider("alpha", [down("alpha")] * (2 * THRESHOLD) + [gate], default=b"back")
    # Nothing else works. A 429 (neutral) rather than a 503, so beta's own breaker stays out
    # of the picture: beta is simply tried once per request and moved past.
    beta = FakeProvider("beta", default=rate_limited("beta"))
    gw, board = build(alpha, beta, clock)
    await half_open(gw, clock)
    beta_before = beta.calls

    branches = [asyncio.create_task(gw.image("image_create", REQ)) for _ in range(4)]
    await gate.started.wait()
    await asyncio.sleep(0.01)  # let the other three reach the probe wait
    assert not any(b.done() for b in branches)
    gate.release.set()
    results = await asyncio.gather(*branches)

    assert [r.provider for r in results] == ["alpha"] * 4
    assert board.get("alpha").state is BreakerState.CLOSED
    # The three waiters tried beta once each (429: no retry), and did not re-try it after.
    assert beta.calls - beta_before == 3
    waiter = next(r for r in results if any(a.reason == "circuit_half_open" for a in r.attempts))
    assert [(a.provider, a.outcome) for a in waiter.attempts] == [
        ("alpha", "skipped"),
        ("beta", "failed"),
        ("alpha", "success"),
    ]


async def test_waiters_fail_without_calling_when_the_probe_fails(clock):
    gate = Gated(then=down("alpha"))
    alpha = FakeProvider("alpha", [down("alpha")] * (2 * THRESHOLD) + [gate], default=down("alpha"))
    gw, board = build(alpha, FakeProvider("beta", default=down("beta")), clock)
    await half_open(gw, clock)

    probe = asyncio.create_task(gw.image("image_create", REQ))
    await gate.started.wait()
    waiter = asyncio.create_task(gw.image("image_create", REQ))
    await asyncio.sleep(0.01)
    calls = alpha.calls
    gate.release.set()

    for task in (probe, waiter):
        with pytest.raises(AllProvidersFailed):
            await task
    assert board.get("alpha").state is BreakerState.OPEN
    assert alpha.calls == calls + 1  # the probe's retry only; the waiter never called alpha


async def test_probe_wait_is_bounded_by_the_deadline(clock):
    gate = Gated()
    alpha = FakeProvider("alpha", [down("alpha")] * (2 * THRESHOLD) + [gate])
    gw, board = build(alpha, FakeProvider("beta", default=down("beta")), clock)
    await half_open(gw, clock)

    probe = asyncio.create_task(gw.image("image_create", REQ))
    await gate.started.wait()
    with pytest.raises(DeadlineExceeded):
        await gw.image("image_create", REQ, deadline_s=0.05)

    assert board.get("alpha").probe_in_flight  # the waiter's timeout did not touch the probe
    gate.release.set()
    await probe
