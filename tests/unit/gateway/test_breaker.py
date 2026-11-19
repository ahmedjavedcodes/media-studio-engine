"""Circuit breaker state machine, on its own (no gateway).

Note: `allow()` returns a Permit (a NamedTuple), which is always truthy. Assert on `.allowed`
and `.probe`, never on the permit itself."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from media_engine.gateway.breaker import (
    BreakerBoard,
    BreakerPolicy,
    BreakerSettings,
    BreakerState,
    CircuitBreaker,
    Permit,
    Verdict,
    load_breaker_policy,
    verdict_for,
)
from media_engine.gateway.errors import ProviderError

from .fakes import Clock

SETTINGS = BreakerSettings(failure_threshold=3, cooldown_seconds=30)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def breaker(clock: Clock) -> CircuitBreaker:
    return CircuitBreaker("cloudflare", SETTINGS, clock)


def fail_once(breaker: CircuitBreaker) -> None:
    permit = breaker.allow()
    assert permit.allowed
    breaker.report(Verdict.FAILURE, permit)


def trip(breaker: CircuitBreaker) -> None:
    for _ in range(SETTINGS.failure_threshold):
        fail_once(breaker)
    assert breaker.state is BreakerState.OPEN


def probe(breaker: CircuitBreaker, clock: Clock) -> Permit:
    trip(breaker)
    clock.advance(SETTINGS.cooldown_seconds)
    permit = breaker.allow()
    assert permit.allowed and permit.probe
    return permit


# -- closed -------------------------------------------------------------------------------------


def test_closed_passes_without_claiming_a_probe_and_trips_at_threshold(breaker):
    for _ in range(SETTINGS.failure_threshold - 1):
        fail_once(breaker)
    assert breaker.state is BreakerState.CLOSED
    assert breaker.allow() == Permit(allowed=True, probe=False)

    fail_once(breaker)
    assert breaker.state is BreakerState.OPEN
    assert not breaker.allow().allowed
    assert breaker.skip_reason == "circuit_open"


def test_success_resets_the_consecutive_count(breaker):
    for _ in range(SETTINGS.failure_threshold - 1):
        fail_once(breaker)
    breaker.report(Verdict.SUCCESS, breaker.allow())
    fail_once(breaker)
    assert breaker.state is BreakerState.CLOSED
    assert breaker.consecutive_failures == 1


def test_neutral_neither_counts_nor_resets(breaker):
    fail_once(breaker)
    fail_once(breaker)
    breaker.report(Verdict.NEUTRAL, breaker.allow())  # e.g. a 429 between the failures
    assert breaker.consecutive_failures == 2
    fail_once(breaker)
    assert breaker.state is BreakerState.OPEN


# -- half-open probe ----------------------------------------------------------------------------


def test_open_skips_until_cooldown_then_lets_exactly_one_probe_through(breaker, clock):
    trip(breaker)
    clock.advance(SETTINGS.cooldown_seconds - 0.1)
    assert not breaker.allow().allowed

    clock.advance(0.1)
    first = breaker.allow()
    second = breaker.allow()
    assert first == Permit(allowed=True, probe=True)
    assert second == Permit(allowed=False, probe=False)
    assert breaker.state is BreakerState.HALF_OPEN and breaker.probe_in_flight
    assert breaker.skip_reason == "circuit_half_open"


@pytest.mark.parametrize("verdict", [Verdict.SUCCESS, Verdict.NEUTRAL])
def test_any_answer_to_the_probe_closes(breaker, clock, verdict):
    permit = probe(breaker, clock)
    breaker.report(verdict, permit)

    assert breaker.state is BreakerState.CLOSED
    assert breaker.consecutive_failures == 0
    assert not breaker.probe_in_flight


def test_probe_failure_reopens_for_a_fresh_cooldown(breaker, clock):
    permit = probe(breaker, clock)
    breaker.report(Verdict.FAILURE, permit)

    assert breaker.state is BreakerState.OPEN
    clock.advance(SETTINGS.cooldown_seconds - 1)
    assert not breaker.allow().allowed  # cooldown restarted at the probe failure
    clock.advance(1)
    assert breaker.allow().probe


def test_probe_release_frees_the_slot_without_a_verdict(breaker, clock):
    permit = probe(breaker, clock)
    breaker.release(permit)  # e.g. the run was cancelled mid-probe

    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.allow().probe  # the next request may probe


# -- stale calls (started while closed) never touch someone else's probe ------------------------


def test_stale_release_does_not_free_another_calls_probe(breaker, clock):
    stale = breaker.allow()  # in flight while closed...
    assert stale == Permit(allowed=True, probe=False)
    probe(breaker, clock)  # ...the breaker trips, cools down, and someone else probes

    breaker.release(stale)  # the stale call is cancelled

    assert breaker.probe_in_flight
    assert not breaker.allow().allowed  # still exactly one probe


@pytest.mark.parametrize("verdict", list(Verdict))
def test_stale_reports_do_not_move_an_open_or_half_open_breaker(breaker, clock, verdict):
    stale = breaker.allow()
    probe(breaker, clock)

    breaker.report(verdict, stale)

    assert breaker.state is BreakerState.HALF_OPEN  # the probe decides, not old evidence
    assert breaker.probe_in_flight


async def test_wait_for_probe_returns_when_the_probe_reports(breaker, clock):
    permit = probe(breaker, clock)
    waiter = asyncio.create_task(breaker.wait_for_probe())
    await asyncio.sleep(0)
    assert not waiter.done()

    breaker.report(Verdict.SUCCESS, permit)
    await asyncio.wait_for(waiter, timeout=1)


async def test_wait_for_probe_is_immediate_without_a_probe(breaker):
    await breaker.wait_for_probe()


# -- verdicts and config ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ProviderError("p", "m", "x", timeout=True), Verdict.FAILURE),
        (ProviderError("p", "m", "x", status=500), Verdict.FAILURE),
        (ProviderError("p", "m", "x", status=503), Verdict.FAILURE),
        (ProviderError("p", "m", "x", status=429), Verdict.NEUTRAL),  # fallback.py's job
        (ProviderError("p", "m", "x", status=400), Verdict.NEUTRAL),  # our bug
        (ProviderError("p", "m", "x", status=403), Verdict.NEUTRAL),
        (ProviderError("cloudflare", "m", "x", status=429, code=3036), Verdict.NEUTRAL),
        (ProviderError("cloudflare", "m", "x", status=400, code=5035), Verdict.NEUTRAL),
        (ProviderError("p", "m", "x", status=400, refused=True), Verdict.NEUTRAL),
        (ProviderError("p", "m", "x", status=500, refused=True), Verdict.NEUTRAL),
    ],
)
def test_only_timeouts_and_5xx_are_failures(error, expected):
    assert verdict_for(error) is expected


def test_board_keeps_one_breaker_per_provider_with_overrides(clock):
    policy = BreakerPolicy(
        default=SETTINGS,
        providers={"groq": BreakerSettings(failure_threshold=1, cooldown_seconds=5)},
    )
    board = BreakerBoard(policy, clock=clock)

    assert board.get("cloudflare") is board.get("cloudflare")
    assert board.get("cloudflare").settings == SETTINGS
    assert board.get("groq").settings.failure_threshold == 1


def test_shipped_policy_loads_and_bad_policy_is_rejected(tmp_path: Path):
    shipped = load_breaker_policy()
    assert shipped.default.failure_threshold > 0 and shipped.default.cooldown_seconds > 0

    bad = tmp_path / "providers.yaml"
    bad.write_text("breaker:\n  default: {failure_threshold: 0, cooldown_seconds: 30}\n")
    with pytest.raises(ValidationError):
        load_breaker_policy(bad)
