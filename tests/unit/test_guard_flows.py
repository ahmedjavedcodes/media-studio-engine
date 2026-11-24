"""Flow tests S02 and S05 to S10 from test-manual/guardrails/evals/datasets/guardrail_eval.yaml.

Each test is named after its flow id so `pytest -k "s02 or s05 ..."` lists them. Providers are
fakes and the clock is controlled, so no model is called.
"""

import time
from datetime import UTC, datetime, timedelta

import pytest

from app.core.errors import Cooldown, GuardrailUnavailable
from app.guardrails.agent import GuardDeps, Timing, check
from app.guardrails.schemas import Category, Decision, GuardInput
from app.guardrails.strikes import apply_rules
from app.local.fake_strikes import FakeStrikeLedger
from tests.unit.guard_helpers import FakeJev, FakeVision, jev_answers, make_deps, png

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now


def deps_with_clock(jev: FakeJev, vision: FakeVision | None = None, timing: Timing | None = None):  # type: ignore[no-untyped-def]
    clock = Clock()
    ledger = FakeStrikeLedger(now=clock)
    deps = make_deps(jev, vision, ledger, timing)
    deps.clock = clock
    return deps, ledger, clock


def req(rid: str, images: list[bytes] | None = None) -> GuardInput:
    return GuardInput(request_id=rid, user_id="u1", prompt="a prompt", images=images or [])


async def test_s02_same_request_id_twice_is_one_strike() -> None:
    deps, ledger, _ = deps_with_clock(FakeJev(jev_answers(graphic_gore=0.95)))
    a = await check(req("same"), deps)
    b = await check(req("same"), deps)
    assert a.decision is b.decision is Decision.INSTANT_REJECT
    assert a.strike_count == b.strike_count == 1
    assert len(ledger._strikes["u1"]) == 1  # type: ignore[attr-defined]


async def test_s05_five_rejects_in_10_min_is_30_min_and_later_1_min_does_not_shorten() -> None:
    deps, ledger, clock = deps_with_clock(FakeJev(jev_answers(sexual_content=0.95)))
    for n in range(5):  # 5 standard rejects inside 10 minutes
        d = await check(req(f"r{n}"), deps)
        clock.now += timedelta(seconds=121)  # just past the 2 minute cooldown, still inside 10 min
    assert d.account_action == "cooldown_30m" and d.retry_after == 1800
    long_until = (await ledger.get_account("u1")).cooldown_until
    assert long_until is not None
    # While the 30 min cooldown runs, new requests are blocked at step 0 (429).
    with pytest.raises(Cooldown):
        await check(req("blocked"), deps)
    # A request that was already in flight finishes 11 min later (the 5 rejects are now older
    # than 10 min) and wants a 1 min cooldown. It must not shorten the 30 min one.
    clock.now += timedelta(minutes=11)
    late = await apply_rules(
        ledger,
        deps.settings,
        user_id="u1",
        request_id="late",
        decision=Decision.STANDARD_REJECT,
        now=clock.now,
    )
    assert late.account_action == "cooldown_1m"
    assert (await ledger.get_account("u1")).cooldown_until == long_until  # not shortened
    assert late.retry_after == int((long_until - clock.now).total_seconds())  # real wait, not 60
    assert late.retry_after > 60


async def test_s06_vision_times_out_jev_approves_is_503_no_strike_not_a_reject() -> None:
    timing = Timing(vision_timeout_s=0.2, vision_retry_delay_s=None)
    deps, ledger, _ = deps_with_clock(FakeJev(), FakeVision(delay=10), timing)
    with pytest.raises(GuardrailUnavailable) as ei:
        await check(req("r1", [png()]), deps)
    assert ei.value.http_status == 503
    assert ei.value.details and ei.value.details[0]["source"] == "vision"
    assert "u1" not in ledger._strikes and "u1" not in ledger._rejects  # type: ignore[attr-defined]
    assert (await ledger.get_account("u1")).cooldown_until is None


async def test_s07_vision_times_out_jev_standard_reject_is_422() -> None:
    timing = Timing(vision_timeout_s=0.2, vision_retry_delay_s=None)
    deps, _, _ = deps_with_clock(
        FakeJev(jev_answers(sexual_content=0.95)), FakeVision(delay=10), timing
    )
    d = await check(req("r1", [png()]), deps)
    assert d.decision is Decision.STANDARD_REJECT and d.category is Category.NSFW_EXPLICIT


async def test_s08_jev_instant_cancels_running_vision() -> None:
    vision = FakeVision(delay=10)
    deps, _, _ = deps_with_clock(FakeJev(jev_answers(csam=0.97)), vision)
    start = time.monotonic()
    d = await check(req("r1", [png()]), deps)
    assert d.decision is Decision.INSTANT_REJECT
    assert vision.cancelled and time.monotonic() - start < 1.0


async def test_s09_vision_unsafe_without_logprobs_is_standard_reject_needs_review_no_strike() -> (
    None
):
    deps, ledger, _ = deps_with_clock(FakeJev(), FakeVision("unsafe\nS4", None))
    d = await check(req("r1", [png()]), deps)
    assert d.decision is Decision.STANDARD_REJECT and d.category is Category.CSAM
    assert any(v.needs_review for v in d.verdicts)
    assert d.strike_count is None
    assert "u1" not in ledger._strikes  # type: ignore[attr-defined]


async def test_s10_vision_returns_safe_with_a_dot_is_503() -> None:
    deps, ledger, _ = deps_with_clock(FakeJev(), FakeVision("Safe.", 0.02))
    with pytest.raises(GuardrailUnavailable) as ei:
        await check(req("r1", [png()]), deps)
    assert ei.value.details[0]["error"] == "InvalidModelOutput"
    assert "u1" not in ledger._rejects  # type: ignore[attr-defined]


async def test_s04_cooldown_request_is_429_and_not_counted() -> None:
    deps, ledger, clock = deps_with_clock(FakeJev(jev_answers(sexual_content=0.95)))
    await check(req("r1"), deps)
    before = len(ledger._rejects["u1"])  # type: ignore[attr-defined]
    with pytest.raises(Cooldown):
        await check(req("r2"), deps)
    assert len(ledger._rejects["u1"]) == before  # type: ignore[attr-defined]


def test_deps_type_is_exported() -> None:
    assert GuardDeps is not None
