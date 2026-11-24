from datetime import UTC, datetime, timedelta

import pytest

from app.core.config import Settings
from app.guardrails.schemas import Decision
from app.guardrails.strikes import StrikeOutcome, apply_rules
from app.local.fake_strikes import FakeStrikeLedger

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
SETTINGS = Settings(_env_file=None)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw: float) -> None:
        self.now += timedelta(**kw)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def ledger(clock: Clock) -> FakeStrikeLedger:
    return FakeStrikeLedger(now=clock)


async def reject(
    ledger: FakeStrikeLedger, clock: Clock, rid: str, decision: Decision, user: str = "u1"
) -> StrikeOutcome:
    return await apply_rules(
        ledger, SETTINGS, user_id=user, request_id=rid, decision=decision, now=clock.now
    )


def test_defaults_match_rules() -> None:
    assert SETTINGS.instant_strike_limit == 5
    assert SETTINGS.strike_window_days == 30
    assert SETTINGS.standard_cooldown_sec == 60
    assert SETTINGS.standard_repeat_cooldown_sec == 120


async def test_approve_does_nothing(ledger: FakeStrikeLedger, clock: Clock) -> None:
    out = await reject(ledger, clock, "r1", Decision.APPROVE)
    assert out == StrikeOutcome()
    assert (await ledger.get_account("u1")).cooldown_until is None


async def test_strikes_1_to_4_warn_and_5_bans(ledger: FakeStrikeLedger, clock: Clock) -> None:
    for n in range(1, 5):
        out = await reject(ledger, clock, f"r{n}", Decision.INSTANT_REJECT)
        assert (out.account_action, out.strike_count) == ("warn", n)
        assert (await ledger.get_account("u1")).status == "active"
        clock.advance(minutes=15)  # keep clear of the 5-rejects-in-10-minutes rule
    out = await reject(ledger, clock, "r5", Decision.INSTANT_REJECT)
    assert (out.account_action, out.strike_count) == ("ban", 5)
    assert (await ledger.get_account("u1")).status == "banned"


async def test_old_strikes_are_not_counted(ledger: FakeStrikeLedger, clock: Clock) -> None:
    for n in range(1, 5):
        await reject(ledger, clock, f"r{n}", Decision.INSTANT_REJECT)
        clock.advance(minutes=15)
    clock.advance(days=31)
    out = await reject(ledger, clock, "late", Decision.INSTANT_REJECT)
    assert (out.account_action, out.strike_count) == ("warn", 1)
    assert (await ledger.get_account("u1")).status == "active"


async def test_same_request_id_twice_is_one_strike(ledger: FakeStrikeLedger, clock: Clock) -> None:
    a = await reject(ledger, clock, "same", Decision.INSTANT_REJECT)
    b = await reject(ledger, clock, "same", Decision.INSTANT_REJECT)
    assert a.strike_count == b.strike_count == 1


async def test_instant_reject_has_no_cooldown(ledger: FakeStrikeLedger, clock: Clock) -> None:
    out = await reject(ledger, clock, "r1", Decision.INSTANT_REJECT)
    assert out.retry_after is None
    assert (await ledger.get_account("u1")).cooldown_until is None


async def test_first_standard_reject_is_60s(ledger: FakeStrikeLedger, clock: Clock) -> None:
    out = await reject(ledger, clock, "r1", Decision.STANDARD_REJECT)
    assert (out.account_action, out.retry_after, out.strike_count) == ("cooldown_1m", 60, None)
    assert (await ledger.get_account("u1")).cooldown_until == T0 + timedelta(seconds=60)


async def test_repeat_standard_within_10_min_is_120s(
    ledger: FakeStrikeLedger, clock: Clock
) -> None:
    await reject(ledger, clock, "r1", Decision.STANDARD_REJECT)
    clock.advance(minutes=2)
    out = await reject(ledger, clock, "r2", Decision.STANDARD_REJECT)
    assert (out.account_action, out.retry_after) == ("cooldown_2m", 120)


async def test_repeat_standard_after_10_min_is_60s(ledger: FakeStrikeLedger, clock: Clock) -> None:
    await reject(ledger, clock, "r1", Decision.STANDARD_REJECT)
    clock.advance(minutes=11)
    out = await reject(ledger, clock, "r2", Decision.STANDARD_REJECT)
    assert (out.account_action, out.retry_after) == ("cooldown_1m", 60)


async def test_instant_reject_does_not_make_next_standard_a_repeat(
    ledger: FakeStrikeLedger, clock: Clock
) -> None:
    await reject(ledger, clock, "r1", Decision.INSTANT_REJECT)
    out = await reject(ledger, clock, "r2", Decision.STANDARD_REJECT)
    assert out.retry_after == 60


async def test_five_rejects_in_10_min_is_30_min(ledger: FakeStrikeLedger, clock: Clock) -> None:
    for n in range(4):
        await reject(ledger, clock, f"r{n}", Decision.STANDARD_REJECT)
        clock.advance(seconds=100)
    out = await reject(ledger, clock, "r4", Decision.STANDARD_REJECT)
    assert (out.account_action, out.retry_after) == ("cooldown_30m", 1800)
    assert (await ledger.get_account("u1")).cooldown_until == clock.now + timedelta(minutes=30)


async def test_sixth_reject_inside_window_renews_30_min(
    ledger: FakeStrikeLedger, clock: Clock
) -> None:
    for n in range(5):
        await reject(ledger, clock, f"r{n}", Decision.STANDARD_REJECT)
    clock.advance(seconds=5)
    out = await reject(ledger, clock, "r6", Decision.STANDARD_REJECT)
    assert out.account_action == "cooldown_30m"


async def test_five_mixed_rejects_also_trigger_30_min(
    ledger: FakeStrikeLedger, clock: Clock
) -> None:
    for n in range(4):
        await reject(ledger, clock, f"i{n}", Decision.INSTANT_REJECT)
    out = await reject(ledger, clock, "s", Decision.STANDARD_REJECT)
    assert out.account_action == "cooldown_30m"


async def test_later_short_cooldown_does_not_shorten_long_one(
    ledger: FakeStrikeLedger, clock: Clock
) -> None:
    for n in range(5):
        await reject(ledger, clock, f"r{n}", Decision.STANDARD_REJECT)
    long_until = (await ledger.get_account("u1")).cooldown_until
    assert long_until == T0 + timedelta(minutes=30)
    # the 5 rejects are now older than 10 minutes. A request that was already in flight
    # finishes and wants a 1 minute cooldown. The 30 minute one must stay.
    clock.advance(minutes=11)
    out = await reject(ledger, clock, "late", Decision.STANDARD_REJECT)
    assert out.account_action == "cooldown_1m"
    assert (await ledger.get_account("u1")).cooldown_until == long_until
    assert out.retry_after == 19 * 60  # reports the real wait, not 60


async def test_ban_overrides_cooldown(ledger: FakeStrikeLedger, clock: Clock) -> None:
    for n in range(4):
        await reject(ledger, clock, f"i{n}", Decision.INSTANT_REJECT)  # 4 rejects, 4 strikes
    out = await reject(ledger, clock, "i4", Decision.INSTANT_REJECT)  # 5th: ban, not cooldown
    assert out.account_action == "ban"
    assert (await ledger.get_account("u1")).cooldown_until is None


async def test_users_are_separate(ledger: FakeStrikeLedger, clock: Clock) -> None:
    await reject(ledger, clock, "r1", Decision.STANDARD_REJECT, user="a")
    assert (await ledger.get_account("b")).cooldown_until is None
