"""Strikes and cooldowns. Replaces spec section 10.

- Instant reject: 1 strike. Strikes 1 to 4 warn. Strike 5 (in 30 days) bans.
- Standard reject: no strike. 1 minute cooldown, or 2 minutes if there was another
  standard reject in the last 10 minutes.
- 5 rejects of any kind in 10 minutes: 30 minute cooldown.
- A new cooldown never shortens one that is already running.
Tiers only decide INSTANT vs STANDARD (policy.decide). They do not change these limits.
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.contracts.ports import StrikeLedger
from app.core.config import Settings
from app.guardrails.policy import LONG_COOLDOWN_SEC, REJECT_WINDOW_MINUTES, REJECTS_FOR_COOLDOWN
from app.guardrails.schemas import AccountAction, Decision


@dataclass(frozen=True)
class StrikeOutcome:
    account_action: AccountAction = "none"
    strike_count: int | None = None
    retry_after: int | None = None


def _cooldown_action(seconds: int, settings: Settings) -> AccountAction:
    if seconds >= LONG_COOLDOWN_SEC:
        return "cooldown_30m"
    if seconds >= settings.standard_repeat_cooldown_sec:
        return "cooldown_2m"
    return "cooldown_1m"


async def apply_rules(
    ledger: StrikeLedger,
    settings: Settings,
    *,
    user_id: str,
    request_id: str,
    decision: Decision,
    now: datetime | None = None,
) -> StrikeOutcome:
    """Record a reject and work out the account action. Safe to call twice for one request_id."""
    if decision is Decision.APPROVE:
        return StrikeOutcome()
    now = now or datetime.now(UTC)
    instant = decision is Decision.INSTANT_REJECT

    counts = await ledger.add_reject(
        user_id, request_id, "instant" if instant else "standard", REJECT_WINDOW_MINUTES
    )

    strike_count: int | None = None
    action: AccountAction = "none"
    cooldown_sec = 0

    if instant:
        strike_count = await ledger.add_strike(user_id, request_id, settings.strike_window_days)
        if strike_count >= settings.instant_strike_limit:
            await ledger.ban(user_id, f"{strike_count} instant-reject strikes")
            return StrikeOutcome("ban", strike_count, None)
        action = "warn"
    else:
        repeat = counts.standard >= 2
        cooldown_sec = (
            settings.standard_repeat_cooldown_sec if repeat else settings.standard_cooldown_sec
        )

    if counts.total >= REJECTS_FOR_COOLDOWN:
        cooldown_sec = max(cooldown_sec, LONG_COOLDOWN_SEC)

    retry_after: int | None = None
    if cooldown_sec:
        action = _cooldown_action(cooldown_sec, settings)
        until = now + timedelta(seconds=cooldown_sec)
        current = (await ledger.get_account(user_id)).cooldown_until
        if current is not None and current >= until:
            until = current  # never shorten a running cooldown
        else:
            await ledger.set_cooldown(user_id, until)
        retry_after = max(1, math.ceil((until - now).total_seconds()))
    return StrikeOutcome(action, strike_count, retry_after)
