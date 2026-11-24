"""In-memory StrikeLedger for dev and tests. Resets when the process restarts."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal

from app.contracts.ports import AccountState, RejectCounts


def _utcnow() -> datetime:
    return datetime.now(UTC)


class FakeStrikeLedger:
    def __init__(self, now: Callable[[], datetime] = _utcnow) -> None:
        self._now = now
        self._strikes: dict[str, dict[str, datetime]] = {}
        self._rejects: dict[str, dict[str, tuple[datetime, str]]] = {}
        self._cooldowns: dict[str, datetime] = {}
        self._status: dict[str, Literal["active", "frozen", "banned"]] = {}
        self.ban_reasons: dict[str, str] = {}

    async def get_account(self, user_id: str) -> AccountState:
        return AccountState(
            status=self._status.get(user_id, "active"),
            cooldown_until=self._cooldowns.get(user_id),
        )

    async def add_strike(self, user_id: str, request_id: str, window_days: int) -> int:
        mine = self._strikes.setdefault(user_id, {})
        mine.setdefault(request_id, self._now())  # same request_id: no second strike
        cutoff = self._now() - timedelta(days=window_days)
        return sum(1 for ts in mine.values() if ts > cutoff)

    async def add_reject(
        self,
        user_id: str,
        request_id: str,
        kind: Literal["standard", "instant"],
        window_minutes: int,
    ) -> RejectCounts:
        mine = self._rejects.setdefault(user_id, {})
        mine.setdefault(request_id, (self._now(), kind))
        cutoff = self._now() - timedelta(minutes=window_minutes)
        recent = [k for ts, k in mine.values() if ts > cutoff]
        return RejectCounts(total=len(recent), standard=recent.count("standard"))

    async def set_cooldown(self, user_id: str, until: datetime) -> None:
        current = self._cooldowns.get(user_id)
        if current is None or until > current:
            self._cooldowns[user_id] = until

    async def ban(self, user_id: str, reason: str) -> None:
        self._status[user_id] = "banned"
        self.ban_reasons[user_id] = reason
