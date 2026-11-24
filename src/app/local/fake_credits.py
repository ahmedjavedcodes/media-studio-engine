"""In-memory credits for dev and tests. Resets when the process restarts."""

import uuid
from dataclasses import dataclass

from app.core.errors import OutOfCredits


@dataclass
class _Hold:
    user_id: str
    amount: int
    done: bool = False


class FakeCredits:
    def __init__(self, start_balance: int = 100) -> None:
        self._start = start_balance
        self._balances: dict[str, int] = {}
        self._holds: dict[str, _Hold] = {}

    def _bal(self, user_id: str) -> int:
        return self._balances.setdefault(user_id, self._start)

    async def balance(self, user_id: str) -> int:
        return self._bal(user_id)

    async def hold(self, user_id: str, run_id: str, amount: int) -> str:
        if self._bal(user_id) < amount:
            raise OutOfCredits(f"You need {amount} credits for this. Please top up.")
        self._balances[user_id] = self._bal(user_id) - amount
        hold_id = f"{run_id}-{uuid.uuid4().hex[:8]}"
        self._holds[hold_id] = _Hold(user_id, amount)
        return hold_id

    async def settle(self, hold_id: str, charge: int) -> int:
        hold = self._holds[hold_id]
        if hold.done:
            return 0
        hold.done = True
        charged = max(0, min(charge, hold.amount))
        self._balances[hold.user_id] = self._bal(hold.user_id) + hold.amount - charged
        return charged

    async def release(self, hold_id: str) -> None:
        hold = self._holds.get(hold_id)
        if hold is None or hold.done:
            return
        hold.done = True
        self._balances[hold.user_id] = self._bal(hold.user_id) + hold.amount
