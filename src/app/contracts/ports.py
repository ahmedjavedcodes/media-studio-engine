"""Interfaces the engine depends on. Real and fake versions plug in behind them."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol


class JevPort(Protocol):
    """TypeSafe Jev. Returns the raw `answers` object for the given noul questions."""

    model: str

    async def ask(
        self, *, state: dict[str, Any], questions: dict[str, Any], timeout: float
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class VisionRaw:
    """What a Llama Guard 4 call returned, before parsing."""

    text: str
    # (token, logprob) pairs for the first generated token and its top alternatives.
    # None when the provider gave no logprobs.
    first_token_logprobs: list[tuple[str, float]] | None


class VisionPort(Protocol):
    model: str

    async def classify(self, *, text: str, image_urls: list[str], timeout: float) -> VisionRaw: ...


@dataclass(frozen=True)
class AccountState:
    status: Literal["active", "frozen", "banned"] = "active"
    cooldown_until: datetime | None = None


@dataclass(frozen=True)
class RejectCounts:
    """Rejects in the look-back window, including the one just recorded."""

    total: int
    standard: int


class StrikeLedger(Protocol):
    """Account state, strikes and rejects. Every write is idempotent by `request_id`."""

    async def get_account(self, user_id: str) -> AccountState: ...

    async def add_strike(self, user_id: str, request_id: str, window_days: int) -> int:
        """Record a strike. Returns the user's strikes in the last `window_days` days."""
        ...

    async def add_reject(
        self,
        user_id: str,
        request_id: str,
        kind: Literal["standard", "instant"],
        window_minutes: int,
    ) -> RejectCounts:
        """Record a reject. Returns the counts in the last `window_minutes` minutes."""
        ...

    async def set_cooldown(self, user_id: str, until: datetime) -> None:
        """Start a cooldown. Never shortens one that is already running."""
        ...

    async def ban(self, user_id: str, reason: str) -> None:
        """Freeze the account, revoke its tokens and ban it for good."""
        ...
