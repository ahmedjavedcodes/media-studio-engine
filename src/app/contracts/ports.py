"""Interfaces the engine depends on. Real and fake versions plug in behind them."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel


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
    usage: dict[str, Any] | None = None  # tokens and neurons, for the dev trace


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


# ---- image create (execution-steps/Image Create Agent/TXT2IMG-AGENT-SPEC.md) -------------------


class LLMPort(Protocol):
    """One text or vision model that answers in a Pydantic schema. No free text parsing."""

    model: str

    async def complete_json[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        images: list[str] | None = None,  # data URLs
        max_tokens: int,
        temperature: float,
        timeout: float,
        reasoning_effort: str | None = None,
        usage: dict[str, Any]
        | None = None,  # filled with prompt_tokens, completion_tokens, neurons
    ) -> T: ...


@dataclass(frozen=True)
class ImageOptions:
    """Per-call options. Built by CODE from the request and the model card, never by the LLM."""

    references: tuple[str, ...] = ()  # reference images: data URLs or https URLs
    quality: str | None = None  # GPT Image only. Never `low` on the 2.6 models
    resolution: str | None = None  # "1K" and so on, for providers that take one
    output_format: str | None = None  # "svg" for vector models, "gif", "png" ...
    background: str | None = None  # "transparent"
    brand_colors: tuple[str, ...] = ()  # "#RRGGBB", Recraft only
    complexity: int | None = None  # GPT Image 2.6 only
    endpoint: str = "images"  # OpenRouter: "images" or "chat" (Gemini, multi-turn)
    history: tuple[dict[str, Any], ...] = ()  # earlier chat messages, for multi-turn edits


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes  # the file bytes. The type is read from them, never assumed
    cost_usd: float | None = None  # billed cost when the provider reports it
    media_type: str | None = None  # what the provider said (image/svg+xml ...). A hint only


class ImageProviderPort(Protocol):
    name: str

    async def generate(
        self,
        *,
        slug: str,
        family: str,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        aspect_ratio: str,
        seed: int | None,
        timeout: float,
        estimated_cost_usd: float = 0.0,  # what the registry expects this call to cost
        options: ImageOptions | None = None,
    ) -> GeneratedImage: ...


class CreditPort(Protocol):
    """Hold credits before generating, charge only for delivered images, refund the rest."""

    async def hold(self, user_id: str, run_id: str, amount: int) -> str:
        """Returns a hold id. Raises OutOfCredits."""
        ...

    async def settle(self, hold_id: str, charge: int) -> int:
        """Charge `charge` (at most the held amount), refund the rest. Returns the charge."""
        ...

    async def release(self, hold_id: str) -> None:
        """Refund the whole hold. Safe to call twice."""
        ...

    async def balance(self, user_id: str) -> int: ...
