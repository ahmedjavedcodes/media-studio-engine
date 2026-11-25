"""When Cloudflare fails, ask Groq. Only for a failure the call can be retried after: a 429, a 5xx
or a timeout (`ProviderCallFailed.retryable`). A bad answer or a refusal never falls back, so a
content refusal is never shopped around. Groq is paid per token, so this is on only when the owner
sets GROQ_FALLBACK=true.

`model` and `provider` show who answered last, for the dev trace. With images made in parallel
the label can name the other one's answer. It never changes a result.
"""

from typing import Any

import structlog
from pydantic import BaseModel

from app.contracts.ports import LLMPort, VisionPort, VisionRaw
from app.core.errors import ProviderCallFailed

log = structlog.get_logger("provider")


def _note(usage: dict[str, Any] | None, backup_model: str, why: str) -> None:
    log.warning("provider_fallback", to="groq", model=backup_model, why=why)
    if usage is not None:
        usage["fallback_to"] = f"groq:{backup_model}"
        usage["fallback_why"] = why


class FallbackLLM:
    def __init__(self, primary: LLMPort, backup: LLMPort) -> None:
        self._primary = primary
        self._backup = backup
        self.model = primary.model
        self.provider = primary.provider

    async def complete_json[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        images: list[str] | None = None,
        max_tokens: int,
        temperature: float,
        timeout: float,
        reasoning_effort: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> T:
        args: dict[str, Any] = {
            "system": system,
            "user": user,
            "schema": schema,
            "images": images,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
            "reasoning_effort": reasoning_effort,
            "usage": usage,
        }
        try:
            out: T = await self._primary.complete_json(**args)
            self.model, self.provider = self._primary.model, self._primary.provider
            return out
        except ProviderCallFailed as exc:
            if not exc.retryable:
                raise
            _note(usage, self._backup.model, str(exc))
            backed_up: T = await self._backup.complete_json(**args)
            self.model, self.provider = self._backup.model, self._backup.provider
            return backed_up


class FallbackVision:
    def __init__(self, primary: VisionPort, backup: VisionPort) -> None:
        self._primary = primary
        self._backup = backup
        self.model = primary.model
        self.provider = getattr(primary, "provider", "cloudflare")

    async def classify(self, *, text: str, image_urls: list[str], timeout: float) -> VisionRaw:
        try:
            out = await self._primary.classify(text=text, image_urls=image_urls, timeout=timeout)
            self.model = self._primary.model
            self.provider = getattr(self._primary, "provider", "cloudflare")
            return out
        except ProviderCallFailed as exc:
            if not exc.retryable:
                raise
            _note(None, self._backup.model, str(exc))
            out = await self._backup.classify(text=text, image_urls=image_urls, timeout=timeout)
            self.model = self._backup.model
            self.provider = getattr(self._backup, "provider", "groq")
            return out
