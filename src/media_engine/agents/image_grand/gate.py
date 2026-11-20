"""The Gate and the safety reviewer: the run's ONE input safety decision, before any spend.

    Gate: one small structured call (IMAGE-CREATE-OPTIMIZE §4.1) returning
        {safety, safety_categories, route, task_type, prompt_is_detailed}
    SafetyReviewer: only when the Gate says `review`; one call on the `safety_review` chain
        (gpt-oss-safeguard-20b, 3 requests/minute, so never on the hot path) -> allow | block

Both FAIL CLOSED. The old router defaulted to `simple` when its call failed, which is right for
routing and wrong for safety: a Gate that could not decide is not an "allow". Their failures are
FAILED envelopes (with the failure kind), and the caller ends the run before any image is paid
for. The Gate and the Brief are deliberately two separate calls (§3): a model with one job and a
strict schema is much harder to talk out of its job than one that also writes prompts.

inputs (Gate):           prompt: str, settings: dict[str, str] (optional; user text too)
inputs (SafetyReviewer): prompt: str, settings (optional), safety_categories: list[str]
"""

from __future__ import annotations

from typing import Any

from media_engine.agents.base import Agent, AgentConfig
from media_engine.agents.image_grand.limiter import SlidingWindowLimiter
from media_engine.contracts import AuthContext, Envelope, Goal, Spend, Status, StoppingCondition
from media_engine.core.logging import get_logger
from media_engine.core.prompts import PROMPTS_DIR, load_prompt, render, tag_stripper
from media_engine.gateway.llm_gateway import LLMGateway
from media_engine.ports.event_sink import EventSink
from media_engine.schemas import GateDecision, SafetyReview

GATE_INPUT = PROMPTS_DIR / "gate" / "gate.input.md"
REVIEW_INPUT = PROMPTS_DIR / "gate" / "safety_review.input.md"
GATE_SUMMARY = "Checked the request."
REVIEW_SUMMARY = "Reviewed the request."
_TAG = tag_stripper("user_request", "concerns", "settings")
NONE_GIVEN = "none given"

log = get_logger("agents.gate")


def _clean_prompt(inputs: dict[str, Any]) -> str:
    raw = inputs.get("prompt")
    return _TAG.sub("", raw).strip() if isinstance(raw, str) else ""


def _clean_settings(inputs: dict[str, Any]) -> str:
    """User-chosen settings as `key: value` lines, tags stripped. They are judged with the
    prompt: a `style` value is as much user text as the request itself."""
    lines = []
    for key, value in (inputs.get("settings") or {}).items():
        k, v = _TAG.sub("", str(key)).strip(), _TAG.sub("", str(value)).strip()
        if k and v:
            lines.append(f"{k}: {v}")
    return "\n".join(lines) or NONE_GIVEN


class Gate(Agent):
    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope:
        prompt = _clean_prompt(inputs)
        if not prompt:
            return self.failed("no prompt given", Spend(), "invalid_input")
        content = render(load_prompt(GATE_INPUT), prompt=prompt, settings=_clean_settings(inputs))
        result = await self.call_structured(auth, stop, content, GateDecision)
        if result.output is None:
            # Fail closed: an undecided Gate is never an allow.
            log.warning("gate.undecided", failure=result.failure, error=result.error)
            return self.failed(
                f"the request could not be checked ({result.failure})",
                result.spend,
                result.failure,
            )
        return self.envelope(Status.DONE, result.output.model_dump(), GATE_SUMMARY, result.spend)


class SafetyReviewer(Agent):
    """`limiter` keeps the process under the safeguard model's rate limit: the review waits up
    to `wait_s` for a slot and otherwise fails with `rate_limited` WITHOUT calling the model."""

    def __init__(
        self,
        config: AgentConfig,
        gateway: LLMGateway | None,
        events: EventSink,
        *,
        limiter: SlidingWindowLimiter | None = None,
        wait_s: float = 0.0,
    ) -> None:
        super().__init__(config, gateway, events)
        self._limiter = limiter
        self._wait_s = wait_s

    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope:
        prompt = _clean_prompt(inputs)
        if not prompt:
            return self.failed("no prompt given", Spend(), "invalid_input")
        if self._limiter is not None and not await self._limiter.acquire(self._wait_s):
            log.warning("safety_review.rate_limited", wait_s=self._wait_s)
            return self.failed(
                "the safety review is rate limited right now", Spend(), "rate_limited"
            )
        concerns = [
            _TAG.sub("", str(c)).strip() for c in inputs.get("safety_categories") or [] if c
        ]
        content = render(
            load_prompt(REVIEW_INPUT),
            prompt=prompt,
            settings=_clean_settings(inputs),
            concerns=", ".join(concerns) or NONE_GIVEN,
        )
        result = await self.call_structured(auth, stop, content, SafetyReview)
        if result.output is None:
            log.warning("safety_review.undecided", failure=result.failure, error=result.error)
            return self.failed(
                f"the safety review could not be completed ({result.failure})",
                result.spend,
                result.failure,
            )
        return self.envelope(Status.DONE, result.output.model_dump(), REVIEW_SUMMARY, result.spend)


__all__ = ["Gate", "SafetyReviewer"]
