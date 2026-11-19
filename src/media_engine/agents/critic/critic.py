"""Critic: one vision call returning a quality score and an independent safety verdict.

inputs:
    image_id:  str        the image to review, resolved through the ArtifactStore (required)
    prompt:    str        the user's RAW request (required). Scored against this and must_have,
                          never against the enhanced prompt: that is not even sent.
    must_have: list[str]  requirements from the user's words (optional; may be empty)

output (envelope):
    DONE:     image_id, quality_score, critique, safety_verdict="allow"
    REJECTED: blocked_image_id, safety_verdict="block", safety_reason. The image is NOT under
              `image_id`, so it can never be picked up as a deliverable.
    FAILED:   failure kind. The image was NOT safety-checked and must not be delivered.

Safety is a verdict, not a score. `block` is terminal: REJECTED whatever the quality score,
and it wins over the repair path. If ANY reply in this step said "block", even one that failed
validation or was cut off mid-JSON, the step is REJECTED. Otherwise an invalid "block" could be
repaired into "allow".

A blocked image is marked orphaned (reason "safety_block") so it is never delivered and the
sweep removes it. TODO(backend): whether safety-blocked images are instead retained for abuse
review is a product/legal decision; the backend sweep can branch on the reason.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from media_engine.agents.base import (
    Agent,
    AgentConfig,
    Structured,
    describe_validation_error,
    extract_json,
)
from media_engine.contracts import AuthContext, Envelope, Goal, Spend, Status, StoppingCondition
from media_engine.core.images import NotAnImage, data_url, downscale
from media_engine.core.logging import get_logger
from media_engine.core.prompts import PROMPTS_DIR, load_prompt, render, tag_stripper
from media_engine.gateway.llm_gateway import LLMGateway
from media_engine.ports.artifact_store import ArtifactStore
from media_engine.ports.event_sink import EventSink
from media_engine.schemas import CriticOutput

INPUT_TEMPLATE = PROMPTS_DIR / "image_create" / "critic.input.md"
DEFAULT_SUMMARY = "Reviewed your image."
BLOCKED_SUMMARY = "This image could not be delivered because it did not pass the safety review."
DEFAULT_BLOCK_REASON = "safety policy"
NONE_GIVEN = "none"
ORPHAN_REASON = "safety_block"

# Text-level fallback for replies with no decodable verdict (e.g. cut off at the token cap
# mid-JSON). Fail-closed on purpose: a stray match blocks an image, a miss could deliver one.
_BLOCK_TEXT = re.compile(r'"safety_verdict"\s*:\s*"\s*block\s*"', re.IGNORECASE)
_REASON_TEXT = re.compile(r'"safety_reason"\s*:\s*"([^"\\]{1,80})"')

log = get_logger("agents.critic")

_TAG = tag_stripper("user_request", "must_have")


def sanitize(text: str) -> str:
    return _TAG.sub("", text).strip()


class CriticParams(BaseModel):
    """`params` block for critic in agents.yaml. No defaults: config owns the values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_side: int = Field(gt=0)  # long side of the image sent for review, in px
    jpeg_quality: int = Field(ge=1, le=95)


class CriticInputs(BaseModel):
    model_config = ConfigDict(extra="ignore")  # e.g. an enhanced_prompt passed by mistake

    image_id: str = Field(min_length=1)
    prompt: str
    must_have: list[str] = Field(default_factory=list)


def _clean_reason(reason: object) -> str:
    return reason.strip() if isinstance(reason, str) and reason.strip() else DEFAULT_BLOCK_REASON


def reply_block_reason(text: str) -> str | None:
    """The block reason if this one reply says block, else None. The JSON verdict is used when
    the reply has one; the text check runs only when it does not (no decodable JSON, or JSON
    without a verdict), so a decoded "allow" is never overruled by stray text."""
    candidate = extract_json(text)
    body: object = json.loads(candidate) if candidate is not None else None
    if isinstance(body, dict) and isinstance(body.get("safety_verdict"), str):
        if body["safety_verdict"].strip().lower() == "block":
            return _clean_reason(body.get("safety_reason"))
        return None
    if _BLOCK_TEXT.search(text):
        match = _REASON_TEXT.search(text)
        return _clean_reason(match.group(1) if match else None)
    return None


def blocked_reason(result: Structured[CriticOutput]) -> str | None:
    """The safety reason if ANY reply in the step said block (validated or not), else None."""
    if result.output is not None and result.output.safety_verdict == "block":
        return _clean_reason(result.output.safety_reason)
    for call in result.calls:
        reason = reply_block_reason(call.text)
        if reason is not None:
            return reason
    return None


class Critic(Agent):
    def __init__(
        self,
        config: AgentConfig,
        gateway: LLMGateway | None,
        events: EventSink,
        *,
        artifacts: ArtifactStore,
    ) -> None:
        if gateway is None:
            raise ValueError("The Critic needs an LLM gateway (it makes the vision call)")
        super().__init__(config, gateway, events)
        self._artifacts = artifacts
        # Validate the params block at construction, so bad config fails at startup.
        self.params = CriticParams.model_validate(config.model_policy.params)

    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope:
        try:
            args = CriticInputs.model_validate(inputs)
        except ValidationError as err:
            return self.failed(
                "invalid inputs:\n" + describe_validation_error(err), Spend(), "invalid_input"
            )
        user_prompt = sanitize(args.prompt)
        if not user_prompt:
            return self.failed("no prompt given", Spend(), "invalid_input")

        try:
            data = await self._artifacts.get(args.image_id)
        except Exception as err:  # the store is a backend port: any failure is "storage"
            log.exception("critic.load_failed")
            return self.failed(f"could not load image: {type(err).__name__}", Spend(), "storage")
        try:
            # Decode + resize is CPU work: keep it off the event loop.
            review, mime = await asyncio.to_thread(
                downscale, data, self.params.max_side, self.params.jpeg_quality
            )
        except NotAnImage as err:
            return self.failed(str(err), Spend(), "bad_image")

        must_have = [m for m in (sanitize(item) for item in args.must_have) if m]
        text = render(
            load_prompt(INPUT_TEMPLATE),
            prompt=user_prompt,
            must_have="\n".join(f"- {m}" for m in must_have) or NONE_GIVEN,
        )
        # OpenAI-style multimodal content. UNVERIFIED on Cloudflare's chat endpoint for the
        # vision chain models: confirm with playground/smoke_test.py --only cf-vision-scout.
        content: list[dict[str, Any]] = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url(review, mime)}},
        ]

        result = await self.call_structured(auth, stop, content, CriticOutput)

        # Safety first, before looking at whether the output validated or what it scored.
        reason = blocked_reason(result)
        if reason is not None:
            log.warning("critic.blocked", reason=reason, calls=len(result.calls))
            try:
                await self._artifacts.mark_orphaned(args.image_id, ORPHAN_REASON)
            except Exception:  # never let bookkeeping turn a block into anything else
                log.exception("critic.mark_orphaned_failed")
            return self.envelope(
                Status.REJECTED,
                {
                    "blocked_image_id": args.image_id,
                    "safety_verdict": "block",
                    "safety_reason": reason,
                },
                BLOCKED_SUMMARY,
                result.spend,
                reason=f"safety: {reason}",
            )

        if result.output is None:
            return self.failed(
                result.error or "critic produced no output", result.spend, result.failure
            )
        out = result.output
        return self.envelope(
            Status.DONE,
            {
                "image_id": args.image_id,
                "quality_score": out.quality_score,
                "critique": out.critique,
                "safety_verdict": "allow",
            },
            out.summary.strip() or DEFAULT_SUMMARY,
            result.spend,
        )
