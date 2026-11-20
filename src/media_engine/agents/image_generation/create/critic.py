"""Critic: ONE vision call that sees every image of the run and ranks them (§5.4).

inputs:
    image_ids: list[str]  the images to review, resolved through the ArtifactStore (required)
    prompt:    str        the user's RAW request (required). Scored against this and must_have,
                          never against the enhanced prompt: that is not even sent.
    must_have: list[str]  requirements from the user's words (optional; may be empty)
    task_type: str        picks the rubric prompt: `create` today (edit / merge later, §10)

The images sent are the stored THUMBNAILS (`params.thumbnail_size`, made once at store time);
only a missing thumbnail falls back to downscaling the original.

output (envelope):
    DONE:   reviews: one entry per image the model ruled on, in rank order:
            image_id, verdict (allow|block), safety_reason, scores (the four dimensions),
            quality_score (their mean; safety is never part of it), rank (1 = best allowed,
            normalised here: gaps and ties in the model's ranks are closed), critique (only
            when a dimension is below threshold). An image the model left out has no entry:
            it is unchecked and must not be delivered.
    FAILED: failure kind. NO image was safety-checked.

Safety is a verdict, not a score. A block drops THAT image only (§5.5): it is marked orphaned
(reason "safety_block") and never delivered; the others stand. If a reply that could not be
read mentions "block" anywhere, the step fails without a retry: an unreadable block cannot be
attributed to an image, and a retry could turn it into "allow".

RETRY (create is linear, so no supervisor loop): a call with no usable output and no "block"
text is tried ONCE more (`params.retries`), inside this step, on the same images. Never for a
failure another try cannot fix (NO_RETRY). It never generates, so it never spends on an image.

TODO(backend): whether safety-blocked images are retained for abuse review instead of swept
is a product/legal decision; the sweep can branch on the orphan reason.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from media_engine.agents.base import (
    Agent,
    AgentConfig,
    Structured,
    describe_validation_error,
)
from media_engine.contracts import AuthContext, Envelope, Goal, Spend, Status, StoppingCondition
from media_engine.core.images import NotAnImage, data_url, downscale, sniff_media_type
from media_engine.core.logging import get_logger
from media_engine.core.prompts import PROMPTS_DIR, load_prompt, render, tag_stripper
from media_engine.gateway.llm_gateway import LLMGateway
from media_engine.ports.artifact_store import ArtifactStore
from media_engine.ports.event_sink import EventSink
from media_engine.schemas import CriticOutput, ImageReview

INPUT_TEMPLATE = PROMPTS_DIR / "critic" / "critic.input.md"
DEFAULT_SUMMARY = "Reviewed your image."
DEFAULT_BLOCK_REASON = "safety policy"
# Why a batch failed when a reply said "block" that no valid answer carried. Logged
# (critic.unattributed_block) on every occurrence, so its rate can be counted.
UNATTRIBUTED_BLOCK = "a reply said block and the final answer did not"
NONE_GIVEN = "none"
ORPHAN_REASON = "safety_block"

# Text-level check for replies that could not be read (e.g. cut off at the token cap mid-JSON).
_BLOCK_TEXT = re.compile(r'"safety_verdict"\s*:\s*"\s*block\s*"', re.IGNORECASE)

log = get_logger("agents.critic")

_TAG = tag_stripper("user_request", "must_have")


def sanitize(text: str) -> str:
    return _TAG.sub("", text).strip()


class CriticParams(BaseModel):
    """`params` block for critic in agents.yaml. No defaults: config owns the values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thumbnail_size: int = Field(gt=0)  # which stored thumbnail to send (long side, px)
    # Fallback only, when the thumbnail is missing: shrink the original to `thumbnail_size`.
    jpeg_quality: int = Field(ge=1, le=95)
    retries: int = Field(ge=0, le=1)  # extra attempts after a review with no usable output
    # The most images the vision model takes in ONE request (Groq qwen3.8-27b: 3). More images
    # are reviewed in balanced groups, one call each.
    max_images_per_call: int = Field(ge=1)


# Failures another attempt cannot fix: retrying would only repeat them (or route around a
# provider's refusal). The images stay unchecked and are never delivered.
NO_RETRY = frozenset({"invalid_input", "bad_request", "no_provider", "content_refused"})

# task_type -> rubric prompt (relative to prompts/). Edit and merge add theirs (§10).
RUBRICS = {"create": "critic/critic.md"}


class CriticInputs(BaseModel):
    model_config = ConfigDict(extra="ignore")  # e.g. an enhanced_prompt passed by mistake

    image_ids: list[str] = Field(min_length=1)
    prompt: str
    must_have: list[str] = Field(default_factory=list)
    task_type: str = "create"


def _clean_reason(reason: object) -> str:
    return reason.strip() if isinstance(reason, str) and reason.strip() else DEFAULT_BLOCK_REASON


def split_evenly(count: int, most: int) -> list[list[int]]:
    """Positions 0..count-1 in the fewest groups of at most `most`, as even as possible
    (4 images, most 3 -> [0, 1], [2, 3]; never 3 + 1)."""
    groups = -(-count // most)
    size, extra = divmod(count, groups)
    out: list[list[int]] = []
    start = 0
    for g in range(groups):
        end = start + size + (1 if g < extra else 0)
        out.append(list(range(start, end)))
        start = end
    return out


def says_block(text: str) -> bool:
    """Does an UNREADABLE reply say block? (A readable one is judged per image.) Fail-closed on
    purpose: a stray match drops the review, a miss could deliver an unsafe image."""
    return _BLOCK_TEXT.search(text) is not None


def ranked(reviews: list[ImageReview]) -> list[ImageReview]:
    """Allowed reviews first in rank order (the model's rank, then the mean score, then the
    position), then the blocked ones. Duplicate or out-of-range indexes are the caller's job."""

    def key(r: ImageReview) -> tuple[int, int, int, int]:
        return (
            r.safety_verdict == "block",
            r.rank if r.rank is not None else 10**6,
            -r.quality_score,
            r.index,
        )

    return sorted(reviews, key=key)


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

    async def _review_image(self, image_id: str) -> tuple[bytes, str]:
        """The bytes sent for one image: its stored thumbnail, else the original shrunk."""
        try:
            thumb = await self._artifacts.get_thumbnail(image_id, self.params.thumbnail_size)
            mime = sniff_media_type(thumb)
            if mime is not None:
                return thumb, mime
        except Exception:  # noqa: BLE001 - no thumbnail (or a flaky store): use the original
            log.info("critic.no_thumbnail", image_id=image_id)
        data = await self._artifacts.get(image_id)
        # Decode + resize is CPU work: keep it off the event loop.
        return await asyncio.to_thread(
            downscale, data, self.params.thumbnail_size, self.params.jpeg_quality
        )

    async def _one_call_per_image(
        self,
        auth: AuthContext,
        stop: StoppingCondition,
        system: str,
        contents: list[list[dict[str, Any]]],
        failed_batch: Structured[CriticOutput],
    ) -> Structured[CriticOutput]:
        """One call per image, after the batch failed. All spend, calls and trace records join
        the failed batch's, so the step accounts for everything. Each call reviews one image;
        its entry is re-indexed to the image's real position. An image whose call fails has no
        review (unchecked). An unreadable "block" in any call stops the fallback at once."""
        total = failed_batch
        reviews: list[ImageReview] = []
        for position, content in enumerate(contents, start=1):
            one = await self.call_structured(auth, stop, content, CriticOutput, system=system)
            total.spend = total.spend.plus(one.spend)
            total.calls.extend(one.calls)
            total.records.extend(one.records)
            if any(says_block(c.text) for c in one.calls):
                if one.output is not None:
                    reviews.extend(
                        r.model_copy(update={"index": position}) for r in one.output.images[:1]
                    )
                break  # the caller's block rule decides; do not spend on the rest
            if one.output is not None and one.output.images:
                reviews.append(one.output.images[0].model_copy(update={"index": position}))
        if reviews:
            total.output = CriticOutput(images=reviews)
            total.failure, total.error = None, None
        return total

    async def _review_group(
        self,
        auth: AuthContext,
        stop: StoppingCondition,
        system: str,
        content_for: Callable[[list[tuple[bytes, str]]], list[dict[str, Any]]],
        sent: list[tuple[bytes, str]],
    ) -> tuple[Structured[CriticOutput], bool]:
        """One group of images: a call, its in-step retry, and (for more than one image) the
        per-image fallback. Returns (result, unattributed): `unattributed` is True when a reply
        said "block" and the answer we kept blocks nothing."""
        content = content_for(sent)
        result = await self.call_structured(auth, stop, content, CriticOutput, system=system)
        for _ in range(self.params.retries):
            if result.output is not None or any(says_block(c.text) for c in result.calls):
                break
            if result.failure in NO_RETRY:
                break
            log.info("critic.retry", failure=result.failure)
            again = await self.call_structured(auth, stop, content, CriticOutput, system=system)
            # One step, one account: both attempts' spend, calls and trace records.
            again.spend = result.spend.plus(again.spend)
            again.calls = [*result.calls, *again.calls]
            again.records = [*result.records, *again.records]
            result = again
        # The group gave nothing usable (after its retry): one call per image, so images already
        # paid for are not lost. Not after a refusal, a missing provider or our own bad request
        # (another call would only repeat them), and never after an unreadable "block". Spend
        # stays on this step.
        if (
            result.output is None
            and len(sent) > 1
            and result.failure not in NO_RETRY
            and not any(says_block(c.text) for c in result.calls)
        ):
            log.warning("critic.per_image_fallback", images=len(sent), failure=result.failure)
            result = await self._one_call_per_image(
                auth, stop, system, [content_for([item]) for item in sent], result
            )
        said_block = any(says_block(c.text) for c in result.calls)
        blocks_something = result.output is not None and any(
            r.safety_verdict == "block" for r in result.output.images
        )
        return result, said_block and not blocks_something

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
        rubric = RUBRICS.get(args.task_type)
        if rubric is None:
            return self.failed(f"no rubric for task {args.task_type!r}", Spend(), "invalid_input")
        ids = list(dict.fromkeys(args.image_ids))  # one entry per image, order kept

        try:
            sent = [await self._review_image(image_id) for image_id in ids]
        except NotAnImage as err:
            return self.failed(str(err), Spend(), "bad_image")
        except Exception as err:  # the store is a backend port: any failure is "storage"
            log.exception("critic.load_failed")
            return self.failed(f"could not load image: {type(err).__name__}", Spend(), "storage")

        must_have = [m for m in (sanitize(item) for item in args.must_have) if m]
        system = render(load_prompt(PROMPTS_DIR / rubric))

        def content_for(picked: list[tuple[bytes, str]]) -> list[dict[str, Any]]:
            """OpenAI-style multimodal content: the text, then a label and an image per image."""
            text = render(
                load_prompt(INPUT_TEMPLATE),
                prompt=user_prompt,
                must_have="\n".join(f"- {m}" for m in must_have) or NONE_GIVEN,
                count=str(len(picked)),
            )
            parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
            for number, (data, mime) in enumerate(picked, start=1):
                parts.append({"type": "text", "text": f"Image {number}"})
                parts.append({"type": "image_url", "image_url": {"url": data_url(data, mime)}})
            return parts

        # The vision model takes a limited number of images per request (Groq qwen3.8-27b: 3,
        # measured 2026-11-20: a 4-image request is an HTTP 400). Split into balanced groups, one
        # call each, run together. Within a group the model ranks; across groups the scores do.
        groups = split_evenly(len(sent), self.params.max_images_per_call)
        outcomes = await asyncio.gather(
            *[
                self._review_group(auth, stop, system, content_for, [sent[i] for i in group])
                for group in groups
            ]
        )
        result: Structured[CriticOutput] = Structured(output=None, spend=Spend())
        merged: list[ImageReview] = []
        for group, (got, unattributed) in zip(groups, outcomes, strict=True):
            result.spend = result.spend.plus(got.spend)
            result.calls.extend(got.calls)
            result.records.extend(got.records)
            if unattributed:
                # An unreadable "block" cannot be attributed to an image, and a repair, retry or
                # fallback must never turn it into "allow": fail the WHOLE step (safety first).
                log.warning("critic.unattributed_block", calls=len(result.calls), images=len(ids))
                return self.failed(UNATTRIBUTED_BLOCK, result.spend, "invalid_output")
            if got.output is None:
                result.error, result.failure = got.error, got.failure
                continue
            for review in got.output.images:
                if 1 <= review.index <= len(group):  # position inside the group -> global
                    merged.append(
                        review.model_copy(
                            update={
                                "index": group[review.index - 1] + 1,
                                "rank": review.rank if len(groups) == 1 else None,
                            }
                        )
                    )
        if not merged:
            return self.failed(
                result.error or "critic produced no output", result.spend, result.failure
            )
        result.output = CriticOutput(images=merged)
        # One review per image: drop unknown or repeated indexes, keep the first.
        seen: dict[int, ImageReview] = {}
        for review in result.output.images:
            if 1 <= review.index <= len(ids):
                seen.setdefault(review.index, review)
        ordered = ranked(list(seen.values()))
        reviews: list[dict[str, Any]] = []
        position = 0
        for review in ordered:
            image_id = ids[review.index - 1]
            entry: dict[str, Any] = {"image_id": image_id, "verdict": review.safety_verdict}
            if review.safety_verdict == "block":
                reason = _clean_reason(review.safety_reason)
                log.warning("critic.blocked", reason=reason, image_id=image_id)
                try:
                    await self._artifacts.mark_orphaned(image_id, ORPHAN_REASON)
                except Exception:  # never let bookkeeping turn a block into anything else
                    log.exception("critic.mark_orphaned_failed")
                entry["safety_reason"] = reason
            else:
                position += 1
                entry.update(
                    rank=position,
                    scores=review.scores,
                    quality_score=review.quality_score,
                    critique=review.critique.strip(),
                )
            reviews.append(entry)
        if not reviews:
            return self.failed("critic reviewed none of the images", result.spend, "invalid_output")
        return self.envelope(Status.DONE, {"reviews": reviews}, DEFAULT_SUMMARY, result.spend)
