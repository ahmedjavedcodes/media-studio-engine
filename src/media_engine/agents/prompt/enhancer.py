"""Prompt Enhancer: turns the user's request into a detailed image prompt plus testable
`must_have` requirements taken only from the user's own words.

inputs:
    prompt:   str             the user's raw request (required)
    settings: dict[str, str]  already-chosen values, e.g. {"aspect_ratio": "16:9"} (optional)
    feedback: str             critique of a previous attempt, for the re_enhance lever (optional)

output (envelope):
    enhanced_prompt, must_have, must_have_dropped, missing_info (list of {kind, question}),
    text_wanted, user_prompt

Returns DONE, or NEEDS_INPUT when `missing_info` is non-empty (the Image Grand Agent resolves
preset kinds from defaults before escalating). Never talks to the user.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from media_engine.agents.base import Agent, AgentConfig
from media_engine.contracts import AuthContext, Envelope, Goal, Spend, Status, StoppingCondition
from media_engine.core.logging import get_logger
from media_engine.core.prompts import PROMPTS_DIR, load_prompt, render
from media_engine.gateway.llm_gateway import LLMGateway
from media_engine.ports.event_sink import EventSink
from media_engine.schemas import EnhancerOutput, MissingItem

INPUT_TEMPLATE = PROMPTS_DIR / "image_create" / "prompt_enhancer.input.md"
DEFAULT_SUMMARY = "Prepared a detailed description of your image."
NONE_GIVEN = "none"

log = get_logger("agents.prompt_enhancer")


class EnhancerParams(BaseModel):
    """`params` block for prompt_enhancer in agents.yaml. No defaults: config owns the values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grounding_min_fraction: float = Field(gt=0.0, le=1.0)
    stem_len: int = Field(gt=0)
    min_word_len: int = Field(gt=0)


# -- sanitising --------------------------------------------------------------------------------

# Every user-supplied value is wrapped in one of these tag blocks. Strip copies of the tags from
# the values themselves so no value can close its block early and smuggle instructions after it.
_TAG = re.compile(r"</?\s*(?:user_request|settings|feedback)\s*>", re.IGNORECASE)


def sanitize(text: str) -> str:
    return _TAG.sub("", text).strip()


def format_settings(settings: dict[str, Any]) -> str:
    lines = []
    for key, value in settings.items():
        k, v = sanitize(str(key)), sanitize(str(value))
        if k and v:
            lines.append(f"{k}: {v}")
    return "\n".join(lines) or NONE_GIVEN


# -- must_have grounding -----------------------------------------------------------------------
#
# An item is kept only if enough of its word stems appear in the user's request. This catches
# the enhancer promoting its own additions ("golden hour lighting") to requirements. Crude on
# purpose: light suffix stripping plus a fixed-length prefix, no stemming library.

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
# Han, Hiragana, Katakana: written without spaces, so each character is its own token.
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")
_SUFFIXES = ("ing", "ed", "s")
_STOPWORDS = frozenset(
    [
        "the",
        "and",
        "with",
        "for",
        "that",
        "this",
        "from",
        "into",
        "onto",
        "over",
        "under",
        "some",
        "any",
        "are",
        "has",
        "have",
        "its",
    ]
)


def _strip_suffix(word: str, min_len: int) -> str:
    for suffix in _SUFFIXES:
        if suffix == "s" and word.endswith("ss"):
            continue  # glass, dress
        if word.endswith(suffix) and len(word) - len(suffix) >= min_len:
            return word[: -len(suffix)]
    return word


def stems(text: str, params: EnhancerParams) -> set[str]:
    out: set[str] = set()
    for word in _WORD.findall(text.lower()):
        cjk = _CJK.findall(word)
        if cjk:
            out.update(cjk)
            continue
        if len(word) < params.min_word_len or word in _STOPWORDS:
            continue
        out.add(_strip_suffix(word, params.min_word_len)[: params.stem_len])
    return out


def grounded_fraction(item: str, user_prompt: str, params: EnhancerParams) -> float:
    if item.strip().lower() in user_prompt.lower():
        return 1.0
    item_stems = stems(item, params)
    if not item_stems:
        return 0.0
    return len(item_stems & stems(user_prompt, params)) / len(item_stems)


def is_grounded(item: str, user_prompt: str, params: EnhancerParams) -> bool:
    return grounded_fraction(item, user_prompt, params) >= params.grounding_min_fraction


def ground_must_have(
    items: list[str], user_prompt: str, params: EnhancerParams
) -> tuple[list[str], list[str]]:
    """Split into (kept, dropped). `kept` may be empty: the Critic always also scores against
    the user's raw prompt, so nothing is invented to fill the gap."""
    kept: list[str] = []
    dropped: list[str] = []
    for item in items:
        (kept if is_grounded(item, user_prompt, params) else dropped).append(item)
    return kept, dropped


def drop_already_set(missing: list[MissingItem], settings: dict[str, Any]) -> list[MissingItem]:
    """Never re-request a value that settings already supply (backs up the prompt rule)."""
    return [m for m in missing if m.kind == "question" or m.kind not in settings]


def describe_missing(missing: list[MissingItem]) -> str:
    return "; ".join(m.question if m.kind == "question" and m.question else m.kind for m in missing)


class PromptEnhancer(Agent):
    def __init__(self, config: AgentConfig, gateway: LLMGateway | None, events: EventSink) -> None:
        super().__init__(config, gateway, events)
        # Validate the params block at construction, so bad config fails at startup.
        self.params = EnhancerParams.model_validate(config.model_policy.params)

    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope:
        params = self.params
        raw = inputs.get("prompt")
        user_prompt = sanitize(raw) if isinstance(raw, str) else ""
        if not user_prompt:
            return self.failed("no prompt given", Spend())

        settings: dict[str, Any] = dict(inputs.get("settings") or {})
        feedback_raw = inputs.get("feedback")
        feedback = sanitize(feedback_raw) if isinstance(feedback_raw, str) else ""
        content = render(
            load_prompt(INPUT_TEMPLATE),
            prompt=user_prompt,
            settings=format_settings(settings),
            feedback=feedback or NONE_GIVEN,
        )

        result = await self.call_structured(auth, stop, content, EnhancerOutput)
        if result.output is None:
            return self.failed(result.error or "enhancer produced no output", result.spend)
        out = result.output

        must_have, dropped = ground_must_have(out.must_have, user_prompt, params)
        if dropped:
            log.info("enhancer.must_have_dropped", dropped=len(dropped), kept=len(must_have))
        missing = drop_already_set(out.missing_info, settings)

        output: dict[str, Any] = {
            "enhanced_prompt": out.enhanced_prompt,
            "must_have": must_have,
            "must_have_dropped": dropped,
            "missing_info": [m.model_dump() for m in missing],
            "text_wanted": out.text_wanted,
            "user_prompt": user_prompt,
        }
        summary = out.summary.strip() or DEFAULT_SUMMARY
        if missing:
            return self.envelope(
                Status.NEEDS_INPUT,
                output,
                summary,
                result.spend,
                reason="missing: " + describe_missing(missing),
            )
        return self.envelope(Status.DONE, output, summary, result.spend)
