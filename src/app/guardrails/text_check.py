"""Jev text check. Spec section 7.2. One yes/no question per category, yes = violation."""

import math
import time
from functools import lru_cache
from typing import Any

import yaml

from app.contracts.ports import JevPort
from app.core.config import prompts_dir
from app.core.errors import InvalidModelOutput
from app.guardrails.clean import CleanText
from app.guardrails.policy import decide, decision_rank, tier_of, tier_rank
from app.guardrails.schemas import Category, Decision, GuardVerdict

# question key -> category
QUESTION_CATEGORY: dict[str, Category] = {
    "csam": Category.CSAM,
    "ncii_undressing": Category.NCII_UNDRESSING,
    "gore_self_harm": Category.GORE_SELF_HARM,
    "terrorism_weapons": Category.TERRORISM_WEAPONS,
    "nsfw_explicit": Category.NSFW_EXPLICIT,
    "celebrity_likeness": Category.CELEBRITY_LIKENESS,
    "hate_harassment": Category.HATE_HARASSMENT,
    "prompt_injection": Category.PROMPT_INJECTION,
    "pii_watermark": Category.PII_WATERMARK,
}


@lru_cache
def load_questions() -> dict[str, Any]:
    raw = yaml.safe_load((prompts_dir() / "guardrail" / "jev_questions.yaml").read_text("utf-8"))
    common = str(raw["common"]).strip()
    questions: dict[str, Any] = {}
    for key, q in raw["questions"].items():
        questions[key] = {
            "type": "noul",
            "instructions": f"{common}\n\n{str(q['instructions']).strip()}",
            "criteria": {"true": str(q["violation"]).strip(), "false": str(q["allowed"]).strip()},
        }
    missing = set(QUESTION_CATEGORY) - set(questions)
    if missing:
        raise RuntimeError(f"jev_questions.yaml is missing: {sorted(missing)}")
    return questions


def read_scores(answers: dict[str, Any]) -> dict[str, float]:
    """Pull `answers[key].noul` for every category. Missing or invalid -> fail closed."""
    scores: dict[str, float] = {}
    for key in QUESTION_CATEGORY:
        answer = answers.get(key)
        value = answer.get("noul") if isinstance(answer, dict) else None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise InvalidModelOutput(f"jev answer '{key}' is missing or not a number")
        if math.isnan(value) or not 0.0 <= value <= 1.0:
            raise InvalidModelOutput(f"jev answer '{key}' is out of range")
        scores[key] = float(value)
    return scores


def verdict_from_scores(scores: dict[str, float], *, model: str, latency_ms: int) -> GuardVerdict:
    """The strictest category wins: decision, then tier, then score."""
    best: tuple[int, int, float, Category, Decision] | None = None
    for key, p in scores.items():
        cat = QUESTION_CATEGORY[key]
        dec = decide(cat, p)
        rank = (decision_rank(dec), tier_rank(tier_of(cat)), p)
        if best is None or rank > best[:3]:
            best = (*rank, cat, dec)
    assert best is not None
    _, _, p, cat, dec = best
    if dec is Decision.APPROVE:
        return GuardVerdict(
            source="jev",
            decision=dec,
            category=Category.SAFE,
            confidence=max(scores.values()),
            reason="no category over threshold",
            scores=scores,
            model=model,
            latency_ms=latency_ms,
        )
    return GuardVerdict(
        source="jev",
        decision=dec,
        category=cat,
        confidence=p,
        reason=f"{cat.value} p={p:.2f}",
        scores=scores,
        model=model,
        latency_ms=latency_ms,
    )


async def check_text(
    jev: JevPort, text: CleanText, *, has_image: bool, timeout: float
) -> GuardVerdict:
    """One Jev attempt. Raises ProviderCallFailed or InvalidModelOutput (caller fails closed)."""
    state = {
        "prompt_raw": text.prompt_raw,
        "prompt_clean": text.prompt_clean,
        "has_image": has_image,
    }
    start = time.monotonic()
    answers = await jev.ask(state=state, questions=load_questions(), timeout=timeout)
    scores = read_scores(answers)
    return verdict_from_scores(
        scores, model=jev.model, latency_ms=int((time.monotonic() - start) * 1000)
    )
