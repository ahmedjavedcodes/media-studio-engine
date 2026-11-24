"""Jev text check. Spec section 7.2. One yes/no question per category, yes = violation."""

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yaml

from app.contracts.ports import JevPort
from app.core.config import prompts_dir
from app.core.errors import InvalidModelOutput
from app.guardrails.clean import CleanText
from app.guardrails.policy import combo_scores, decide, decision_rank, tier_of, tier_rank
from app.guardrails.schemas import Category, Decision, GuardVerdict


@dataclass(frozen=True)
class QuestionSet:
    """prompts/guardrail/jev_questions.yaml, ready to send."""

    context: str
    questions: dict[str, dict[str, Any]]  # question id -> Jev noul question
    category_of: dict[str, Category | None]  # None = helper used only by a combo rule


@lru_cache
def load_questions() -> QuestionSet:
    raw = yaml.safe_load((prompts_dir() / "guardrail" / "jev_questions.yaml").read_text("utf-8"))
    questions: dict[str, dict[str, Any]] = {}
    category_of: dict[str, Category | None] = {}
    for key, q in raw["questions"].items():
        question: dict[str, Any] = {"type": "noul", "instructions": str(q["instructions"]).strip()}
        criteria = {
            side: str(q[name]).strip()
            for side, name in (("true", "violation"), ("false", "allowed"))
            if q.get(name)
        }
        if criteria:  # omit criteria when both are missing
            question["criteria"] = criteria
        questions[key] = question
        category_of[key] = Category(q["category"]) if q.get("category") else None
    return QuestionSet(str(raw["context"]).strip(), questions, category_of)


def read_scores(answers: dict[str, Any]) -> dict[str, float]:
    """Pull `answers[id].noul` for every question. Missing or invalid -> fail closed."""
    scores: dict[str, float] = {}
    for key in load_questions().questions:
        answer = answers.get(key)
        value = answer.get("noul") if isinstance(answer, dict) else None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise InvalidModelOutput(f"jev answer '{key}' is missing or not a number")
        if math.isnan(value) or not 0.0 <= value <= 1.0:
            raise InvalidModelOutput(f"jev answer '{key}' is out of range")
        scores[key] = float(value)
    return scores


def category_scores(scores: dict[str, float]) -> dict[Category, float]:
    """Highest question score per category, then the combo rules."""
    out: dict[Category, float] = {}
    for key, p in scores.items():
        cat = load_questions().category_of[key]
        if cat is not None:
            out[cat] = max(out.get(cat, 0.0), p)
    for cat, p in combo_scores(scores).items():
        out[cat] = max(out.get(cat, 0.0), p)
    return out


def verdict_from_scores(scores: dict[str, float], *, model: str, latency_ms: int) -> GuardVerdict:
    """The strictest category wins: decision, then tier, then score."""
    best: tuple[int, int, float, Category, Decision] | None = None
    for cat, p in category_scores(scores).items():
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
    qs = load_questions()
    state = {
        "context": qs.context,
        "prompt_raw": text.prompt_raw,
        "prompt_clean": text.prompt_clean,
        "has_image": has_image,
    }
    start = time.monotonic()
    answers = await jev.ask(state=state, questions=qs.questions, timeout=timeout)
    scores = read_scores(answers)
    return verdict_from_scores(
        scores, model=jev.model, latency_ms=int((time.monotonic() - start) * 1000)
    )
