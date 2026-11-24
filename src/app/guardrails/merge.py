"""Strictest answer wins. Spec section 8."""

from app.guardrails.policy import decision_rank, tier_of, tier_rank
from app.guardrails.schemas import Category, Decision, GuardVerdict


def _strictness(v: GuardVerdict) -> tuple[int, int, float]:
    """Decision first, then tier, then confidence (None counts as lowest)."""
    conf = v.confidence if v.confidence is not None else -1.0
    return (decision_rank(v.decision), tier_rank(tier_of(v.category)), conf)


def strictest(verdicts: list[GuardVerdict]) -> GuardVerdict:
    if not verdicts:
        raise ValueError("no verdicts to merge")
    return max(verdicts, key=_strictness)


def merge(verdicts: list[GuardVerdict]) -> tuple[Decision, Category, float | None]:
    """Final decision, category and confidence. APPROVE only if every verdict is APPROVE."""
    best = strictest(verdicts)
    if best.decision is Decision.APPROVE:
        return Decision.APPROVE, Category.SAFE, best.confidence
    return best.decision, best.category, best.confidence
