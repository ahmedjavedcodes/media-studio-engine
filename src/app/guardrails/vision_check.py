"""Llama Guard 4 check on images + cleaned text. Spec section 7.4.

Llama Guard returns text, not JSON: `safe`, or `unsafe` then codes like `S1,S12`.
"""

import math
import re
import time

from app.contracts.ports import VisionPort
from app.core.errors import InvalidModelOutput
from app.guardrails.image_prep import PreparedImage
from app.guardrails.merge import strictest
from app.guardrails.policy import S_CODE_CATEGORY, decide, tier_of, tier_rank
from app.guardrails.schemas import Category, Decision, GuardVerdict

_CODE = re.compile(r"^S\d{1,2}$")


def parse_guard_text(text: str) -> tuple[bool, list[str]]:
    """Return (is_unsafe, codes). Anything that is not `safe` / `unsafe` fails closed."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        raise InvalidModelOutput("vision model returned no text")
    head = lines[0].lower()
    if head == "safe":
        return False, []
    if head != "unsafe":
        raise InvalidModelOutput("vision model returned an unknown answer")
    codes: list[str] = []
    if len(lines) > 1:
        codes = [c.strip().upper() for c in lines[1].split(",") if c.strip()]
        if not all(_CODE.match(c) for c in codes):
            raise InvalidModelOutput("vision model returned bad hazard codes")
    return True, codes


def p_unsafe(pairs: list[tuple[str, float]] | None) -> float | None:
    """Probability of `unsafe` among the first-token candidates. None if there are no logprobs."""
    if not pairs:
        return None
    unsafe = sum(math.exp(lp) for tok, lp in pairs if tok.strip().lower() == "unsafe")
    safe = sum(math.exp(lp) for tok, lp in pairs if tok.strip().lower() == "safe")
    total = unsafe + safe
    return unsafe / total if total > 0 else None


def category_for(codes: list[str]) -> Category:
    """Highest tier wins when there are several codes. No or unknown codes: OTHER_UNSAFE."""
    cats = [S_CODE_CATEGORY.get(c, Category.OTHER_UNSAFE) for c in codes] or [Category.OTHER_UNSAFE]
    return max(cats, key=lambda c: tier_rank(tier_of(c)))


def verdict_from_output(
    text: str,
    pairs: list[tuple[str, float]] | None,
    *,
    model: str,
    latency_ms: int,
) -> GuardVerdict:
    unsafe, codes = parse_guard_text(text)
    p = p_unsafe(pairs)
    scores = {"unsafe": p} if p is not None else {}

    def make(
        decision: Decision,
        category: Category,
        confidence: float | None,
        reason: str,
        needs_review: bool = False,
    ) -> GuardVerdict:
        return GuardVerdict(
            source="vision",
            decision=decision,
            category=category,
            confidence=confidence,
            reason=reason,
            scores=scores,
            needs_review=needs_review,
            raw_output=text,
            model=model,
            latency_ms=latency_ms,
        )

    if not unsafe:
        return make(Decision.APPROVE, Category.SAFE, p, "llama guard: safe")
    cat = category_for(codes)
    shown = ",".join(codes)
    if p is None:  # no score: reject, but a human should look
        return make(
            Decision.STANDARD_REJECT, cat, None, f"llama guard: unsafe {shown} (no logprobs)", True
        )
    decision = decide(cat, p)
    if decision is Decision.APPROVE:
        reason = f"llama guard: unsafe {shown} below threshold p={p:.2f}"
        return make(decision, Category.SAFE, p, reason)
    return make(decision, cat, p, f"llama guard: unsafe {shown} p={p:.2f}")


async def check_vision_batch(
    vision: VisionPort, prompt_clean: str, images: list[PreparedImage], *, timeout: float
) -> GuardVerdict:
    """One Llama Guard call for up to 5 images. Raises on failure (caller fails closed)."""
    start = time.monotonic()
    raw = await vision.classify(
        text=prompt_clean or "(no text)",
        image_urls=[i.data_url for i in images],
        timeout=timeout,
    )
    return verdict_from_output(
        raw.text,
        raw.first_token_logprobs,
        model=vision.model,
        latency_ms=int((time.monotonic() - start) * 1000),
    )


def combine_batches(results: list[GuardVerdict | BaseException]) -> GuardVerdict:
    """Merge the batch results into one vision verdict. A reject beats a failed batch."""
    verdicts = [r for r in results if isinstance(r, GuardVerdict)]
    errors = [r for r in results if isinstance(r, BaseException)]
    rejects = [v for v in verdicts if v.decision is not Decision.APPROVE]
    if not rejects and errors:
        raise errors[0]
    winner = strictest(rejects or verdicts)
    return winner.model_copy(
        update={
            "needs_review": any(v.needs_review for v in rejects),
            "latency_ms": max(v.latency_ms for v in verdicts),
        }
    )
