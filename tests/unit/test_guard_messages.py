import pytest

from app.guardrails.messages import blocked_by, build_reject_body, message_for, tip_for
from app.guardrails.schemas import Category, Decision, GuardDecision, GuardVerdict
from app.guardrails.text_check import QUESTION_CATEGORY


def verdict(source: str, decision: Decision, cat: Category) -> GuardVerdict:
    return GuardVerdict(
        source=source,  # type: ignore[arg-type]
        decision=decision,
        category=cat,
        confidence=0.9,
        reason="SECRET INTERNAL REASON",
        scores={"csam": 0.93},
        model="m",
        latency_ms=1,
    )


def decision(
    dec: Decision, cat: Category, sources: tuple[str, ...] = ("jev",), **kw: object
) -> GuardDecision:
    return GuardDecision(
        decision=dec,
        category=cat,
        confidence=0.9,
        verdicts=[verdict(s, dec, cat) for s in sources],
        **kw,  # type: ignore[arg-type]
    )


def test_every_reject_category_has_message() -> None:
    for cat in Category:
        if cat is not Category.SAFE:
            assert message_for(cat)
    assert tip_for(Category.CSAM) is None
    assert tip_for(Category.HATE_HARASSMENT) is None
    assert tip_for(Category.NSFW_EXPLICIT) == "Try a fully clothed version."
    assert set(QUESTION_CATEGORY.values()) <= set(Category)


@pytest.mark.parametrize(
    ("sources", "expected"),
    [(("jev",), "prompt"), (("vision",), "image"), (("jev", "vision"), "both")],
)
def test_blocked_by(sources: tuple[str, ...], expected: str) -> None:
    assert (
        blocked_by(decision(Decision.STANDARD_REJECT, Category.NSFW_EXPLICIT, sources)) == expected
    )


def test_standard_body_has_retry_and_no_internals() -> None:
    d = decision(
        Decision.STANDARD_REJECT,
        Category.NSFW_EXPLICIT,
        account_action="cooldown_1m",
        retry_after=60,
    )
    body = build_reject_body(d, strike_limit=5)
    assert body == {
        "code": "content_blocked",
        "category": "NSFW_EXPLICIT",
        "blocked_by": "prompt",
        "message": "Your request asks for nudity or sexual content.",
        "tip": "Try a fully clothed version.",
        "warning": False,
        "retry_after": 60,
        "retry_text": "You can try again in 1 minute.",
    }
    text = str(body)
    assert "SECRET" not in text and "0.9" not in text and "scores" not in text


def test_retry_text_plural_and_long() -> None:
    d = decision(Decision.STANDARD_REJECT, Category.NSFW_EXPLICIT, retry_after=120)
    assert build_reject_body(d, strike_limit=5)["retry_text"] == "You can try again in 2 minutes."
    d = decision(Decision.STANDARD_REJECT, Category.NSFW_EXPLICIT, retry_after=1800)
    assert build_reject_body(d, strike_limit=5)["retry_text"] == "You can try again in 30 minutes."


def test_instant_body_has_strike_text() -> None:
    d = decision(
        Decision.INSTANT_REJECT, Category.CSAM, ("vision",), account_action="warn", strike_count=2
    )
    body = build_reject_body(d, strike_limit=5)
    assert body["warning"] is True
    assert body["blocked_by"] == "image"
    assert body["strike_count"] == 2
    assert body["strike_text"] == "Strike 2 of 5. At 5 strikes your account will be banned."
    assert body["banned"] is False
    assert "retry_after" not in body
    assert body["tip"] is None


def test_fifth_strike_says_banned() -> None:
    d = decision(Decision.INSTANT_REJECT, Category.CSAM, account_action="ban", strike_count=5)
    body = build_reject_body(d, strike_limit=5)
    assert body["strike_text"] == "Strike 5 of 5. Your account has been banned."
    assert body["banned"] is True
