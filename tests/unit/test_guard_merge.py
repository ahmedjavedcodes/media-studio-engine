from app.guardrails.merge import merge, strictest
from app.guardrails.schemas import Category, Decision, GuardVerdict


def v(
    decision: Decision, category: Category, conf: float | None, source: str = "jev"
) -> GuardVerdict:
    return GuardVerdict(
        source=source,  # type: ignore[arg-type]
        decision=decision,
        category=category,
        confidence=conf,
        reason="",
        model="m",
        latency_ms=1,
    )


def test_approve_only_if_all_approve() -> None:
    a = v(Decision.APPROVE, Category.SAFE, 0.1)
    b = v(Decision.APPROVE, Category.SAFE, 0.2, "vision")
    assert merge([a, b])[0] is Decision.APPROVE
    assert merge([a, b])[1] is Category.SAFE


def test_instant_beats_standard_beats_approve() -> None:
    ok = v(Decision.APPROVE, Category.SAFE, 0.1)
    std = v(Decision.STANDARD_REJECT, Category.NSFW_EXPLICIT, 0.99, "vision")
    inst = v(Decision.INSTANT_REJECT, Category.CSAM, 0.86)
    assert merge([ok, std])[:2] == (Decision.STANDARD_REJECT, Category.NSFW_EXPLICIT)
    assert merge([ok, std, inst])[:2] == (Decision.INSTANT_REJECT, Category.CSAM)


def test_tie_goes_to_higher_tier_then_confidence() -> None:
    nsfw = v(Decision.STANDARD_REJECT, Category.NSFW_EXPLICIT, 0.99)
    gore = v(Decision.STANDARD_REJECT, Category.GORE_SELF_HARM, 0.65, "vision")
    assert strictest([nsfw, gore]) is gore
    a = v(Decision.STANDARD_REJECT, Category.NSFW_EXPLICIT, 0.7)
    b = v(Decision.STANDARD_REJECT, Category.NSFW_EXPLICIT, 0.9, "vision")
    assert strictest([a, b]) is b
    c = v(Decision.STANDARD_REJECT, Category.NSFW_EXPLICIT, None, "vision")
    assert strictest([c, a]) is a
