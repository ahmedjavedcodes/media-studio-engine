import pytest

from app.guardrails.policy import CATEGORY_TIER, S_CODE_CATEGORY, decide, tier_of
from app.guardrails.schemas import Category, Decision, Tier


def test_every_category_has_a_tier() -> None:
    assert set(CATEGORY_TIER) == set(Category)


def test_s_code_map_matches_spec() -> None:
    assert S_CODE_CATEGORY["S4"] is Category.CSAM
    assert S_CODE_CATEGORY["S12"] is Category.NSFW_EXPLICIT
    assert {S_CODE_CATEGORY[c] for c in ("S1", "S2", "S6", "S13", "S14")} == {Category.OTHER_UNSAFE}
    assert S_CODE_CATEGORY["S7"] is S_CODE_CATEGORY["S8"] is Category.PII_WATERMARK
    assert len(S_CODE_CATEGORY) == 14


@pytest.mark.parametrize(
    ("category", "p", "expected"),
    [
        (Category.CSAM, 0.99, Decision.INSTANT_REJECT),
        (Category.CSAM, 0.85, Decision.INSTANT_REJECT),
        (Category.CSAM, 0.84, Decision.STANDARD_REJECT),
        (Category.NCII_UNDRESSING, 0.60, Decision.STANDARD_REJECT),
        (Category.NCII_UNDRESSING, 0.59, Decision.APPROVE),
        (Category.NSFW_EXPLICIT, 0.99, Decision.STANDARD_REJECT),
        (Category.NSFW_EXPLICIT, 0.60, Decision.STANDARD_REJECT),
        (Category.NSFW_EXPLICIT, 0.10, Decision.APPROVE),
        (Category.SAFE, 1.0, Decision.APPROVE),
    ],
)
def test_decide(category: Category, p: float, expected: Decision) -> None:
    assert decide(category, p) is expected


def test_tiers() -> None:
    assert tier_of(Category.TERRORISM_WEAPONS) is Tier.CRITICAL
    assert tier_of(Category.GORE_SELF_HARM) is Tier.SEVERE
