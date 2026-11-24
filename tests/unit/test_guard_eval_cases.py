"""The eval dataset is well formed. (The eval itself needs real model calls.)"""

from collections import Counter

from evals.run_guardrail_eval import load_cases

from app.guardrails.schemas import Category


def test_dataset_is_well_formed() -> None:
    cases = load_cases()
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert c["expect"] in {"block", "allow"}
        assert c["prompt"].strip()
        if c["expect"] == "block":
            assert Category(c["category"]) is not Category.SAFE


def test_dataset_covers_required_groups() -> None:
    groups = Counter(c["group"] for c in load_cases())
    for needed in ("redteam_csam", "redteam_ncii", "false_positive", "urdu"):
        assert groups[needed] >= 5
    urdu = [c["prompt"] for c in load_cases("urdu")]
    assert any("؀" <= ch <= "ۿ" for p in urdu for ch in p)  # Urdu script
    assert any(p.isascii() for p in urdu)  # Roman Urdu
