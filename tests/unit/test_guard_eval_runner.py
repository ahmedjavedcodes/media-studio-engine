"""The eval runner's helpers and dataset. (The eval itself makes real model calls.)"""

from collections import Counter

import pytest
from evals.run_guardrail_eval import DEFAULT_DATASET, NullJev, judge, load_dataset, near_threshold

from app.guardrails.text_check import load_questions


def test_dataset_is_well_formed() -> None:
    data = load_dataset(DEFAULT_DATASET)
    ids = [c["id"] for sec in data.values() for c in sec]
    assert len(ids) == len(set(ids))
    groups = Counter({sec: len(cases) for sec, cases in data.items()})
    assert groups["text_block"] == 18 and groups["text_allow"] == 11 and groups["image_text"] == 12
    fixtures = DEFAULT_DATASET.parent.parent / "fixtures" / "guardrail"
    for case in data["image_text"] + data["input_checks"]:
        for name in case.get("images", []):
            assert (fixtures / name).exists(), f"{case['id']}: missing fixture {name}"


def test_near_threshold_lists_scores_within_0_10() -> None:
    found = near_threshold({"NSFW_EXPLICIT": 0.99, "CSAM": 0.55, "HATE": 0.80, "X": 0.3}, 0.9)
    assert {(f["what"], f["threshold"]) for f in found} == {
        ("CSAM", 0.60),
        ("HATE", 0.85),
        ("vision:unsafe", 0.85),
    }
    assert near_threshold({"A": 0.2, "B": 0.72}, None) == []


@pytest.mark.parametrize(
    ("expect", "outcome", "category", "must_block", "verdict"),
    [
        ({"decision": "APPROVE"}, "APPROVE", None, False, "PASS"),
        ({"decision": "APPROVE"}, "STANDARD_REJECT", "NSFW_EXPLICIT", False, "FALSE+"),
        ({"decision": "APPROVE"}, "UNAVAILABLE_503", None, False, "FAIL(503)"),
        (
            {"decision": "INSTANT_REJECT", "category": "CSAM"},
            "INSTANT_REJECT",
            "CSAM",
            True,
            "PASS",
        ),
        (
            {"decision": "INSTANT_REJECT", "category": "CSAM"},
            "STANDARD_REJECT",
            "CSAM",
            True,
            "PASS(block)",
        ),
        ({"decision": "STANDARD_REJECT", "category": "CSAM"}, "APPROVE", None, False, "MISS"),
        ({"decision": "BAD_INPUT"}, "BAD_INPUT", None, False, "PASS"),
    ],
)
def test_judge(
    expect: dict, outcome: str, category: str | None, must_block: bool, verdict: str
) -> None:  # type: ignore[type-arg]
    case = {"expect": expect, "must_block": must_block}
    assert judge(case, outcome, category) == verdict


async def test_null_jev_answers_zero_for_every_question() -> None:
    qs = load_questions().questions
    answers = await NullJev().ask(state={}, questions=qs, timeout=1)
    assert set(answers) == set(qs) and all(a["noul"] == 0.0 for a in answers.values())
