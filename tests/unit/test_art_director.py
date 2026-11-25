"""The Art Director, the ShotPlan, the look library, the plan store and the rules. No provider."""

import json
from pathlib import Path
from typing import Any

import pytest

from app.agents.image_grand.workers import art_director as ad
from app.agents.image_grand.workers import plan_rules as rules
from app.agents.image_grand.workers.shot_plan import Question, ShotPlan
from app.agents.image_grand.workers.subject_type import guess_subject_type
from app.core.errors import BadInput, InvalidModelOutput, ProviderCallFailed
from app.gateway.looks import get_looks, load_looks
from app.local.plan_store import PlanStore
from tests.unit.create_helpers import FakeLLM, good_plan, missing

LOOKS = get_looks()
SEEDS = json.loads((Path(ad.__file__).parents[3] / "local" / "seed_plans.json").read_text())


# ---- the ShotPlan -------------------------------------------------------------------------------


def test_the_plan_accepts_the_loose_shapes_weaker_models_write() -> None:
    p = ShotPlan.model_validate(
        {
            "intent": "product",
            "light": "warm sun from the left",
            "camera": "85mm",
            "text": "",
            "details": "chipped rim",
            "props": list("abcde"),
            "palette": ["a", "b", "c", "d"],
            "depth": None,
        }
    )
    assert p.intent.subject_type == "product" and p.light.quality == "warm sun from the left"
    assert p.camera.lens == "85mm" and p.text.exact is None and not p.wants_text
    assert p.subject_details == ["chipped rim"] and p.props == ["a", "b", "c"]  # at most 3
    assert p.palette == ["a", "b", "c"] and p.depth.mid == ""


def test_text_needs_exact_words_or_it_is_none() -> None:
    assert ShotPlan.model_validate({"text": {"exact": " ", "placement": "top"}}).text.exact is None
    plan = ShotPlan.model_validate({"text": {"exact": "FRESH ROAST", "placement": "top"}})
    assert plan.wants_text and plan.text.exact == "FRESH ROAST"


def test_compact_drops_empty_fields_and_the_questions() -> None:
    compact = missing().compact()
    assert "missing" not in compact and "text" not in compact and compact["subject"]
    assert good_plan().light_text().startswith("window daylight, from the left")
    assert "foreground: marble table" in good_plan().depth_text()


def test_the_plan_round_trips_through_model_dump() -> None:
    p = good_plan()
    assert ShotPlan.model_validate(p.model_dump()) == p


# ---- the look library ---------------------------------------------------------------------------


def test_the_library_has_the_twelve_looks_as_ranges() -> None:
    assert [look.id for look in LOOKS.all()] == [
        "lifestyle", "cinematic", "editorial", "documentary", "studio", "fantasy",
        "flat_illustration", "3d_render", "food", "architectural", "street", "minimal",
    ]  # fmt: skip
    for look in LOOKS.all():
        assert len(look.light) >= 2 and look.lens and len(look.palette) >= 2 and look.depth, look.id
    assert LOOKS.valid_ids()[0] == "auto" and "studio" in LOOKS.valid_ids()
    assert LOOKS.pick("auto") is None and LOOKS.pick("fantasy").id == "fantasy"  # type: ignore[union-attr]


def test_look_text_never_names_light_equipment() -> None:
    for look in LOOKS.all():
        found: list[rules.Violation] = []
        rules._artefacts("", look.as_prompt_line() + look.notes, found)
        assert not found, (look.id, found)
    assert len(LOOKS.library_text()) < 6000  # it is sent on every call


def test_an_empty_library_file_fails_loudly(tmp_path: Path) -> None:
    f = tmp_path / "l.yaml"
    f.write_text("looks: {}\n")
    with pytest.raises(BadInput):
        load_looks(f)


# ---- subject type -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("perfume bottle on marble", "product"),
        ("a red fox in the snow", "animal"),
        ("two rabbits in a meadow", "animal"),
        ("a quiet beach at sunset", "place"),
        ("a dragon on a castle", "fantasy"),
        ('a poster that says "OPEN"', "poster"),
        ("a bowl of ramen", "food"),
        ("an abstract gradient wallpaper", "abstract"),
        ("ek ladki park mein", "person"),
        ("ماربل پر پرفیوم کی بوتل", "product"),
        ("ایک بلی کھڑکی پر", "animal"),
        ("something calm", "other"),
    ],
)
def test_subject_type_is_guessed_in_english_roman_urdu_and_urdu(prompt: str, expected: str) -> None:
    assert guess_subject_type(prompt) == expected


# ---- the plan store: seeds, retrieval, project memory --------------------------------------------


def test_thirty_hand_checked_seeds_cover_every_subject_type_and_pass_the_rules() -> None:
    assert len(SEEDS) == 30
    assert {s["subject_type"] for s in SEEDS} == {
        "product", "person", "animal", "place", "food", "fantasy", "poster", "abstract",
    }  # fmt: skip
    for s in SEEDS:
        plan = ShotPlan.model_validate(s["plan"])
        assert plan.look in LOOKS.valid_ids() and plan.intent.subject_type == s["subject_type"]
        assert rules.check_plan(s["prompt"], plan, valid_looks=LOOKS.valid_ids()) == [], s["id"]


def test_the_store_loads_seeds_once_and_returns_the_top_two_similar_plans() -> None:
    store = PlanStore()
    assert (
        store.count("seed") == 30
        and store.load_seeds(Path(ad.__file__).parents[3] / "local" / "seed_plans.json") == 0
    )
    hits = store.similar("a bottle of face serum on a bathroom shelf", subject_type="product")
    assert len(hits) == 2 and all(h.subject_type == "product" for h in hits)
    assert all((h.aesthetic or 0) >= 8 for h in hits)
    assert store.similar("zzz", subject_type="other") == []


def test_a_plan_for_the_very_same_prompt_is_not_returned() -> None:
    hits = PlanStore().similar("perfume bottle on marble", subject_type="product", k=5)
    assert "perfume bottle on marble" not in [h.prompt for h in hits]


def test_roman_urdu_prompts_find_plans_by_subject_type() -> None:
    hits = PlanStore().similar("ek billi khidki par", subject_type="animal")
    assert hits and all(h.subject_type == "animal" for h in hits)


def test_only_plans_scored_eight_or_more_are_retrieved_and_runs_are_stored() -> None:
    store = PlanStore(seeds=None)
    plan = good_plan().model_dump(mode="json")
    store.add(
        run_id="r1",
        index=0,
        subject_type="product",
        look="lifestyle",
        prompt="iced coffee cup",
        plan=plan,
        aesthetic=6.0,
        realism=7.0,
    )
    store.add(
        run_id="r1",
        index=1,
        subject_type="product",
        look="lifestyle",
        prompt="iced coffee glass",
        plan=plan,
        aesthetic=9.0,
        realism=9.0,
    )
    hits = store.similar("an iced coffee", subject_type="product")
    assert [h.id for h in hits] == ["r1-1"] and hits[0].source == "run"


def test_project_memory_uses_the_first_approved_plan() -> None:
    store = PlanStore(seeds=None)
    a = (
        good_plan()
        .model_copy(update={"palette": ["navy", "gold"], "props": ["a key"]})
        .model_dump(mode="json")
    )
    b = good_plan().model_copy(update={"palette": ["red", "black"]}).model_dump(mode="json")
    first = store.add(
        run_id="p",
        index=0,
        subject_type="product",
        look="studio",
        prompt="x",
        plan=a,
        aesthetic=9,
        realism=9,
        project_id="proj1",
    )
    second = store.add(
        run_id="p",
        index=1,
        subject_type="product",
        look="cinematic",
        prompt="y",
        plan=b,
        aesthetic=9,
        realism=9,
        project_id="proj1",
    )
    assert store.project_style("proj1") is None  # nothing approved yet
    assert store.approve(second) and store.approve(first)
    style = store.project_style("proj1")
    assert style == {"look": "studio", "palette": ["navy", "gold"], "props": ["a key"]}
    assert store.project_style("other") is None


# ---- the rules ----------------------------------------------------------------------------------


def codes(plan: ShotPlan, prompt: str = "perfume bottle on marble") -> set[str]:
    return {v.code for v in rules.check_plan(prompt, plan, valid_looks=LOOKS.valid_ids())}


def test_a_complete_plan_is_clean_and_an_empty_one_is_not() -> None:
    assert codes(good_plan(), "iced coffee cup") == set()
    found = {
        (v.code, v.detail)
        for v in rules.check_plan("a bottle", ShotPlan(), valid_looks=LOOKS.valid_ids())
    }
    for field in (
        "subject",
        "world",
        "composition",
        "light",
        "camera",
        "palette",
        "realism_checks",
    ):
        assert ("missing_field", field) in found
    assert ("unknown_look", "empty") in found


def test_invented_text_people_brand_equipment_and_banned_phrases_are_caught() -> None:
    base = good_plan("perfume bottle")
    assert "invented_text" in codes(
        base.model_copy(update={"text": base.text.model_copy(update={"exact": "LUXE"})})
    )
    assert "invented_people" in codes(base.model_copy(update={"props": ["a woman's hand"]}))
    assert "invented_brand" in codes(base.model_copy(update={"subject": "Chanel perfume bottle"}))
    assert "light_equipment" in codes(base.model_copy(update={"props": ["a softbox"]}))
    assert "banned_phrase" in codes(base.model_copy(update={"world": "a seamless grey background"}))
    assert "banned_phrase" in codes(
        base.model_copy(update={"moment": "the bottle floating in air"})
    )


def test_what_the_user_asked_for_is_not_invented() -> None:
    plan = good_plan("perfume bottle").model_copy(update={"props": ["a woman's hand"]})
    assert "invented_people" not in codes(plan, "a perfume bottle held by a woman")
    assert "banned_phrase" not in codes(
        good_plan().model_copy(update={"moment": "a floating island"}), "a floating island"
    )


def test_urdu_prompts_skip_the_people_and_brand_checks() -> None:
    plan = good_plan().model_copy(update={"props": ["a woman's hand"]})
    assert "invented_people" not in codes(plan, "ماربل پر پرفیوم کی بوتل")


def prompt_codes(
    text: str, plan: ShotPlan | None = None, user: str = "perfume bottle on marble"
) -> set[str]:
    return {
        v.code
        for v in rules.check_prompt(user, plan or good_plan("perfume bottle"), text, max_words=60)
    }


def test_packaging_with_no_text_asked_needs_a_blank_label_plate_never_no_label() -> None:
    body = "perfume bottle on marble, " + "soft window light " * 8
    assert "label_plate_missing" in prompt_codes(body)
    assert "label_plate_missing" not in prompt_codes(body + " a blank label plate")
    assert "negated_label" in prompt_codes(body + " no label, unlabeled")
    assert "invented_text" in prompt_codes('a bottle with "LUXE" on it ' + body)


def test_written_prompt_length_generic_words_and_artefacts() -> None:
    assert "too_short" in prompt_codes("a bottle, a blank label plate")
    assert "too_long" in prompt_codes("word " * 70 + "a blank label plate")
    assert "generic_word" in prompt_codes("a beautiful bottle " + "x " * 40 + "blank label plate")
    assert "light_equipment" in prompt_codes("softbox " + "x " * 40 + "blank label plate")
    assert rules.word_range(60) == (42, 60)


def test_text_that_was_asked_for_may_be_quoted() -> None:
    plan = good_plan("poster").model_copy(
        update={"text": good_plan().text.model_copy(update={"exact": "FRESH ROAST"})}
    )
    assert "invented_text" not in prompt_codes(
        'the headline "FRESH ROAST" ' + "x " * 40, plan, 'a poster saying "FRESH ROAST"'
    )


def test_sanitize_and_the_retry_note() -> None:
    plan = good_plan().model_copy(
        update={"text": good_plan().text.model_copy(update={"exact": "LUXE"})}
    )
    assert rules.sanitize_plan("a cup", plan).text.exact is None
    assert rules.sanitize_plan('a cup that says "LUXE"', plan).text.exact == "LUXE"
    assert (
        rules.sanitize_prompt("a cup", good_plan(), 'a cup marked "LUXE" on marble')
        == "a cup marked on marble"
    )
    note = rules.retry_note(
        [rules.Violation("negated_label", "no label"), rules.Violation("banned_phrase", "floating")]
    )
    assert "blank label plate" in note and "floating" in note


def test_only_style_questions_are_needless() -> None:
    def asks(text: str) -> ShotPlan:
        q = missing().missing[0].model_copy(update={"text": text})
        return missing().model_copy(update={"missing": [q]})

    assert rules.needless_question(asks("What style and details should it have?"))
    assert rules.needless_question(asks("Which lighting do you prefer?"))
    assert not rules.needless_question(asks("What should the image show?"))
    assert not rules.needless_question(good_plan())


# ---- the Art Director's call --------------------------------------------------------------------


def test_the_system_prompt_orders_look_library_fixed_look_project_and_examples() -> None:
    store = PlanStore()
    shots = store.similar("a bottle of serum", subject_type="product")
    text = ad.system_prompt(
        "BASE", LOOKS, "4:5", n=2, fixed_look=LOOKS.get("studio"),
        shots=shots, project_style={"look": "studio", "palette": ["navy"], "props": []},
    )  # fmt: skip
    order = [
        text.index(x)
        for x in (
            "BASE",
            "## Look library",
            "chose the look `studio`",
            "## Project style",
            "## Frame",
            "## Good plans",
        )
    ]
    assert order == sorted(order) and "vertical frame" in text and "n = 2" in text
    assert text.count("\nPrompt: ") == 2  # two few-shot plans
    assert "## Good plans" not in ad.system_prompt("BASE", LOOKS, "1:1")


def test_the_method_prompt_has_the_six_steps_and_the_rules() -> None:
    base = (Path(ad.__file__).parents[5] / "prompts/create/txt2img/art_director.md").read_text()
    for step in (
        "1. INTENT",
        "2. LOOK",
        "3. WORLD + MOMENT",
        "4. SHOT",
        "5. REALISM",
        "6. SHIP TEST",
    ):
        assert step in base
    for needle in (
        "ELEVATE",
        "NEVER add people",
        "blank label plate",
        "NEVER by equipment",
        "seamless grey background",
        "do NOT ask",
        "settings win",
        "variation_axes",
    ):
        assert needle in base, needle  # fmt: skip
    assert len(base) < 6000


async def direct(
    llm: FakeLLM, prompt: str = "perfume bottle on marble", **kw: Any
) -> tuple[ShotPlan, dict[str, Any]]:
    report: dict[str, Any] = {}
    plan = await ad.direct(
        llm, base_prompt="BASE", prompt=prompt, aspect_ratio="1:1", resolution="1K", answers={},
        allow_questions=True, n=1, looks=LOOKS, max_tokens=2500, timeout=5, report=report, **kw,
    )  # fmt: skip
    return plan, report


async def test_one_call_at_temperature_0_4_with_medium_reasoning() -> None:
    llm = FakeLLM("m", [good_plan("perfume bottle")])
    plan, report = await direct(llm)
    call = llm.calls[0]
    assert (
        len(llm.calls) == 1 and call["temperature"] == 0.4 and call["reasoning_effort"] == "medium"
    )
    assert call["schema"] is ShotPlan and "<user_prompt>" in call["user"]
    assert report["attempts"] == 1 and report["violations"] == [] and plan.look == "lifestyle"


async def test_a_rule_break_gets_one_corrective_retry_with_a_fix_note() -> None:
    bad = good_plan("perfume bottle").model_copy(update={"props": ["a woman's hand"]})
    llm = FakeLLM("m", [bad, good_plan("perfume bottle")])
    plan, report = await direct(llm)
    assert len(llm.calls) == 2 and report["retry_reason"] == ["invented_people"]
    assert "no people" in llm.calls[1]["user"] and "fix" not in llm.calls[0]["user"]
    assert report["violations"] == [] and plan.props == ["a paper straw"]


async def test_a_style_question_is_retried_with_questions_off() -> None:
    q = missing().missing[0].model_copy(update={"text": "What style do you want?"})
    asks = missing().model_copy(update={"missing": [q]})
    llm = FakeLLM("m", [asks, good_plan("perfume bottle")])
    plan, report = await direct(llm)
    assert not plan.missing and report["retry_reason"] == ["needless_question"]
    assert '"allow_questions": false' in llm.calls[1]["user"]


async def test_a_subject_question_is_kept_and_a_stubborn_style_asker_is_not_asked_again() -> None:
    plan, _ = await direct(FakeLLM("m", [missing()]), prompt="make it nice")
    assert plan.missing and plan.missing[0].options[-1].id == "other"
    q = missing().missing[0].model_copy(update={"text": "What style?"})
    asks = missing().model_copy(update={"missing": [q]})
    llm = FakeLLM("m", [asks, asks])
    plan, _ = await direct(llm)
    assert len(llm.calls) == 2 and not plan.missing and plan.subject == "iced coffee cup"


async def test_a_refused_retry_keeps_the_first_answer_but_a_refused_first_call_fails() -> None:
    bad = good_plan("perfume bottle").model_copy(update={"props": ["a woman's hand"]})
    llm = FakeLLM("m", [bad, ProviderCallFailed("provider returned 429", retryable=True)])
    plan, report = await direct(llm)
    assert plan.props == ["a woman's hand"] and report["attempts"] == 2  # the run goes on
    with pytest.raises(ProviderCallFailed):
        await direct(FakeLLM("m", [ProviderCallFailed("429", retryable=True)]))


async def test_a_cut_off_answer_is_retried_with_more_room() -> None:
    llm = FakeLLM("m", [InvalidModelOutput("cut off"), good_plan("perfume bottle")])
    await direct(llm)
    assert [c["max_tokens"] for c in llm.calls] == [2500, 5000]


async def test_the_users_look_overrides_the_models_pick_and_an_unknown_look_is_cleared() -> None:
    plan, _ = await direct(
        FakeLLM("m", [good_plan("perfume bottle")]), fixed_look=LOOKS.get("cinematic")
    )
    assert plan.look == "cinematic"
    odd = good_plan("perfume bottle").model_copy(update={"look": "grunge"})
    plan, report = await direct(FakeLLM("m", [odd, odd]))
    assert plan.look == "" and any(v["code"] == "unknown_look" for v in report["violations"])


async def test_unrequested_text_is_removed_by_code() -> None:
    texty = good_plan("perfume bottle").model_copy(
        update={"text": good_plan().text.model_copy(update={"exact": "LUXE"})}
    )
    plan, _ = await direct(FakeLLM("m", [texty, texty]))
    assert plan.text.exact is None


def test_a_question_model_is_still_a_valid_question() -> None:
    assert isinstance(missing().missing[0], Question)


async def test_a_rate_limit_on_the_first_call_waits_once_then_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    waits: list[float] = []

    async def fake_sleep(s: float) -> None:
        waits.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    err = ProviderCallFailed("provider returned 429", retryable=True, retry_after=7.0)
    llm = FakeLLM("m", [err, good_plan("perfume bottle")])
    plan, _ = await direct(llm)
    assert waits == [7.0] and len(llm.calls) == 2 and plan.subject
    twice = FakeLLM("m", [err, err])
    with pytest.raises(ProviderCallFailed):
        await direct(twice)
