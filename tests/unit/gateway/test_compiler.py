"""The Compiler (§5.2): Brief + chosen model -> that model's prompt. Plain code, no model."""

from __future__ import annotations

import httpx
import pytest
import respx

from media_engine.gateway.compiler import BANNED_WORDS, compile_prompt, style_for_cloudflare
from media_engine.gateway.types import ImageRequest

from .conftest import CF_RUN

BRIEF = {
    "subject": "a red fox",
    "action": "trotting across snow",
    "setting": "a pine forest at dawn",
    "lighting": "low sun from the left",
    "camera": "50mm lens at eye level",
    "detail_anchors": ["frost on the whiskers", "orange fur", "blue shadows"],
}


def test_sentences_style_reads_as_sentences():
    out = compile_prompt(BRIEF, "sentences", 60)
    assert out.text.startswith("A red fox trotting across snow, a pine forest at dawn.")
    assert "Lit by low sun from the left" in out.text and out.issues == []


def test_tags_style_is_lowercase_comma_separated_and_nouns_keep_case():
    tags = compile_prompt({**BRIEF, "subject": "A Red Fox"}, "tags", 60).text
    assert tags == tags.lower() and "." not in tags
    assert compile_prompt({**BRIEF, "subject": "A Red Fox"}, "nouns", 60).text.startswith(
        "A Red Fox"
    )


@pytest.mark.parametrize("style", ["nouns", "sentences", "tags"])
@pytest.mark.parametrize("budget", [12, 20, 40, 80])
def test_every_compiled_prompt_fits_its_word_budget(style, budget):
    assert compile_prompt(BRIEF, style, budget).words <= budget


def test_the_least_important_fields_are_trimmed_first():
    out = compile_prompt(BRIEF, "sentences", 24).text
    assert "red fox" in out and "pine forest" in out and "low sun" in out
    assert "blue shadows" not in out  # anchors go first


@pytest.mark.parametrize("word", BANNED_WORDS)
def test_a_clause_with_banned_vocabulary_is_removed_whole(word):
    brief = {**BRIEF, "setting": f"a pine forest, {word} mood of calm", "lighting": "low sun"}
    out = compile_prompt(brief, "sentences", 60).text
    assert word not in out.lower() and "mood of calm" not in out and "pine forest" in out


def test_a_trailing_mood_clause_is_dropped():
    brief = {**BRIEF, "camera": "50mm lens, evoking a sense of calm"}
    assert "calm" not in compile_prompt(brief, "sentences", 60).text


def test_it_reports_what_is_missing_and_never_invents_it():
    out = compile_prompt({"subject": "a cup"}, "sentences", 60)
    assert out.issues == ["no light direction", "0 detail anchor(s), need 2"]
    assert out.text == "A cup."


def test_the_users_words_stand_in_when_the_brief_has_no_subject():
    out = compile_prompt({"lighting": "low sun"}, "sentences", 60, fallback="a red fox")
    assert "red fox" in out.text and "no concrete subject" in out.issues


def test_cloudflare_styles_are_known_and_unknown_models_get_a_default():
    assert style_for_cloudflare("@cf/black-forest-labs/flux-1-schnell") == ("sentences", 60)
    assert style_for_cloudflare("@cf/new/model") == ("sentences", 60)


def test_the_flux_2_family_shares_one_budget():
    for model in ("flux-2-klein-4b", "flux-2-klein-9b"):
        assert style_for_cloudflare(f"@cf/black-forest-labs/{model}") == ("sentences", 60)


# -- negatives, the final recheck, empty fields, lead order -------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "gone"),
    [
        ("setting", "a pine forest, no people, no cars", "people"),
        ("setting", "a pine forest, without any leaves", "leaves"),
        ("action", "standing in a field with no shoes", "shoes"),
        ("lighting", "low sun, without glare", "glare"),
    ],
)
def test_negatives_are_dropped_and_the_positive_part_stays(field, value, gone):
    out = compile_prompt({**BRIEF, field: value}, "sentences", 60).text
    assert gone not in out and "red fox" in out


def test_a_word_that_merely_starts_with_no_is_kept():
    out = compile_prompt({**BRIEF, "setting": "a nordic lake, no-parking sign"}, "sentences", 60)
    assert "nordic lake" in out.text and "no-parking sign" in out.text


def test_a_banned_phrase_built_across_two_fields_is_caught_by_the_final_recheck():
    brief = {**BRIEF, "subject": "a soft", "action": "diffused lamp"}
    out = compile_prompt(brief, "sentences", 60)
    assert "soft diffused" not in out.text.lower() and "pine forest" in out.text
    assert "final check removed banned or negative wording" in out.issues
    assert out.words == len(out.text.split())


def test_a_clean_prompt_is_not_touched_by_the_final_recheck():
    out = compile_prompt(BRIEF, "sentences", 60)
    assert out.issues == [] and out.text.endswith("blue shadows.")


@pytest.mark.parametrize("style", ["nouns", "sentences", "tags"])
def test_an_empty_camera_is_omitted_in_every_style(style):
    out = compile_prompt({**BRIEF, "camera": ""}, style, 60).text
    assert "lens" not in out and ", ," not in out and ". ." not in out and ". ," not in out


def test_the_background_lead_opens_with_the_setting_then_the_subject():
    out = compile_prompt(BRIEF, "sentences", 60, lead="background").text
    assert out.startswith("A pine forest at dawn. ")
    assert out.index("red fox") < out.index("Lit by") < out.index("frost on the whiskers")


def test_the_default_lead_is_subject_first_and_unchanged():
    assert compile_prompt(BRIEF, "sentences", 60, lead="subject") == compile_prompt(
        BRIEF, "sentences", 60
    )


@pytest.mark.parametrize("budget", [12, 20, 40])
def test_the_background_lead_still_fits_its_budget(budget):
    assert compile_prompt(BRIEF, "sentences", budget, lead="background").words <= budget


def test_the_cards_set_styles_budgets_and_lead_as_agreed():
    from media_engine.gateway.image_registry import image_registry

    cards = {spec.slug: spec for spec in image_registry().all()}
    flux2 = [s for slug, s in cards.items() if slug.startswith("black-forest-labs/flux.2")]
    assert flux2 and all(s.prompt_style == "sentences" and s.prompt_max_words == 60 for s in flux2)
    seedream = [s for slug, s in cards.items() if slug.startswith("bytedance-seed/seedream")]
    assert seedream and all(s.prompt_max_words == 70 for s in seedream)
    assert cards["qwen/qwen-image-3"].prompt_style == "sentences"
    assert cards["recraft/recraft-v4.1"].prompt_style == "tags"
    assert cards["openai/gpt-image-2"].lead_order == "background"
    others = {s.lead_order for slug, s in cards.items() if slug != "openai/gpt-image-2"}
    assert others == {"subject"}
    assert not any(s.prompt_style == "nouns" for s in cards.values())  # retired from the cards


@respx.mock
async def test_the_gateway_passes_the_cards_lead_order_to_the_compiler(gateway, monkeypatch):
    from media_engine.gateway import router

    seen: dict[str, str] = {}
    real = router.compile_prompt

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen["lead"] = kwargs["lead"]
        return real(*args, **kwargs)

    monkeypatch.setattr(router, "compile_prompt", spy)
    respx.post(CF_RUN + "/@cf/black-forest-labs/flux-1-schnell").mock(
        return_value=httpx.Response(200, json={"result": {"image": _png()}})
    )

    await gateway.image("image_create", ImageRequest(prompt="x", brief=BRIEF), plan="free")

    assert seen["lead"] == "subject"  # a Cloudflare model has no card: the default


def _png() -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()
