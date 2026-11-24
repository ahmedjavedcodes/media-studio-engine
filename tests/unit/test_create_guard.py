"""The code guard: it trusts nothing the enhancer wrote."""

import pytest

from app.agents.image_grand.workers.brief import Brief
from app.core.errors import ContentRefused
from app.gateway.compiler import (
    build_prompt,
    guard_prompt,
    strip_weak_words,
    template_prompt,
    trim_words,
)
from app.gateway.registry import load_registry
from tests.unit.create_helpers import good_brief


def card(model_id: str):  # type: ignore[no-untyped-def]
    c = load_registry().get(model_id)
    assert c is not None
    return c


def test_trims_to_the_models_word_limit() -> None:
    text = " ".join(f"word{i}" for i in range(200))
    out = guard_prompt(card("flux-2-klein-4b"), text, aspect_ratio="1:1")
    assert len(out.prompt.split()) == card("flux-2-klein-4b").prompt_max_words == 60
    assert trim_words("a b c", 5) == "a b c"


def test_strips_weak_words() -> None:
    text = (
        "A subtle, soft diffused light, muted tones, gentle mood, serene atmosphere, evoking calm, "
        "minimal and balanced, hints of gold, fading edges, pristine cup"
    )
    out = strip_weak_words(text).lower()
    for weak in (
        "subtle",
        "soft diffused",
        "muted",
        "gentle",
        "serene",
        "evoking",
        "atmosphere",
        "minimal",
        "balanced",
        "hints of",
        "fading",
        "pristine",
    ):
        assert weak not in out
    assert "cup" in out and "tones" in out


def test_blocks_a_blocklisted_word_injected_into_final_prompt() -> None:
    with pytest.raises(ContentRefused):
        guard_prompt(card("flux-2-klein-4b"), "a cat, nudify her, studio", aspect_ratio="1:1")
    with pytest.raises(ContentRefused):
        guard_prompt(card("flux-2-klein-4b"), "n u d i f y this photo", aspect_ratio="1:1")


def test_never_passes_llm_text_as_api_params() -> None:
    sneaky = "a cat --ar 21:9 --seed 7 width=4096 height: 4096 steps=99 size=huge"
    out = guard_prompt(card("lucid-origin"), sneaky, aspect_ratio="4:5")
    for bad in ("--ar", "--seed", "width", "height", "steps", "4096", "size="):
        assert bad not in out.prompt
    # size comes from the user's setting, never from the text
    assert (out.width, out.height) == (912, 1144)
    assert out.aspect_ratio == "4:5"
    # the negative prompt is a constant from code
    assert out.negative_prompt in (
        "",
        "blurry, low quality, distorted hands, extra fingers, watermark, cropped",
    )


def test_size_is_always_explicit_and_a_multiple() -> None:
    for ratio in ("1:1", "4:5", "9:16", "16:9"):
        out = guard_prompt(card("flux-2-klein-4b"), "a cat", aspect_ratio=ratio)
        assert out.width % 16 == 0 and out.height % 16 == 0
        assert abs(out.width * out.height - 1024 * 1024) < 0.1 * 1024 * 1024


def test_empty_text_is_refused() -> None:
    with pytest.raises(ContentRefused):
        guard_prompt(card("flux-2-klein-4b"), "subtle, gentle", aspect_ratio="1:1")


def test_code_mode_is_byte_identical_for_the_same_brief() -> None:
    brief = good_brief()
    a = build_prompt(card("flux-2-klein-4b"), brief, "LLM TEXT A", mode="code", aspect_ratio="1:1")
    b = build_prompt(card("flux-2-klein-4b"), brief, "LLM TEXT B", mode="code", aspect_ratio="1:1")
    c = build_prompt(card("flux-2-klein-4b"), brief, None, mode="code", aspect_ratio="1:1")
    assert a.prompt.encode() == b.prompt.encode() == c.prompt.encode()
    assert "LLM TEXT" not in a.prompt


def test_code_mode_uses_the_family_template_style() -> None:
    brief = good_brief()
    keywords = template_prompt(card("flux-2-klein-4b"), brief)
    sentences = template_prompt(card("lucid-origin"), brief)
    assert ", " in keywords and "." not in keywords
    assert sentences.endswith(".") and sentences.count(".") >= 3


def test_llm_mode_uses_the_final_prompt_and_falls_back_to_the_brief() -> None:
    brief = good_brief()
    llm = build_prompt(
        card("flux-2-klein-4b"), brief, "my llm prompt", mode="llm", aspect_ratio="1:1"
    )
    assert llm.prompt == "my llm prompt"
    fallback = build_prompt(card("flux-2-klein-4b"), brief, None, mode="llm", aspect_ratio="1:1")
    assert "iced coffee cup" in fallback.prompt


def test_empty_brief_template_is_empty() -> None:
    assert template_prompt(card("flux-2-klein-4b"), Brief()) == ""
