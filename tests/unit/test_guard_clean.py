import time

import pytest

from app.core.errors import BadInput
from app.guardrails.clean import clean_text


def test_raw_is_untouched() -> None:
    raw = "  A Red FOX​  "
    out = clean_text(raw)
    assert out.prompt_raw == raw
    assert out.prompt_clean == "a red fox"


def test_nfkc_and_control_chars() -> None:
    assert clean_text("ｎｕｄｅ\x00\x07 photo").prompt_clean == "nude photo"
    assert clean_text("a​‍bc﻿").prompt_clean == "abc"


def test_whitespace_collapses() -> None:
    assert clean_text("a \n\t  b").prompt_clean == "a b"


def test_spaced_letters_are_joined() -> None:
    assert clean_text("make her n u d e please").prompt_clean == "make her nude please"
    assert clean_text("n.u.d.e").prompt_clean == "n.u.d.e"  # only spaces are joined


def test_two_single_letters_are_not_joined() -> None:
    assert clean_text("plan a b roll").prompt_clean == "plan a b roll"


def test_leetspeak_only_on_mixed_tokens() -> None:
    assert clean_text("n0de ph0to").prompt_clean == "node photo"
    assert clean_text("nud3 b0dy").prompt_clean == "nude body"
    assert clean_text("p@ssw0rd $uper").prompt_clean == "password super"


@pytest.mark.parametrize("token", ["4k", "8k", "2d", "3d", "1080p", "16:9", "9:16", "35mm", "2024"])
def test_leetspeak_skips_normal_tokens(token: str) -> None:
    assert clean_text(f"city {token} view").prompt_clean == f"city {token} view"


def test_too_long_is_400() -> None:
    with pytest.raises(BadInput):
        clean_text("a" * 4001)
    assert clean_text("a" * 4000).prompt_raw == "a" * 4000


def test_bad_utf8_is_400() -> None:
    with pytest.raises(BadInput):
        clean_text(b"\xff\xfe\xfa")
    with pytest.raises(BadInput):
        clean_text("lone surrogate \ud800")


def test_no_translation() -> None:
    assert clean_text("یہ ایک تصویر ہے").prompt_clean == "یہ ایک تصویر ہے"


def test_fast() -> None:
    text = "a red fox in the snow, 4k, n u d e " * 100
    start = time.perf_counter()
    clean_text(text[:4000])
    assert time.perf_counter() - start < 0.005
