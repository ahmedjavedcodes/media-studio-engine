"""Image-create presets (presets/library/image_create.yaml).

The rule from the catalog decision (2026-11-20): every aspect ratio the Brief can emit must exist
in the presets. Before, the Brief prompt offered 4:5 but the presets did not list it, so a 4:5
Brief silently fell back to the default 1:1."""

from __future__ import annotations

import re

import pytest

from media_engine.core.prompts import PROMPTS_DIR, load_prompt
from media_engine.services.create.loader import load_create_config

CFG = load_create_config()
RATIOS = CFG.presets.aspect_ratios
_RATIO = re.compile(r"\b(\d{1,2}):(\d{1,2})\b")


def test_every_ratio_the_brief_prompt_names_is_a_preset():
    text = load_prompt(PROMPTS_DIR / CFG.agent("prompt_enhancer").prompt)
    named = {f"{w}:{h}" for w, h in _RATIO.findall(text)}
    assert named, "the Brief prompt names no aspect ratio: the test would prove nothing"
    assert named <= set(RATIOS), f"missing from the presets: {sorted(named - set(RATIOS))}"


def test_the_catalog_ratios_are_offered():
    assert {"1:1", "4:5", "3:4", "16:9", "9:16", "3:2"} <= set(RATIOS)


@pytest.mark.parametrize("ratio", sorted(RATIOS))
def test_each_preset_size_matches_its_ratio_and_the_free_models_limits(ratio):
    size = RATIOS[ratio]
    w, h = (int(x) for x in ratio.split(":"))
    assert abs(size.width / size.height - w / h) <= CFG.policy.aspect_ratio_tolerance
    assert max(size.width, size.height) <= 1024  # flux-1-schnell max side
    assert size.width % 16 == 0 and size.height % 16 == 0


def test_the_default_ratio_is_a_preset():
    assert CFG.presets.defaults.aspect_ratio in RATIOS
