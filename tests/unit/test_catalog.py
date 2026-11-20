"""The plan catalogue (services/catalog.py), against the catalog decisions of 2026-11-20."""

from __future__ import annotations

import pytest

from media_engine.gateway.image_registry import image_registry
from media_engine.services.catalog import catalog

PLANS = ("free", "pro", "max", "ultra")

# The offered set, by the lowest plan that unlocks each option (catalog decisions 1-3).
OFFERED = {
    "free": {
        "openai/gpt-image-2@low",
        "black-forest-labs/flux.2-klein-4b",
        "bytedance-seed/seedream-5-0-flash",
    },
    "pro": {
        "black-forest-labs/flux.2-pro",
        "qwen/qwen-image-3",
        "google/gemini-3.1-flash-lite-image",
    },
    "max": {
        "recraft/recraft-v4.1",
        "bytedance-seed/seedream-5-0-lite",
        "black-forest-labs/flux.2-flex",
        "sourceful/riverflow-v2.5-pro",
    },
    "ultra": {
        "google/gemini-3.1-flash-image",
        "black-forest-labs/flux.2-max",
        "google/gemini-3-pro-image",
        "openai/gpt-image-2@high",
    },
}


def test_the_offered_models_and_their_plans_match_the_decided_table():
    by_plan: dict[str, set[str]] = {p: set() for p in PLANS}
    for model in catalog("ultra").models:
        by_plan[model.plan].add(model.id)
    assert by_plan == OFFERED


@pytest.mark.parametrize("plan", PLANS)
def test_plans_are_cumulative_and_higher_tiers_are_locked_with_an_upgrade_hint(plan):
    rank = PLANS.index(plan)
    for model in catalog(plan).models:
        unlocked = PLANS.index(model.plan) <= rank
        assert model.locked is not unlocked
        assert model.upgrade_to == (None if unlocked else model.plan)


def test_unavailable_and_unoffered_models_are_never_listed():
    ids = {m.slug for m in catalog("ultra").models}
    assert "meta/muse-image" not in ids  # available: false (no endpoint serves it)
    assert "google/gemini-nano-banana-2.1" not in ids  # offered: false (no measured price)
    assert image_registry().get("google/gemini-nano-banana-2.1") is not None  # still registered


def test_a_locked_option_is_exactly_what_the_gateway_would_refuse():
    # The catalogue's lock and the gateway's enforcement come from the same check.
    registry = image_registry()
    for plan in PLANS:
        for model in catalog(plan).models:
            spec = registry.get(model.slug)
            assert spec is not None
            assert model.locked is (not spec.allowed_for(plan, model.settings))


def test_gpt_image_2_high_is_an_ultra_option_of_the_same_model():
    high = next(m for m in catalog("max").models if m.id == "openai/gpt-image-2@high")
    assert high.settings == {"quality": "high"} and high.plan == "ultra" and high.locked


def test_only_gpt_image_2_high_is_batch_only():
    # The flag is per setting value, not per model: the low option of the same model stays
    # interactive.
    flagged = {m.id for m in catalog("ultra").models if m.batch_only}
    assert flagged == {"openai/gpt-image-2@high"}
    spec = image_registry().get("openai/gpt-image-2")
    assert spec is not None and spec.is_batch_only({"quality": "high"})
    assert not spec.is_batch_only({"quality": "low"}) and not spec.is_batch_only({})


@pytest.mark.parametrize(("plan", "cap"), [("free", 2), ("pro", 4), ("max", 4), ("ultra", 4)])
def test_the_image_cap_follows_the_plan(plan, cap):
    assert catalog(plan).max_images == cap


@pytest.mark.parametrize(("plan", "ceiling"), [("pro", "2K"), ("max", "2K"), ("ultra", "4K")])
def test_4k_is_ultra_only_and_max_tops_out_at_2k(plan, ceiling):
    assert catalog(plan).max_resolution == ceiling


def test_sizes_modes_operations_and_the_auto_default():
    c = catalog("free")
    assert c.default_model == "auto"
    assert {"1:1", "4:5", "3:4", "16:9", "9:16", "3:2"} <= set(c.aspect_ratios)
    assert c.default_aspect_ratio in c.aspect_ratios
    assert c.modes == ["auto", "manual"]
    assert c.operations == {
        "create": "live",
        "edit": "coming_soon",
        "merge": "coming_soon",
        "restyle": "coming_soon",
    }


def test_options_are_ordered_by_plan_then_price():
    models = catalog("ultra").models
    keys = [(PLANS.index(m.plan), m.price_usd or 0.0) for m in models]
    assert keys == sorted(keys)
