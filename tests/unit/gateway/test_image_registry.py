"""OpenRouter image registry: models.yaml merged with the catalogue cache. No HTTP at all."""

from __future__ import annotations

from pathlib import Path

import pytest

from media_engine.gateway.image_registry import MODELS_PATH, ImageRegistry, ModelEntry, merge
from media_engine.gateway.openrouter_models import ImageModelInfo, ModelCache
from media_engine.gateway.registry import card_or_default, get_card

GPT_IMAGE = "openai/gpt-image-2"
KLEIN = "black-forest-labs/flux.2-klein-4b"
QWEN = "qwen/qwen-image-3"

# Catalogue records as GET /api/v1/images/models returned them on 2026-11-20 (trimmed).
GPT_IMAGE_INFO = ImageModelInfo(
    id=GPT_IMAGE,
    supported_parameters={
        "aspect_ratio": {"type": "enum", "values": ["1:1", "16:9", "auto"]},
        "quality": {"type": "enum", "values": ["auto", "low", "medium", "high"]},
        "n": {"type": "range", "min": 1, "max": 10},
        "input_references": {"type": "range", "min": 0, "max": 16},
    },
    input_modalities=["text", "image"],
    output_modalities=["image"],
    pricing=[{"billable": "output_image", "unit": "token", "cost_usd": 3e-05}],
)
KLEIN_INFO = ImageModelInfo(
    id=KLEIN,
    supported_parameters={
        "aspect_ratio": {"type": "enum", "values": ["1:1", "16:9", "auto"]},
        "n": {"type": "range", "min": 1, "max": 1},
        "input_references": {"type": "range", "min": 0, "max": 4},
        "seed": {"type": "boolean"},
    },
    input_modalities=["text", "image"],
    output_modalities=["image"],
    pricing=[{"billable": "output_image", "unit": "megapixel", "cost_usd": 0.014}],
)

ALL_SLUGS = {
    "openai/gpt-image-2",
    "meta/muse-image",
    "black-forest-labs/flux.2-klein-4b",
    "bytedance-seed/seedream-5-0-flash",
    "black-forest-labs/flux.2-pro",
    "qwen/qwen-image-3",
    "google/gemini-3.1-flash-lite-image",
    "recraft/recraft-v4.1",
    "bytedance-seed/seedream-5-0-lite",
    "black-forest-labs/flux.2-flex",
    "sourceful/riverflow-v2.5-pro",
    "google/gemini-3.1-flash-image",
    "black-forest-labs/flux.2-max",
    "google/gemini-nano-banana-2.1",
    "google/gemini-3-pro-image",
}


def entry(**overrides: object) -> ModelEntry:
    base: dict[str, object] = {
        "slug": KLEIN,
        "name": "FLUX.2 Klein 4B",
        "tier": "entry",
        "billing_unit": "megapixel",
        "price_measured_usd": 0.014,
        "supports_seed": True,
        "n_per_call": 1,
        "max_input_images": 4,
        "prompt_style": "nouns",
        "prompt_max_words": 60,
    }
    return ModelEntry.model_validate(base | overrides)


def cache_with(tmp_path: Path, *infos: ImageModelInfo) -> ModelCache:
    cache = ModelCache(tmp_path / "models.json")
    cache.save(list(infos))
    return cache


# -- models.yaml itself -----------------------------------------------------------------------


def test_models_yaml_lists_the_15_spec_models_with_every_registry_field():
    registry = ImageRegistry.load(None)
    assert {s.slug for s in registry.all()} == ALL_SLUGS
    for spec in registry.all():
        assert spec.prompt_max_words > 0 and spec.prompt_style in {"nouns", "sentences", "tags"}
        assert spec.n_per_call >= 1 and spec.max_input_images >= 0
        assert spec.billing_unit in {"image", "megapixel", "token"}


def test_the_eval_default_is_flux_klein_and_its_seed_is_verified():
    spec = ImageRegistry.load(None).eval_default()
    assert spec is not None and spec.slug == KLEIN
    assert spec.supports_seed and spec.seed_verified


def test_seed_is_verified_only_on_flux_klein():
    verified = [s.slug for s in ImageRegistry.load(None).all() if s.seed_verified]
    assert verified == [KLEIN]  # spec §6.4: the only model tested for reproducibility


def test_tier_names_are_entry_pro_max_ultra_and_entry_replaces_free():
    tiers = {s.tier for s in ImageRegistry.load(None).all()}
    assert tiers == {"entry", "pro", "max", "ultra"}


def test_muse_is_marked_unavailable():
    # 2026-11-20: listed by the catalogue with no parameters and no serving endpoint
    spec = ImageRegistry.load(None).get("meta/muse-image")
    assert spec is not None and not spec.available
    assert "meta/muse-image" not in {s.slug for s in ImageRegistry.load(None).for_plan("ultra")}


def test_duplicate_slugs_are_rejected(tmp_path):
    text = MODELS_PATH.read_text()
    first = text.index("  - slug: openai/gpt-image-2")
    second = text.index("  - slug: meta/muse-image")
    path = tmp_path / "models.yaml"
    path.write_text(text + text[first:second])
    with pytest.raises(ValueError, match="more than once"):
        ImageRegistry.load(None, path=path)


# -- plan access ------------------------------------------------------------------------------


def test_entry_models_are_open_to_the_free_plan_and_pro_models_are_not():
    registry = ImageRegistry.load(None)
    free = {s.slug for s in registry.for_plan("free")}
    assert KLEIN in free and GPT_IMAGE in free
    assert QWEN not in free
    assert QWEN in {s.slug for s in registry.for_plan("pro")}


def test_gpt_image_high_and_unmeasured_medium_quality_are_ultra_only():
    spec = ImageRegistry.load(None).get(GPT_IMAGE)
    assert spec is not None
    assert spec.allowed_for("free", {"quality": "low"})
    assert spec.allowed_for("free")  # nothing price-multiplying sent
    for quality in ("high", "medium"):  # medium unmeasured: gated like high (fail closed)
        assert not spec.allowed_for("max", {"quality": quality})
        assert spec.allowed_for("ultra", {"quality": quality})
    reason = spec.plan_block("free", {"quality": "high"})
    assert reason is not None and "needs the ultra plan" in reason


def test_4k_resolution_is_ultra_only_on_every_model_that_offers_it():
    registry = ImageRegistry.load(None)
    for spec in registry.all():
        assert spec.min_tier_for.get("resolution", {}).get("4K") == "ultra", spec.slug
    riverflow = registry.get("sourceful/riverflow-v2.5-pro")
    assert riverflow is not None
    assert not riverflow.allowed_for("max", {"resolution": "4K"})
    assert riverflow.allowed_for("max", {"resolution": "2K"})


def test_a_model_tier_above_the_plan_is_blocked():
    spec = ImageRegistry.load(None).get(QWEN)  # pro tier
    assert spec is not None
    assert spec.plan_block("free") == f"{QWEN} is a pro-tier model; plan free may not use it"
    assert spec.allowed_for("pro")


# -- merging with the catalogue ---------------------------------------------------------------


def test_without_a_cache_the_yaml_values_are_used_as_written():
    spec, warnings = merge(entry(n_per_call=3), None)
    assert spec.n_per_call == 3 and spec.supports_seed
    assert not spec.in_catalogue and spec.listed_pricing == []
    assert warnings == []


def test_the_catalogue_wins_on_capabilities_and_each_disagreement_warns():
    spec, warnings = merge(entry(supports_seed=False, n_per_call=4, max_input_images=8), KLEIN_INFO)
    assert (spec.supports_seed, spec.n_per_call, spec.max_input_images) == (True, 1, 4)
    assert len(warnings) == 3
    assert all("using the catalogue" in w for w in warnings)


def test_capabilities_are_read_from_the_catalogue():
    spec, warnings = merge(
        entry(
            slug=GPT_IMAGE,
            billing_unit="token",
            supports_seed=False,
            n_per_call=10,
            max_input_images=16,
            quality_settings=["auto", "low", "medium", "high"],
            default_quality="low",
        ),
        GPT_IMAGE_INFO,
    )
    assert warnings == []
    assert spec.supported_aspect_ratios == ["1:1", "16:9", "auto"]
    assert spec.supported_sizes == []  # gpt-image-2 has no `resolution` parameter
    assert spec.quality_settings == ["auto", "low", "medium", "high"]
    assert spec.default_quality == "low"
    assert spec.supports_edit and not spec.supports_mask
    assert spec.listed_pricing[0]["unit"] == "token"


def test_a_default_quality_the_model_does_not_accept_is_dropped_with_a_warning():
    spec, warnings = merge(entry(quality_settings=["ultra"], default_quality="ultra"), KLEIN_INFO)
    assert spec.default_quality is None
    assert any("default_quality" in w for w in warnings)


def test_a_seed_marked_verified_on_a_model_without_seed_is_not_trusted():
    spec, warnings = merge(entry(slug=GPT_IMAGE, seed_verified=True), GPT_IMAGE_INFO)
    assert not spec.supports_seed and not spec.seed_verified
    assert any("seed_verified" in w for w in warnings)


def test_a_yaml_model_missing_from_the_cache_is_reported(tmp_path):
    registry = ImageRegistry.load(cache_with(tmp_path, KLEIN_INFO))
    assert any(w.startswith(f"{GPT_IMAGE}: not in the catalogue cache") for w in registry.warnings)
    gpt = registry.get(GPT_IMAGE)
    assert gpt is not None and not gpt.in_catalogue  # still usable from its yaml values


# -- model cards for the rest of the gateway --------------------------------------------------


def test_openrouter_image_cards_come_from_the_registry():
    klein = card_or_default("openrouter", KLEIN, "image")
    assert not klein.free and klein.accepts_seed and klein.honours_seed is True
    gpt = card_or_default("openrouter", GPT_IMAGE, "image")
    assert not gpt.accepts_seed and gpt.honours_seed is False
    qwen = get_card("openrouter", QWEN)
    assert qwen is not None and qwen.accepts_seed and qwen.honours_seed is None  # unverified


def test_unknown_openrouter_models_still_get_a_default_card():
    card = card_or_default("openrouter", "someone/unknown-model", "image")
    assert card.accepts_seed and card.honours_seed is None
    assert get_card("openrouter", "openai/gpt-oss-120b") is None  # a chat model: no image card
