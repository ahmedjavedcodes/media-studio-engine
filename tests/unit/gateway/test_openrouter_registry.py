"""OpenRouter adapter driven by the image registry, and the USD estimate. HTTP mocked by respx;
no request here ever reaches OpenRouter."""

from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
import respx
from PIL import Image

from media_engine.core.config import Settings
from media_engine.core.images import sniff_media_type
from media_engine.gateway.budget import EDIT_MARKUP_DEFAULT, estimate_image_usd
from media_engine.gateway.errors import ProviderError
from media_engine.gateway.image_registry import (
    ImageModelSpec,
    ImageRegistry,
    ModelEntry,
    merge,
    use_image_registry,
)
from media_engine.gateway.openrouter_models import ImageModelInfo, ModelCache
from media_engine.gateway.providers.openrouter import OpenRouterProvider
from media_engine.gateway.types import ImageRequest

from .test_image_registry import GPT_IMAGE, GPT_IMAGE_INFO, KLEIN, KLEIN_INFO

OR_IMAGES = "https://openrouter.ai/api/v1/images"
SVG = b'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"/>'


def encoded(image: bytes) -> str:
    return base64.b64encode(image).decode()


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, "PNG")
    return buf.getvalue()


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, "JPEG")
    return buf.getvalue()


def webp() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(buf, "WEBP")
    return buf.getvalue()


def ok(*items: dict, cost: float = 0.01) -> httpx.Response:
    return httpx.Response(200, json={"data": list(items), "usage": {"cost": cost}})


@pytest.fixture
async def provider(tmp_path):
    # THE registry: the real models.yaml with a test catalogue (two models).
    cache = ModelCache(tmp_path / "models.json")
    cache.save([GPT_IMAGE_INFO, KLEIN_INFO])
    use_image_registry(ImageRegistry.load(cache))
    settings = Settings(_env_file=None, openrouter_api_key="or-test-key", allow_paid=True)
    async with httpx.AsyncClient() as client:
        yield OpenRouterProvider(client, settings)


def sent(route: respx.Route) -> dict:
    return json.loads(route.calls[0].request.content)


# -- seed ---------------------------------------------------------------------------------------


@respx.mock
async def test_seed_is_sent_to_a_model_that_takes_one(provider):
    route = respx.post(OR_IMAGES).mock(return_value=ok({"b64_json": encoded(png())}))
    await provider.generate_image(KLEIN, ImageRequest(prompt="a fox", seed=42))
    assert sent(route)["seed"] == 42


@respx.mock
async def test_seed_is_dropped_for_a_model_without_one_instead_of_failing_the_call(provider):
    # Sent anyway, it would be rejected locally as an unsupported parameter (a 400: no fallback).
    route = respx.post(OR_IMAGES).mock(return_value=ok({"b64_json": encoded(png())}))
    await provider.generate_image(GPT_IMAGE, ImageRequest(prompt="a fox", seed=42))
    assert "seed" not in sent(route)
    assert not provider.accepts_seed(GPT_IMAGE) and provider.accepts_seed(KLEIN)


async def test_seed_support_of_an_unregistered_model_comes_from_the_catalogue(tmp_path):
    other = ImageModelInfo(id="x/y", supported_parameters={"n": {"type": "range", "max": 1}})
    cache = ModelCache(tmp_path / "c.json")
    cache.save([other])
    use_image_registry(ImageRegistry.load(cache))
    async with httpx.AsyncClient() as client:
        p = OpenRouterProvider(client, Settings(_env_file=None))
        assert not p.accepts_seed("x/y")
        assert p.accepts_seed("not/listed")  # unknown everywhere: send it, validation decides


# -- quality ------------------------------------------------------------------------------------


@respx.mock
async def test_quality_defaults_to_the_registry_value_never_the_models_own(provider):
    route = respx.post(OR_IMAGES).mock(return_value=ok({"b64_json": encoded(png())}))
    await provider.generate_image(GPT_IMAGE, ImageRequest(prompt="a fox"))
    assert sent(route)["quality"] == "low"


@respx.mock
async def test_an_explicit_quality_is_sent_as_given(provider):
    route = respx.post(OR_IMAGES).mock(return_value=ok({"b64_json": encoded(png())}))
    await provider.generate_image(GPT_IMAGE, ImageRequest(prompt="a fox", quality="medium"))
    assert sent(route)["quality"] == "medium"


@respx.mock
async def test_a_model_without_quality_settings_gets_no_quality(provider):
    route = respx.post(OR_IMAGES).mock(return_value=ok({"b64_json": encoded(png())}))
    await provider.generate_image(KLEIN, ImageRequest(prompt="a fox"))
    assert "quality" not in sent(route)


@respx.mock
async def test_an_unsupported_quality_is_rejected_locally_without_a_paid_call(provider):
    route = respx.post(OR_IMAGES)
    with pytest.raises(ProviderError) as info:
        await provider.generate_image(GPT_IMAGE, ImageRequest(prompt="a fox", quality="ultra"))
    assert info.value.status == 400 and info.value.code == "local_validation"
    assert not route.called


@respx.mock
async def test_a_direct_provider_call_with_n_above_the_limit_is_rejected_locally(provider):
    # The gateway routes this case to another model first (test_plan_guard.py); a call that
    # bypasses the gateway still never silently clamps: a caller asking for 2 must not get 1.
    route = respx.post(OR_IMAGES)
    with pytest.raises(ProviderError, match="out of range"):
        await provider.generate_image(KLEIN, ImageRequest(prompt="a fox", n=2))
    assert not route.called


# -- media type ---------------------------------------------------------------------------------


@respx.mock
async def test_media_type_is_read_per_image_never_assumed(provider):
    respx.post(OR_IMAGES).mock(
        return_value=ok(
            {"b64_json": encoded(png())},
            {"b64_json": encoded(jpeg())},
            {"b64_json": encoded(b"opaque"), "media_type": "image/webp"},
            {"b64_json": encoded(SVG)},
            {"b64_json": encoded(b"opaque"), "output_format": "jpeg"},
            {"b64_json": encoded(b"opaque")},
        )
    )
    result = await provider.generate_image(GPT_IMAGE, ImageRequest(prompt="a fox", n=6))
    assert result.media_types == [
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/svg+xml",
        "image/jpeg",
        None,  # unrecognised: reported as unknown, not guessed as PNG
    ]


def test_sniff_media_type():
    assert sniff_media_type(png()) == "image/png"
    assert sniff_media_type(jpeg()) == "image/jpeg"
    assert sniff_media_type(webp()) == "image/webp"
    assert sniff_media_type(SVG) == "image/svg+xml"
    assert sniff_media_type(b"  <svg/>") == "image/svg+xml"
    assert sniff_media_type(b"GIF89a") is None


# -- USD estimate ---------------------------------------------------------------------------------


def spec(**overrides: object) -> ImageModelSpec:
    base: dict[str, object] = {
        "slug": "a/b",
        "name": "B",
        "tier": "pro",
        "billing_unit": "image",
        "price_measured_usd": 0.03,
        "supports_seed": True,
        "n_per_call": 6,
        "max_input_images": 4,
        "prompt_style": "nouns",
        "prompt_max_words": 60,
    }
    return merge(ModelEntry.model_validate(base | overrides), None)[0]


def test_per_image_billing_is_flat_times_n():
    assert estimate_image_usd(spec(), ImageRequest(prompt="x", n=4)) == pytest.approx(0.12)
    big = ImageRequest(prompt="x", width=2048, height=2048)
    assert estimate_image_usd(spec(), big) == pytest.approx(0.03)


def test_megapixel_billing_scales_with_output_size():
    mp = spec(billing_unit="megapixel", price_measured_usd=0.014)
    assert estimate_image_usd(mp, ImageRequest(prompt="x")) == pytest.approx(0.014)
    big = ImageRequest(prompt="x", width=2048, height=2048)
    assert estimate_image_usd(mp, big) == pytest.approx(0.056)  # 4 MP


def test_token_billing_uses_the_measured_price_for_the_quality():
    gpt = ImageRegistry.load(None).get(GPT_IMAGE)
    assert gpt is not None
    assert estimate_image_usd(gpt, ImageRequest(prompt="x")) == pytest.approx(0.006)  # low
    assert estimate_image_usd(gpt, ImageRequest(prompt="x", quality="high")) == pytest.approx(0.211)
    # medium was never measured: assume the dearest measured quality, never the cheapest
    assert estimate_image_usd(gpt, ImageRequest(prompt="x", quality="medium")) == pytest.approx(
        0.211
    )


def test_edits_use_the_measured_edit_price_else_a_pessimistic_markup():
    pro = ImageRegistry.load(None).get("black-forest-labs/flux.2-pro")
    assert pro is not None
    assert estimate_image_usd(pro, ImageRequest(prompt="x"), "edit") == pytest.approx(0.045)
    assert estimate_image_usd(spec(), ImageRequest(prompt="x"), "edit") == pytest.approx(
        0.03 * EDIT_MARKUP_DEFAULT
    )
    with_ref = ImageRequest(prompt="x", input_images=[png()])  # references imply an edit
    assert estimate_image_usd(spec(), with_ref) == pytest.approx(0.03 * EDIT_MARKUP_DEFAULT)


def test_listed_input_reference_prices_are_added():
    s = spec()
    priced = s.model_copy(
        update={"listed_pricing": [{"billable": "input_image", "unit": "image", "cost_usd": 0.003}]}
    )
    req = ImageRequest(prompt="x", input_images=[png(), png()])
    assert estimate_image_usd(priced, req, "merge") == pytest.approx(
        0.03 * EDIT_MARKUP_DEFAULT + 0.006
    )


def test_an_unmeasured_model_falls_back_to_its_listed_price_else_unknown():
    unmeasured = spec(price_measured_usd=None)
    assert estimate_image_usd(unmeasured, ImageRequest(prompt="x")) is None
    listed = unmeasured.model_copy(
        update={"listed_pricing": [{"billable": "output_image", "unit": "image", "cost_usd": 0.05}]}
    )
    assert estimate_image_usd(listed, ImageRequest(prompt="x")) == pytest.approx(0.05)
    token_listed = unmeasured.model_copy(
        update={"listed_pricing": [{"billable": "output_image", "unit": "token", "cost_usd": 3e-5}]}
    )
    assert estimate_image_usd(token_listed, ImageRequest(prompt="x")) is None
