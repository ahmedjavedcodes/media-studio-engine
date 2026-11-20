"""Replays a REAL OpenRouter /images response (recorded once, paid $0.00596, by
playground/record_openrouter_response.py) through the production gateway path. Pins the
response field names to what OpenRouter actually returns, not to what the docs suggest."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
import respx

from media_engine.core.config import Settings
from media_engine.core.images import inspect_image, sniff_media_type
from media_engine.gateway.fallback import AvailabilityState
from media_engine.gateway.image_registry import ImageRegistry, use_image_registry
from media_engine.gateway.openrouter_models import ImageModelInfo, ModelCache
from media_engine.gateway.router import Gateway, parse_ref
from media_engine.gateway.types import ImageRequest

from .conftest import NOW
from .test_image_registry import GPT_IMAGE_INFO

FIXTURE = Path(__file__).resolve().parents[1].parent / "fixtures/openrouter/gpt-image-2.images.json"
RECORDED = json.loads(FIXTURE.read_text(encoding="utf-8"))
MODEL = "openai/gpt-image-2"


@pytest.fixture
async def gw(tmp_path):
    cache = ModelCache(tmp_path / "catalogue.json")
    cache.save([ImageModelInfo(**vars(GPT_IMAGE_INFO))])
    use_image_registry(ImageRegistry.load(cache))  # the real models.yaml
    settings = Settings(_env_file=None, openrouter_api_key="or-test-key", allow_paid=True)
    gateway = Gateway(
        settings,
        chains={"image_create": [parse_ref(f"openrouter/{MODEL}")]},
        state=AvailabilityState(now=lambda: NOW),
    )
    yield gateway
    await gateway.aclose()


def test_the_recording_is_a_real_successful_response():
    assert RECORDED["status"] == 200
    assert RECORDED["endpoint"] == "POST /api/v1/images"
    body = RECORDED["body"]
    assert set(body) >= {"created", "data", "usage"}
    (item,) = body["data"]
    assert set(item) == {"b64_json", "media_type"}  # the real field names
    assert body["usage"]["cost"] == pytest.approx(0.00596)
    assert body["usage"]["completion_tokens_details"]["image_tokens"] == 196  # quality low


@respx.mock
async def test_production_sends_the_recorded_request_and_parses_the_recorded_response(gw):
    route = respx.post("https://openrouter.ai/api/v1/images").mock(
        return_value=httpx.Response(RECORDED["status"], json=RECORDED["body"])
    )
    result = await gw.image(
        "image_create",
        ImageRequest(prompt=RECORDED["request"]["prompt"], width=1024, height=1024),
        plan="free",
    )
    # what production sends today is exactly what was recorded (quality explicit, low)
    assert json.loads(route.calls[0].request.content) == RECORDED["request"]

    (image,) = result.output.images
    assert image == base64.b64decode(RECORDED["body"]["data"][0]["b64_json"])
    assert result.output.media_types == ["image/png"]  # read from the response
    assert sniff_media_type(image) == "image/png"  # and the bytes agree
    info = inspect_image(image)
    assert (info.mime, info.width, info.height) == ("image/png", 1024, 1024)
    assert result.cost_usd == pytest.approx(0.00596)  # usage.cost, as billed


@respx.mock
async def test_media_type_is_read_from_the_response_not_assumed(gw):
    body = json.loads(json.dumps(RECORDED["body"]))
    body["data"][0]["media_type"] = "image/webp"  # the response is the authority
    respx.post("https://openrouter.ai/api/v1/images").mock(
        return_value=httpx.Response(200, json=body)
    )
    result = await gw.image("image_create", ImageRequest(prompt="x"), plan="free")
    assert result.output.media_types == ["image/webp"]

    del body["data"][0]["media_type"]  # missing: fall back to the bytes, still not assumed
    respx.post("https://openrouter.ai/api/v1/images").mock(
        return_value=httpx.Response(200, json=body)
    )
    result = await gw.image("image_create", ImageRequest(prompt="x"), plan="free")
    assert result.output.media_types == ["image/png"]
