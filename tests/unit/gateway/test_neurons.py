"""Neuron accounting: actual (Cloudflare LLM usage) vs registry estimate (images)."""

from __future__ import annotations

import base64
import io
from datetime import UTC, datetime

import httpx
import respx
from PIL import Image

from media_engine.gateway.budget import DailyNeuronMeter, estimate_image_neurons
from media_engine.gateway.registry import get_card
from media_engine.gateway.types import ImageRequest, LLMRequest

from .conftest import CF_CHAT, CF_RUN, GROQ_CHAT, NOW, cf_error, chat_ok

REQ = LLMRequest(messages=[{"role": "user", "content": "x"}])
SCHNELL = f"{CF_RUN}/@cf/black-forest-labs/flux-1-schnell"


def _llm_with_neurons(neurons: float) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": 9, "neurons": neurons},
        },
    )


def _image_ok() -> httpx.Response:
    return httpx.Response(200, json={"result": {"image": base64.b64encode(b"i").decode()}})


def _png(size: int = 256) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), "white").save(buf, "PNG")
    return buf.getvalue()


@respx.mock
async def test_cloudflare_llm_uses_actual_neurons_not_estimate(gateway):
    respx.post(CF_CHAT).mock(return_value=_llm_with_neurons(2.5))

    result = await gateway.chat("fast_llm", REQ)

    assert result.neurons == 2.5
    assert result.neurons_est is None
    assert result.daily_neurons_total == 2.5
    assert gateway.meter.actual == 2.5 and gateway.meter.estimated == 0


@respx.mock
async def test_cloudflare_image_uses_registry_estimate(gateway):
    respx.post(SCHNELL).mock(return_value=_image_ok())

    # 1024x1024 = 4 tiles * 4.80 + default 4 steps * 9.60
    result = await gateway.image("image_create", ImageRequest(prompt="p"))

    assert result.neurons is None
    assert result.neurons_est == 57.6
    assert result.daily_neurons_total == 57.6
    assert gateway.meter.estimated == 57.6 and gateway.meter.actual == 0


@respx.mock
async def test_daily_total_mixes_actual_and_estimate(gateway):
    respx.post(CF_CHAT).mock(return_value=_llm_with_neurons(2.5))
    respx.post(SCHNELL).mock(return_value=_image_ok())

    await gateway.chat("fast_llm", REQ)
    result = await gateway.image("image_create", ImageRequest(prompt="p", steps=8))

    # image: 4 * 4.80 + 8 * 9.60 = 96.0 ; total = 2.5 + 96.0
    assert result.neurons_est == 96.0
    assert result.daily_neurons_total == 98.5
    assert gateway.meter.remaining == 10_000 - 98.5


@respx.mock
async def test_cloudflare_llm_without_neurons_in_usage_counts_nothing(gateway):
    respx.post(CF_CHAT).mock(return_value=chat_ok())

    result = await gateway.chat("fast_llm", REQ)

    assert result.neurons is None and result.neurons_est is None
    assert gateway.meter.total_today == 0


@respx.mock
async def test_groq_calls_are_not_counted_against_cloudflare(gateway):
    respx.post(CF_CHAT).mock(return_value=cf_error(429, 3036))
    respx.post(GROQ_CHAT).mock(return_value=_llm_with_neurons(999))

    result = await gateway.chat("fast_llm", REQ)

    assert result.provider == "groq"
    assert result.neurons is None and result.daily_neurons_total is None
    assert gateway.meter.total_today == 0


def test_meter_resets_at_utc_midnight():
    clock = {"now": NOW}
    meter = DailyNeuronMeter(now=lambda: clock["now"])
    meter.add(actual=100.0, estimate=None)
    meter.add(actual=None, estimate=50.0)
    assert meter.total_today == 150.0

    clock["now"] = datetime(2026, 11, 14, 0, 0, 1, tzinfo=UTC)
    assert meter.total_today == 0
    assert meter.add(actual=3.0, estimate=None) == 3.0


def test_estimates_for_flux2_models():
    klein4 = get_card("cloudflare", "@cf/black-forest-labs/flux-2-klein-4b")
    klein9 = get_card("cloudflare", "@cf/black-forest-labs/flux-2-klein-9b")
    assert klein4 and klein9
    edit = ImageRequest(prompt="p", input_images=[_png()])
    # klein-4b: 4 output tiles * 26.05 + 1 input tile * 5.37
    assert estimate_image_neurons(klein4, edit) == 109.57
    # klein-9b: first MP 1363.64 + one input MP 181.82
    assert estimate_image_neurons(klein9, edit) == 1545.46
