"""OpenRouter provider + fallback rules. All HTTP is mocked with respx."""

from __future__ import annotations

import base64
import io
import json
from datetime import timedelta

import httpx
import pytest
import respx
from PIL import Image
from structlog.testing import capture_logs

from media_engine.core.config import Settings
from media_engine.gateway.errors import BadRequestError
from media_engine.gateway.openrouter_models import ImageModelInfo, ModelCache, sync_image_models
from media_engine.gateway.providers.cloudflare import CloudflareProvider
from media_engine.gateway.providers.groq import GroqProvider
from media_engine.gateway.providers.openrouter import OpenRouterProvider
from media_engine.gateway.router import Gateway, parse_ref
from media_engine.gateway.types import ImageRequest, LLMRequest

from .conftest import ACCOUNT, CF_CHAT, CF_RUN, GROQ_CHAT, NOW, cf_error, chat_ok

OR_IMAGES = "https://openrouter.ai/api/v1/images"
OR_CHAT = "https://openrouter.ai/api/v1/chat/completions"
GEMINI = "google/gemini-2.5-flash-image"
SEEDREAM = "bytedance-seed/seedream-4.5"
SCHNELL = f"{CF_RUN}/@cf/black-forest-labs/flux-1-schnell"


def _png(color: str = "white", size: int = 16) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, "PNG")
    return buf.getvalue()


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buf, "JPEG")
    return buf.getvalue()


def image_ok(cost: float | None = 0.039, n: int = 1) -> httpx.Response:
    body: dict = {
        "created": 1,
        "data": [{"b64_json": base64.b64encode(b"img%d" % i).decode()} for i in range(n)],
    }
    if cost is not None:
        body["usage"] = {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "cost": cost,
        }
    return httpx.Response(200, json=body)


def or_error(status: int, message: str = "boom", headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status, headers=headers, json={"error": {"code": status, "message": message}}
    )


@pytest.fixture
def or_settings() -> Settings:
    return Settings(
        _env_file=None,
        cloudflare_account_id=ACCOUNT,
        cloudflare_api_token="cf-test-token",
        groq_api_key="groq-test-key",
        openrouter_api_key="or-test-key",
    )


@pytest.fixture
def or_cache(tmp_path) -> ModelCache:
    cache = ModelCache(tmp_path / "models.json")
    cache.save(
        [
            ImageModelInfo(
                id=GEMINI,
                supported_parameters={
                    "aspect_ratio": {
                        "type": "enum",
                        "values": ["1:1", "2:3", "3:2", "16:9", "9:16"],
                    },
                    "input_references": {"type": "range", "min": 0, "max": 3},
                    "n": {"type": "range", "min": 1, "max": 1},
                },
                input_modalities=["image", "text"],
                output_modalities=["image", "text"],
            ),
            ImageModelInfo(
                id=SEEDREAM,
                supported_parameters={
                    "aspect_ratio": {"type": "enum", "values": ["1:1", "3:2", "16:9", "auto"]},
                    "resolution": {"type": "enum", "values": ["1K", "2K", "4K"]},
                    "n": {"type": "range", "min": 1, "max": 10},
                    "seed": {"type": "boolean"},
                    "input_references": {"type": "range", "min": 0, "max": 14},
                },
                input_modalities=["image", "text"],
                output_modalities=["image"],
            ),
        ]
    )
    return cache


@pytest.fixture
async def gw(or_settings, or_cache, sleeps, state):
    client = httpx.AsyncClient()

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    providers = {
        "cloudflare": CloudflareProvider(client, or_settings),
        "groq": GroqProvider(client, or_settings),
        "openrouter": OpenRouterProvider(client, or_settings, cache=or_cache),
    }
    chains = {
        # OpenRouter first so its failures exercise "next provider".
        "image_create": [
            parse_ref(f"openrouter/{GEMINI}"),
            parse_ref("cloudflare/@cf/black-forest-labs/flux-1-schnell"),
        ],
        "image_merge": [parse_ref(f"openrouter/{SEEDREAM}")],
        "reasoning": [
            parse_ref("openrouter/openai/gpt-oss-120b"),
            parse_ref("groq/openai/gpt-oss-120b"),
        ],
    }
    gateway = Gateway(
        or_settings,
        chains=chains,
        providers=providers,
        state=state,
        client=client,
        sleep=fake_sleep,
    )
    yield gateway
    await client.aclose()


@respx.mock
async def test_image_success_parses_cost_and_sends_expected_payload(gw):
    route = respx.post(OR_IMAGES).mock(return_value=image_ok(cost=0.039))

    result = await gw.image("image_create", ImageRequest(prompt="a cat", width=1024, height=1024))

    assert result.provider == "openrouter" and result.model == GEMINI
    assert result.output.images == [b"img0"]
    assert result.cost_usd == 0.039
    assert result.daily_cost_usd_total == 0.039
    assert gw.cost_meter.total_today == 0.039
    sent = route.calls[0].request
    assert sent.headers["authorization"] == "Bearer or-test-key"
    body = json.loads(sent.content)
    assert body["model"] == GEMINI and body["prompt"] == "a cat"
    assert body["aspect_ratio"] == "1:1"
    assert "resolution" not in body  # gemini has no `resolution` parameter
    assert "n" not in body


@respx.mock
async def test_daily_cost_total_accumulates(gw):
    respx.post(OR_IMAGES).mock(return_value=image_ok(cost=0.04))
    await gw.image("image_create", ImageRequest(prompt="a"))
    result = await gw.image("image_create", ImageRequest(prompt="b"))
    assert result.daily_cost_usd_total == pytest.approx(0.08)


@respx.mock
async def test_missing_cost_in_response_is_none(gw):
    respx.post(OR_IMAGES).mock(return_value=image_ok(cost=None))
    result = await gw.image("image_create", ImageRequest(prompt="a"))
    assert result.cost_usd is None and result.daily_cost_usd_total is None


@respx.mock
async def test_chat_parses_usage_and_cost(gw):
    route = respx.post(OR_CHAT).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                    "cost": 0.0001,
                },
            },
        )
    )

    result = await gw.chat(
        "reasoning", LLMRequest(messages=[{"role": "user", "content": "x"}], reasoning_effort="low")
    )

    assert result.provider == "openrouter"
    assert result.output.text == "hello"
    assert result.output.usage["total_tokens"] == 7
    assert result.cost_usd == 0.0001
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "openai/gpt-oss-120b" and body["reasoning"] == {"effort": "low"}


@respx.mock
async def test_402_marks_openrouter_exhausted_logs_error_and_goes_next(gw, state):
    route = respx.post(OR_IMAGES).mock(return_value=or_error(402, "insufficient credits"))
    cf = respx.post(SCHNELL).mock(
        return_value=httpx.Response(
            200, json={"result": {"image": base64.b64encode(b"cf").decode()}}
        )
    )

    with capture_logs() as logs:
        result = await gw.image("image_create", ImageRequest(prompt="a"))

    assert result.provider == "cloudflare" and result.is_fallback is True
    assert route.call_count == 1  # not retried
    assert cf.call_count == 1
    reason = state.unavailable_reason("openrouter", GEMINI)
    assert reason is not None and "provider_exhausted_until" in reason
    assert reason.endswith((NOW + timedelta(hours=1)).isoformat())
    fallbacks = [e for e in logs if e["event"] == "gateway.fallback"]
    assert fallbacks and fallbacks[0]["log_level"] == "error"
    assert fallbacks[0]["reason"] == "openrouter_out_of_credits"

    # While exhausted, OpenRouter is skipped without a call.
    await gw.image("image_create", ImageRequest(prompt="b"))
    assert route.call_count == 1


@respx.mock
async def test_429_short_retry_after_is_honoured(gw, sleeps):
    route = respx.post(OR_IMAGES).mock(
        side_effect=[or_error(429, "slow down", {"retry-after": "2"}), image_ok()]
    )

    result = await gw.image("image_create", ImageRequest(prompt="a"))

    assert result.provider == "openrouter" and route.call_count == 2
    assert sleeps == [2.0]


@respx.mock
async def test_429_long_retry_after_goes_next_with_cooldown(gw, state, sleeps):
    route = respx.post(OR_IMAGES).mock(return_value=or_error(429, "slow", {"retry-after": "300"}))
    respx.post(SCHNELL).mock(
        return_value=httpx.Response(
            200, json={"result": {"image": base64.b64encode(b"cf").decode()}}
        )
    )

    result = await gw.image("image_create", ImageRequest(prompt="a"))

    assert result.provider == "cloudflare" and route.call_count == 1 and sleeps == []
    assert "cooling_down" in (state.unavailable_reason("openrouter", GEMINI) or "")


@respx.mock
async def test_502_retries_once_then_succeeds(gw, sleeps):
    route = respx.post(OR_IMAGES).mock(side_effect=[or_error(502, "upstream"), image_ok()])

    result = await gw.image("image_create", ImageRequest(prompt="a"))

    assert result.provider == "openrouter" and route.call_count == 2
    assert sleeps == [0.5]


@respx.mock
async def test_502_twice_goes_next_provider(gw):
    route = respx.post(OR_IMAGES).mock(return_value=or_error(502, "upstream"))
    respx.post(SCHNELL).mock(
        return_value=httpx.Response(
            200, json={"result": {"image": base64.b64encode(b"cf").decode()}}
        )
    )

    result = await gw.image("image_create", ImageRequest(prompt="a"))

    assert result.provider == "cloudflare" and route.call_count == 2


@respx.mock
@pytest.mark.parametrize("status", [401, 403, 404, 422])
async def test_other_4xx_still_go_to_next_provider(gw, status):
    route = respx.post(OR_IMAGES).mock(return_value=or_error(status))
    respx.post(SCHNELL).mock(
        return_value=httpx.Response(
            200, json={"result": {"image": base64.b64encode(b"cf").decode()}}
        )
    )

    result = await gw.image("image_create", ImageRequest(prompt="a"))

    assert result.provider == "cloudflare" and route.call_count == 1


@respx.mock
async def test_http_400_from_openrouter_raises_and_does_not_fall_back(gw):
    respx.post(OR_IMAGES).mock(return_value=or_error(400, "bad prompt"))
    cf = respx.post(SCHNELL).mock(return_value=httpx.Response(200, json={}))

    with pytest.raises(BadRequestError):
        await gw.image("image_create", ImageRequest(prompt="a"))
    assert cf.call_count == 0


@respx.mock
async def test_unsupported_resolution_is_rejected_locally_without_a_call(gw):
    route = respx.post(OR_IMAGES).mock(return_value=image_ok())

    with pytest.raises(BadRequestError) as exc:
        await gw.image("image_merge", ImageRequest(prompt="a", resolution="8K"))

    assert "resolution" in str(exc.value) and "8K" in str(exc.value)
    assert route.call_count == 0


@respx.mock
async def test_param_the_model_does_not_have_is_rejected_locally(gw):
    route = respx.post(OR_IMAGES).mock(return_value=image_ok())

    with pytest.raises(BadRequestError) as exc:
        await gw.image("image_create", ImageRequest(prompt="a", resolution="1K"))  # gemini: none

    assert "'resolution' is not supported" in str(exc.value)
    assert route.call_count == 0


@respx.mock
async def test_too_many_input_references_rejected_locally(gw):
    route = respx.post(OR_IMAGES).mock(return_value=image_ok())

    with pytest.raises(BadRequestError) as exc:
        await gw.image("image_create", ImageRequest(prompt="a", input_images=[_png()] * 4))  # max 3

    assert "input_references" in str(exc.value)
    assert route.call_count == 0


@respx.mock
async def test_model_missing_from_cache_is_rejected_locally(gw):
    route = respx.post(OR_IMAGES).mock(return_value=image_ok())
    gw.chains["image_create"] = [parse_ref("openrouter/some/unknown-model")]

    with pytest.raises(BadRequestError) as exc:
        await gw.image("image_create", ImageRequest(prompt="a"))

    assert "not in the OpenRouter image model cache" in str(exc.value)
    assert route.call_count == 0


@respx.mock
async def test_no_cache_file_skips_validation_and_sends_everything(or_settings, tmp_path):
    client = httpx.AsyncClient()
    provider = OpenRouterProvider(client, or_settings, cache=ModelCache(tmp_path / "missing.json"))
    route = respx.post(OR_IMAGES).mock(return_value=image_ok())
    try:
        await provider.generate_image(GEMINI, ImageRequest(prompt="a", resolution="8K"))
    finally:
        await client.aclose()
    assert json.loads(route.calls[0].request.content)["resolution"] == "8K"


@respx.mock
async def test_input_references_are_encoded_as_image_url_data_urls(gw):
    route = respx.post(OR_IMAGES).mock(return_value=image_ok())
    first, second = _png("white"), _jpeg()

    await gw.image("image_merge", ImageRequest(prompt="combine", input_images=[first, second]))

    body = json.loads(route.calls[0].request.content)
    refs = body["input_references"]
    assert [r["type"] for r in refs] == ["image_url", "image_url"]
    url0, url1 = refs[0]["image_url"]["url"], refs[1]["image_url"]["url"]
    assert url0.startswith("data:image/png;base64,") and url1.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url0.split(",", 1)[1]) == first
    assert base64.b64decode(url1.split(",", 1)[1]) == second


@respx.mock
async def test_derived_size_adapts_to_model_support(gw):
    route = respx.post(OR_IMAGES).mock(return_value=image_ok())

    # 512x512 requested; seedream supports 1K/2K/4K only -> smallest supported tier.
    await gw.image("image_merge", ImageRequest(prompt="a", width=512, height=512))
    body = json.loads(route.calls[0].request.content)
    assert body["resolution"] == "1K" and body["aspect_ratio"] == "1:1"

    # 1280x720 -> closest supported ratio of the model
    await gw.image("image_merge", ImageRequest(prompt="a", width=1280, height=720))
    assert json.loads(route.calls[1].request.content)["aspect_ratio"] == "16:9"


@respx.mock
async def test_n_and_seed_sent_when_supported(gw):
    route = respx.post(OR_IMAGES).mock(return_value=image_ok(n=2))

    result = await gw.image("image_merge", ImageRequest(prompt="a", n=2, seed=7))

    body = json.loads(route.calls[0].request.content)
    assert body["n"] == 2 and body["seed"] == 7
    assert len(result.output.images) == 2


@respx.mock
async def test_sync_writes_cache_and_pricing(tmp_path):
    respx.get("https://openrouter.ai/api/v1/images/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": SEEDREAM,
                        "architecture": {
                            "input_modalities": ["image", "text"],
                            "output_modalities": ["image"],
                        },
                        "supported_parameters": {"resolution": {"type": "enum", "values": ["1K"]}},
                    }
                ]
            },
        )
    )
    respx.get(f"https://openrouter.ai/api/v1/images/models/{SEEDREAM}/endpoints").mock(
        return_value=httpx.Response(
            200,
            json={
                "endpoints": [
                    {"pricing": [{"billable": "output_image", "unit": "image", "cost_usd": 0.04}]}
                ]
            },
        )
    )
    cache = ModelCache(tmp_path / "c.json")
    async with httpx.AsyncClient() as client:
        await sync_image_models(client, "or-test-key", cache, price_for=[SEEDREAM])

    fresh = ModelCache(tmp_path / "c.json")
    assert fresh.load() and fresh.model_ids() == [SEEDREAM]
    info = fresh.get(SEEDREAM)
    assert info and info.pricing[0]["cost_usd"] == 0.04
    assert "or-test-key" not in (tmp_path / "c.json").read_text()


@respx.mock
async def test_openrouter_is_skipped_without_a_key(settings, sleeps, state):
    # `settings` fixture has no OpenRouter key: chain entry is skipped, not called.
    route = respx.post(OR_CHAT).mock(return_value=chat_ok())
    respx.post(CF_CHAT).mock(return_value=cf_error(429, 3036))
    respx.post(GROQ_CHAT).mock(return_value=cf_error(500, 0))
    gateway = Gateway(settings, state=state)
    try:
        with pytest.raises(Exception) as exc:
            await gateway.chat("reasoning", LLMRequest(messages=[{"role": "user", "content": "x"}]))
    finally:
        await gateway.aclose()
    assert "provider_not_configured" in str(exc.value)
    assert route.call_count == 0
