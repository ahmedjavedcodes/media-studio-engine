"""The fallback path on the PRODUCTION chain config (gateway/chains.yaml), not a synthetic one.

The mechanics tests (test_fallback_rules.py and friends) use synthetic `cf_first_*` chains to
exercise Cloudflare's error codes. These tests drive the chains that actually ship, so they fail
if production stops doing what the policy says: agents Groq first with Cloudflare as fallback
(one Groq model per agent), free-tier images Cloudflare first, paid entries skipped while
ALLOW_PAID is off. The ORDER itself is also asserted in tests/unit/harness/test_config.py."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from media_engine.core.config import Settings
from media_engine.gateway.errors import AllProvidersFailed
from media_engine.gateway.fallback import AvailabilityState
from media_engine.gateway.router import Gateway
from media_engine.gateway.types import ImageRequest, LLMRequest

from .conftest import ACCOUNT, CF_CHAT, CF_RUN, GROQ_CHAT, NOW, cf_error, chat_ok

REQ = LLMRequest(messages=[{"role": "user", "content": "hello"}])
SCHNELL = f"{CF_RUN}/@cf/black-forest-labs/flux-1-schnell"
KLEIN_4B = f"{CF_RUN}/@cf/black-forest-labs/flux-2-klein-4b"


@pytest.fixture
async def prod():
    """The real Gateway with the real chains.yaml (no `chains=` override). ALLOW_PAID off."""
    settings = Settings(
        _env_file=None,
        cloudflare_account_id=ACCOUNT,
        cloudflare_api_token="cf-test",
        groq_api_key="groq-test",
        openrouter_api_key="or-test",  # configured, so only ALLOW_PAID keeps it off
    )

    async def no_sleep(_: float) -> None:
        return None

    gw = Gateway(settings, state=AvailabilityState(now=lambda: NOW), sleep=no_sleep)
    yield gw
    await gw.aclose()


def model_of(route: respx.Route, i: int = 0) -> str:
    return str(json.loads(route.calls[i].request.content)["model"])


@respx.mock
@pytest.mark.parametrize(
    ("task", "groq_model"),
    [
        ("fast_llm", "openai/gpt-oss-20b"),
        ("reasoning", "openai/gpt-oss-120b"),
        ("vision", "qwen/qwen3.8-27b"),
    ],
)
async def test_agent_chains_are_served_by_their_own_groq_model_first(prod, task, groq_model):
    groq = respx.post(GROQ_CHAT).mock(return_value=chat_ok())
    cf = respx.post(CF_CHAT).mock(return_value=chat_ok())
    result = await prod.chat(task, REQ)
    assert (result.provider, result.model, result.is_fallback) == ("groq", groq_model, False)
    assert model_of(groq) == groq_model
    assert not cf.called  # Cloudflare's neuron pool is left for images


@respx.mock
@pytest.mark.parametrize(
    ("task", "cf_model"),
    [
        ("fast_llm", "@cf/openai/gpt-oss-20b"),
        ("reasoning", "@cf/openai/gpt-oss-120b"),
        ("vision", "@cf/meta/llama-4-scout-17b-16e-instruct"),
    ],
)
async def test_groq_rate_limited_falls_back_to_cloudflare(prod, task, cf_model):
    respx.post(GROQ_CHAT).mock(
        return_value=httpx.Response(429, headers={"retry-after": "120"}, json={"error": {}})
    )
    cf = respx.post(CF_CHAT).mock(return_value=chat_ok())
    result = await prod.chat(task, REQ)
    assert (result.provider, result.model, result.is_fallback) == ("cloudflare", cf_model, True)
    assert model_of(cf) == cf_model


@respx.mock
async def test_groq_down_then_cloudflare_exhausted_ends_without_a_paid_call(prod):
    respx.post(GROQ_CHAT).mock(return_value=httpx.Response(503))
    respx.post(CF_CHAT).mock(return_value=cf_error(429, 3036))
    paid = respx.post("https://openrouter.ai/api/v1/chat/completions")
    with pytest.raises(AllProvidersFailed) as exc:
        await prod.chat("reasoning", REQ)
    assert not paid.called
    trail = [(a.provider, a.outcome, a.reason) for a in exc.value.attempts]
    assert [p for p, _, _ in trail] == ["groq", "cloudflare", "openrouter"]
    assert trail[2][1:] == ("skipped", "paid_disabled")


@respx.mock
async def test_free_images_start_on_cloudflare_schnell(prod):
    schnell = respx.post(SCHNELL).mock(
        return_value=httpx.Response(200, json={"result": {"image": "aGk="}})
    )
    result = await prod.image("image_create", ImageRequest(prompt="a"), plan="free")
    assert (result.provider, result.model) == ("cloudflare", "@cf/black-forest-labs/flux-1-schnell")
    assert schnell.called


@respx.mock
async def test_schnell_failing_falls_back_to_klein_then_never_to_a_paid_model(prod):
    respx.post(SCHNELL).mock(return_value=httpx.Response(503))
    respx.post(KLEIN_4B).mock(return_value=httpx.Response(503))
    paid = respx.post("https://openrouter.ai/api/v1/images")
    with pytest.raises(AllProvidersFailed) as exc:
        await prod.image("image_create", ImageRequest(prompt="a"), plan="free")
    assert not paid.called
    assert [a.model for a in exc.value.attempts][:2] == [
        "@cf/black-forest-labs/flux-1-schnell",
        "@cf/black-forest-labs/flux-2-klein-4b",
    ]
    assert exc.value.attempts[-1].reason == "placeholder"  # paid slot pending E2/E3
