"""How real providers signal a content refusal, and that the existing fallback rules still hold
next to the breaker. All HTTP is mocked with respx."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx
from structlog.testing import capture_logs

from media_engine.core.config import Settings
from media_engine.gateway.breaker import BreakerState
from media_engine.gateway.errors import (
    AllProvidersFailed,
    BadRequestError,
    ContentRefused,
    ProviderError,
)
from media_engine.gateway.fallback import Action, decide
from media_engine.gateway.providers.openrouter import OpenRouterProvider
from media_engine.gateway.types import ImageRequest, LLMRequest

from .conftest import (
    CF_CHAT,
    CF_FIRST_LLM,
    CF_RUN,
    GROQ_CHAT,
    NOW,
    cf_error,
    chat_ok,
)

CF_SCHNELL = f"{CF_RUN}/@cf/black-forest-labs/flux-1-schnell"
CF_KLEIN_4B = f"{CF_RUN}/@cf/black-forest-labs/flux-2-klein-4b"
OR_IMAGES = "https://openrouter.ai/api/v1/images"
REQ = LLMRequest(messages=[{"role": "user", "content": "hello"}])


@pytest.mark.parametrize(
    "error",
    [
        ProviderError("cloudflare", "m", "nsfw", status=400, code=3030),
        ProviderError("groq", "m", "x", status=400, refused=True),
        ProviderError("openrouter", "m", "x", status=403, refused=True),
        ProviderError("cloudflare", "m", "x", status=400, code=5035, refused=True),
    ],
)
def test_refusal_is_decided_before_the_400_and_5035_rules(error):
    decision = decide(error, 1, NOW)
    assert decision.action is Action.REFUSE
    assert not decision.disable_model and decision.exhausted_until is None


@respx.mock
async def test_cloudflare_3030_raises_content_refused_without_trying_klein(gateway):
    schnell = respx.post(CF_SCHNELL).mock(
        return_value=cf_error(400, 3030, "AiError: Input prompt contains NSFW content")
    )
    klein = respx.post(CF_KLEIN_4B).mock(return_value=httpx.Response(200, json={}))

    with pytest.raises(ContentRefused) as exc:
        await gateway.image("image_create", ImageRequest(prompt="x"))

    assert exc.value.code == 3030 and exc.value.provider == "cloudflare"
    assert schnell.call_count == 1 and klein.call_count == 0
    assert gateway.breakers.get("cloudflare").consecutive_failures == 0


@respx.mock
async def test_chat_content_filter_finish_reason_is_a_refusal(gateway):
    respx.post(CF_CHAT).mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]},
        )
    )
    groq = respx.post(GROQ_CHAT).mock(return_value=chat_ok("never"))

    with pytest.raises(ContentRefused):
        await gateway.chat(CF_FIRST_LLM, REQ)

    assert groq.call_count == 0


@pytest.fixture
async def openrouter():
    async with httpx.AsyncClient() as client:
        yield OpenRouterProvider(client, Settings(_env_file=None, openrouter_api_key="or-test"))


@respx.mock
async def test_openrouter_403_with_moderation_reasons_is_a_refusal(openrouter):
    respx.post(OR_IMAGES).mock(
        return_value=httpx.Response(
            403,
            json={
                "error": {
                    "code": 403,
                    "message": "Input was flagged",
                    "metadata": {"reasons": ["sexual"], "provider_name": "x", "model_slug": "y"},
                }
            },
        )
    )
    with pytest.raises(ProviderError) as exc:
        await openrouter.generate_image("google/gemini-2.5-flash-image", ImageRequest(prompt="x"))
    assert exc.value.refused is True


@respx.mock
async def test_openrouter_403_without_reasons_is_not_a_refusal(openrouter):
    # e.g. a spending limit: must keep the old behaviour (move on), not stop the chain.
    respx.post(OR_IMAGES).mock(
        return_value=httpx.Response(403, json={"error": {"code": 403, "message": "Key limit"}})
    )
    with pytest.raises(ProviderError) as exc:
        await openrouter.generate_image("google/gemini-2.5-flash-image", ImageRequest(prompt="x"))
    assert exc.value.refused is False
    assert decide(exc.value, 1, NOW).action is Action.NEXT


# -- the CLAUDE.md rules keep working with the breaker in place ---------------------------------


@respx.mock
async def test_3036_exhaustion_is_fallback_state_not_a_breaker_failure(gateway, state):
    respx.post(CF_CHAT).mock(return_value=cf_error(429, 3036))
    respx.post(GROQ_CHAT).mock(return_value=chat_ok())

    await gateway.chat(CF_FIRST_LLM, REQ)

    until = datetime(2026, 11, 14, tzinfo=UTC).isoformat()
    assert state.unavailable_reason("cloudflare", "x") == f"provider_exhausted_until_{until}"
    breaker = gateway.breakers.get("cloudflare")
    assert breaker.state is BreakerState.CLOSED and breaker.consecutive_failures == 0


@respx.mock
async def test_5035_disables_the_model_without_touching_the_breaker(gateway, state):
    respx.post(CF_SCHNELL).mock(return_value=cf_error(400, 5035))
    respx.post(CF_KLEIN_4B).mock(
        return_value=httpx.Response(200, json={"result": {"image": "aW1n"}})
    )

    result = await gateway.image("image_create", ImageRequest(prompt="x"))

    assert result.model == "@cf/black-forest-labs/flux-2-klein-4b"
    assert gateway.breakers.get("cloudflare").consecutive_failures == 0


@respx.mock
async def test_cloudflare_5xx_on_every_model_opens_its_breaker(gateway):
    # Both Cloudflare image models 503. Each counts once (after its retry) toward the one
    # Cloudflare breaker, using the shipped providers.yaml threshold.
    schnell = respx.post(CF_SCHNELL).mock(return_value=httpx.Response(503))
    respx.post(CF_KLEIN_4B).mock(return_value=httpx.Response(503))
    breaker = gateway.breakers.get("cloudflare")

    requests = 0
    while breaker.state is not BreakerState.OPEN:
        with pytest.raises(AllProvidersFailed):  # the openrouter slot is skipped (paid off)
            await gateway.image("image_create", ImageRequest(prompt="x"))
        requests += 1
        assert requests <= breaker.settings.failure_threshold, "breaker never opened"

    before = schnell.call_count
    with pytest.raises(AllProvidersFailed) as exc:
        await gateway.image("image_create", ImageRequest(prompt="x"))
    assert schnell.call_count == before
    reasons = [a.reason for a in exc.value.attempts if a.provider == "cloudflare"]
    assert reasons == ["circuit_open", "circuit_open"]


# -- M5: Cloudflare 400 wording, distinct logging, metering billed refusals ---------------------


@pytest.mark.parametrize(
    "message",
    [
        "AiError: Input prompt contains NSFW content.",
        "Request blocked: content policy violation",
        "Output flagged by content filter",
        "content_policy_violation: request rejected",  # OpenAI-style code, underscores
        "blocked by content-moderation",
        "Unsafe_content detected",
    ],
)
@respx.mock
async def test_cloudflare_400_with_refusal_wording_is_a_refusal(gateway, message):
    schnell = respx.post(CF_SCHNELL).mock(return_value=cf_error(400, 1000, message))
    klein = respx.post(CF_KLEIN_4B).mock(return_value=httpx.Response(200, json={}))

    with pytest.raises(ContentRefused):
        await gateway.image("image_create", ImageRequest(prompt="x"))

    assert schnell.call_count == 1 and klein.call_count == 0


@pytest.mark.parametrize(
    "message",
    [
        "Invalid content-type: expected multipart/form-data",
        "Invalid content_type",
        "content length too large",
    ],
)
def test_bare_content_in_a_400_is_not_a_refusal(message):
    # "content" alone would turn our own bugs into refusals; only specific phrases count.
    error = ProviderError("cloudflare", "m", message, status=400, code=1000)
    assert decide(error, 1, NOW).action is Action.RAISE


@respx.mock
async def test_unclassified_cloudflare_400_is_logged_under_its_own_event(gateway):
    respx.post(CF_SCHNELL).mock(return_value=cf_error(400, 1000, "Invalid content-type"))

    with capture_logs() as logs, pytest.raises(BadRequestError):
        await gateway.image("image_create", ImageRequest(prompt="x"))

    events = [e["event"] for e in logs]
    assert "gateway.cloudflare_bad_request_unclassified" in events
    assert "gateway.bad_request" not in events


@respx.mock
async def test_other_providers_400_keeps_the_plain_bad_request_event(gateway):
    respx.post(CF_CHAT).mock(return_value=cf_error(429, 3036))
    respx.post(GROQ_CHAT).mock(return_value=httpx.Response(400, json={"error": {"message": "x"}}))

    with capture_logs() as logs, pytest.raises(BadRequestError):
        await gateway.chat(CF_FIRST_LLM, REQ)

    events = [e["event"] for e in logs]
    assert "gateway.bad_request" in events
    assert "gateway.cloudflare_bad_request_unclassified" not in events


@respx.mock
async def test_a_billed_content_filter_reply_is_metered(gateway):
    respx.post(CF_CHAT).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 0, "neurons": 14.5},
            },
        )
    )

    with pytest.raises(ContentRefused) as exc:
        await gateway.chat(CF_FIRST_LLM, REQ)

    assert exc.value.neurons == pytest.approx(14.5)
    assert gateway.meter.total_today == pytest.approx(14.5)


@respx.mock
async def test_an_unbilled_refusal_meters_nothing(gateway):
    respx.post(CF_SCHNELL).mock(return_value=cf_error(400, 3030, "NSFW"))

    with pytest.raises(ContentRefused) as exc:
        await gateway.image("image_create", ImageRequest(prompt="x"))

    assert exc.value.neurons is None
    assert gateway.meter.total_today == 0
