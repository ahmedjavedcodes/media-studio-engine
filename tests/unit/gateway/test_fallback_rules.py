"""One test per switching rule in CLAUDE.md. All providers are mocked with respx."""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import httpx
import pytest
import respx

from media_engine.gateway.errors import AllProvidersFailed, BadRequestError
from media_engine.gateway.fallback import AvailabilityState
from media_engine.gateway.router import Gateway
from media_engine.gateway.types import ImageRequest, LLMRequest

from .conftest import CF_CHAT, CF_RUN, GROQ_CHAT, NOW, cf_error, chat_ok

REQ = LLMRequest(messages=[{"role": "user", "content": "hello"}])
CF_SCHNELL = f"{CF_RUN}/@cf/black-forest-labs/flux-1-schnell"
CF_KLEIN_4B = f"{CF_RUN}/@cf/black-forest-labs/flux-2-klein-4b"
CF_120B = "@cf/openai/gpt-oss-120b"


@respx.mock
async def test_happy_path_uses_cloudflare_first(gateway):
    cf = respx.post(CF_CHAT).mock(return_value=chat_ok("from cf"))
    groq = respx.post(GROQ_CHAT).mock(return_value=chat_ok("from groq"))

    result = await gateway.chat("reasoning", REQ)

    assert result.provider == "cloudflare"
    assert result.model == CF_120B
    assert result.output.text == "from cf"
    assert result.is_fallback is False
    assert cf.call_count == 1 and groq.call_count == 0


@respx.mock
async def test_cf_3036_marks_exhausted_until_midnight_and_skips_without_retry(gateway, state):
    cf = respx.post(CF_CHAT).mock(return_value=cf_error(429, 3036, "daily limit"))
    groq = respx.post(GROQ_CHAT).mock(return_value=chat_ok("from groq"))

    result = await gateway.chat("reasoning", REQ)

    assert result.provider == "groq" and result.is_fallback is True
    assert cf.call_count == 1  # no retry
    until = datetime(2026, 11, 14, tzinfo=UTC).isoformat()
    assert state.unavailable_reason("cloudflare", CF_120B) == f"provider_exhausted_until_{until}"

    # The next call (any task) doesn't even touch Cloudflare.
    await gateway.chat("fast_llm", REQ)
    assert cf.call_count == 1
    assert groq.call_count == 2


@respx.mock
async def test_cf_exhaustion_ends_at_utc_midnight(settings, sleeps):
    clock = {"now": NOW}
    state = AvailabilityState(now=lambda: clock["now"])
    cf = respx.post(CF_CHAT).mock(side_effect=[cf_error(429, 3036), chat_ok("cf is back")])
    respx.post(GROQ_CHAT).mock(return_value=chat_ok("groq"))

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    gw = Gateway(settings, state=state, sleep=fake_sleep)
    try:
        assert (await gw.chat("reasoning", REQ)).provider == "groq"
        clock["now"] = datetime(2026, 11, 14, 0, 0, 1, tzinfo=UTC)
        assert (await gw.chat("reasoning", REQ)).provider == "cloudflare"
        assert cf.call_count == 2
    finally:
        await gw.aclose()


@respx.mock
async def test_cf_3040_retries_once_with_backoff_then_succeeds(gateway, sleeps):
    cf = respx.post(CF_CHAT).mock(
        side_effect=[cf_error(429, 3040, "capacity"), chat_ok("second try")]
    )
    groq = respx.post(GROQ_CHAT).mock(return_value=chat_ok("groq"))

    result = await gateway.chat("reasoning", REQ)

    assert result.provider == "cloudflare" and result.output.text == "second try"
    assert cf.call_count == 2 and groq.call_count == 0
    assert sleeps == [1.0]


@respx.mock
async def test_cf_3040_retries_once_then_next_provider(gateway, sleeps):
    cf = respx.post(CF_CHAT).mock(return_value=cf_error(429, 3040, "capacity"))
    groq = respx.post(GROQ_CHAT).mock(return_value=chat_ok("groq"))

    result = await gateway.chat("reasoning", REQ)

    assert result.provider == "groq" and result.is_fallback is True
    assert cf.call_count == 2 and groq.call_count == 1
    assert sleeps == [1.0]


@respx.mock
async def test_cf_5035_drops_model_and_goes_next(gateway, state):
    cf = respx.post(CF_CHAT).mock(return_value=cf_error(400, 5035, "needs paid plan"))
    respx.post(GROQ_CHAT).mock(return_value=chat_ok("groq"))

    result = await gateway.chat("reasoning", REQ)  # 5035 must win over the 400 rule

    assert result.provider == "groq"
    assert cf.call_count == 1
    assert state.unavailable_reason("cloudflare", CF_120B) == "model_disabled_paid_plan_only"
    # Only that model is dropped, not the whole provider.
    assert state.unavailable_reason("cloudflare", "@cf/openai/gpt-oss-20b") is None


@respx.mock
async def test_groq_429_short_retry_after_is_honoured(gateway, sleeps):
    respx.post(CF_CHAT).mock(return_value=cf_error(429, 3036))
    groq = respx.post(GROQ_CHAT).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "rl"}}),
            chat_ok("after wait"),
        ]
    )

    result = await gateway.chat("reasoning", REQ)

    assert result.provider == "groq" and result.output.text == "after wait"
    assert groq.call_count == 2
    assert sleeps == [2.0]


@respx.mock
async def test_groq_429_long_retry_after_goes_next_and_cools_down(gateway, state, sleeps):
    respx.post(CF_CHAT).mock(return_value=cf_error(429, 3036))
    groq = respx.post(GROQ_CHAT).mock(
        return_value=httpx.Response(429, headers={"retry-after": "600"}, json={"error": {}})
    )

    with pytest.raises(AllProvidersFailed):  # reasoning chain ends in an openrouter placeholder
        await gateway.chat("reasoning", REQ)

    assert groq.call_count == 1  # no sleeping through 10 minutes, no retry
    assert sleeps == []
    assert state.unavailable_reason("groq", "openai/gpt-oss-120b") is not None


@respx.mock
@pytest.mark.parametrize("failure", [httpx.Response(500), httpx.Response(503, text="down")])
async def test_5xx_retries_once_then_next(gateway, sleeps, failure):
    cf = respx.post(CF_CHAT).mock(return_value=failure)
    groq = respx.post(GROQ_CHAT).mock(return_value=chat_ok("groq"))

    result = await gateway.chat("reasoning", REQ)

    assert result.provider == "groq"
    assert cf.call_count == 2 and groq.call_count == 1
    assert sleeps == [0.5]


@respx.mock
async def test_timeout_retries_once_then_next(gateway):
    cf = respx.post(CF_CHAT).mock(side_effect=httpx.ReadTimeout("slow"))
    respx.post(GROQ_CHAT).mock(return_value=chat_ok("groq"))

    result = await gateway.chat("reasoning", REQ)

    assert result.provider == "groq"
    assert cf.call_count == 2


@respx.mock
async def test_5xx_then_success_on_retry_stays_on_provider(gateway):
    cf = respx.post(CF_CHAT).mock(side_effect=[httpx.Response(502), chat_ok("ok")])

    result = await gateway.chat("reasoning", REQ)

    assert result.provider == "cloudflare" and cf.call_count == 2


@respx.mock
async def test_400_does_not_fall_back_and_raises_clear_error(gateway):
    cf = respx.post(CF_CHAT).mock(return_value=cf_error(400, 1234, "bad param: foo"))
    groq = respx.post(GROQ_CHAT).mock(return_value=chat_ok("never"))

    with pytest.raises(BadRequestError) as exc:
        await gateway.chat("reasoning", REQ)

    assert "bad param: foo" in str(exc.value)
    assert exc.value.provider == "cloudflare"
    assert cf.call_count == 1 and groq.call_count == 0


@respx.mock
async def test_groq_400_does_not_fall_back(gateway):
    respx.post(CF_CHAT).mock(return_value=cf_error(429, 3036))
    respx.post(GROQ_CHAT).mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad", "code": "x"}})
    )
    with pytest.raises(BadRequestError):
        await gateway.chat("reasoning", REQ)


@respx.mock
async def test_image_chain_falls_to_second_cloudflare_model_on_5035(gateway):
    png = base64.b64encode(b"fakepng").decode()
    respx.post(CF_SCHNELL).mock(return_value=cf_error(400, 5035))
    klein = respx.post(CF_KLEIN_4B).mock(
        return_value=httpx.Response(200, json={"result": {"image": png}, "success": True})
    )

    result = await gateway.image("image_create", ImageRequest(prompt="a cat"))

    assert result.model == "@cf/black-forest-labs/flux-2-klein-4b"
    assert result.output.images == [b"fakepng"]
    assert result.is_fallback is True
    assert klein.call_count == 1


@respx.mock
async def test_all_providers_failed_reports_skipped_placeholder(gateway):
    respx.post(CF_SCHNELL).mock(return_value=cf_error(429, 3036))

    with pytest.raises(AllProvidersFailed) as exc:
        await gateway.image("image_create", ImageRequest(prompt="a cat"))

    outcomes = {(a.provider, a.outcome) for a in exc.value.attempts}
    assert ("openrouter", "skipped") in outcomes


async def test_wrong_kind_and_unknown_task_are_rejected(gateway):
    with pytest.raises(ValueError):
        await gateway.chat("image_create", REQ)
    with pytest.raises(ValueError):
        await gateway.chat("nope", REQ)
