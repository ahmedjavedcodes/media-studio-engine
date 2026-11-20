"""M2: LLMGateway picks the chain by agent + plan and reports spend; the real Gateway behind it
never reaches a paid provider unless ALLOW_PAID is on."""

from __future__ import annotations

import httpx
import pytest
import respx

from media_engine.contracts import AuthContext
from media_engine.core.config import Settings
from media_engine.gateway.errors import AllProvidersFailed
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver, token_counts
from media_engine.gateway.router import Gateway
from media_engine.services.create.loader import load_create_config
from tests.unit.harness.fakes import ScriptedChat

ACCOUNT = "acct123"
CF_CHAT = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/v1/chat/completions"
GROQ_CHAT = "https://api.groq.com/openai/v1/chat/completions"


def auth(plan: str) -> AuthContext:
    return AuthContext(plan=plan, execution_mode="auto", model_demand="balanced")  # type: ignore[arg-type]


async def test_chain_follows_agents_yaml_per_plan() -> None:
    cfg = load_create_config()
    backend = ScriptedChat(["{}", "{}", "{}"])
    llm = LLMGateway(backend, chain_resolver(cfg.agents))

    await llm.complete("router", auth("free"), [{"role": "user", "content": "x"}])
    await llm.complete("prompt_enhancer", auth("max"), [{"role": "user", "content": "x"}])
    await llm.complete("critic", auth("pro"), [{"role": "user", "content": "x"}])

    assert [task for task, _ in backend.requests] == ["fast_llm", "reasoning", "vision"]


async def test_gateway_keeps_no_per_run_state() -> None:
    # Shared across concurrent runs: accounting must travel in the result, not on the gateway.
    llm = LLMGateway(ScriptedChat(["{}"]), chain_resolver(load_create_config().agents))
    before = dict(vars(llm))
    out = await llm.complete("router", auth("free"), [{"role": "user", "content": "x"}])
    assert vars(llm) == before
    assert (out.prompt_tokens, out.completion_tokens, out.tokens) == (60, 40, 100)


async def test_unknown_agent_is_a_clear_error() -> None:
    llm = LLMGateway(ScriptedChat([]), chain_resolver({}))
    with pytest.raises(KeyError, match="nope"):
        await llm.complete("nope", auth("free"), [])


def test_token_counts_split_and_fall_back() -> None:
    assert token_counts({"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42}) == (
        30,
        12,
        42,
    )
    assert token_counts({"prompt_tokens": 10, "completion_tokens": 5}) == (10, 5, 15)
    assert token_counts({}) == (0, 0, 0)


def _settings(**extra: object) -> Settings:
    return Settings(
        _env_file=None,
        cloudflare_account_id=ACCOUNT,
        cloudflare_api_token="cf-test-token",
        groq_api_key="groq-test-key",
        openrouter_api_key="or-test-key",
        **extra,
    )


@respx.mock
async def test_through_real_gateway_reports_tokens_and_neurons() -> None:
    respx.post(CF_CHAT).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"route":"simple"}'}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 12, "neurons": 4.0},
            },
        )
    )
    # Production chains: the router's Groq model is rate-limited, so Cloudflare (which reports
    # neurons) serves it as the fallback.
    respx.post(GROQ_CHAT).mock(
        return_value=httpx.Response(429, headers={"retry-after": "120"}, json={"error": {}})
    )
    gw = Gateway(_settings())
    try:
        llm = LLMGateway(gw, chain_resolver(load_create_config().agents))
        out = await llm.complete("router", auth("free"), [{"role": "user", "content": "x"}])
    finally:
        await gw.aclose()
    assert out.provider == "cloudflare" and out.is_fallback
    assert (out.prompt_tokens, out.completion_tokens, out.tokens) == (30, 12, 42)
    assert out.neurons == 4.0
    assert out.cost_usd == 0.0


@respx.mock
async def test_paid_provider_is_skipped_by_default_even_with_a_key() -> None:
    respx.post(CF_CHAT).mock(return_value=httpx.Response(503))
    respx.post(GROQ_CHAT).mock(return_value=httpx.Response(503))
    paid = respx.post("https://openrouter.ai/api/v1/chat/completions")

    async def no_sleep(_: float) -> None:
        return None

    gw = Gateway(_settings(), sleep=no_sleep)
    try:
        llm = LLMGateway(gw, chain_resolver(load_create_config().agents))
        with pytest.raises(AllProvidersFailed) as info:
            await llm.complete("router", auth("free"), [{"role": "user", "content": "x"}])
    finally:
        await gw.aclose()
    assert not paid.called
    assert any(a.reason == "paid_disabled" for a in info.value.attempts)
