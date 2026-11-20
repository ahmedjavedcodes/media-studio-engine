from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from media_engine.core.config import Settings
from media_engine.gateway.fallback import AvailabilityState
from media_engine.gateway.router import Gateway, load_chains, parse_ref

ACCOUNT = "acct123"
CF_CHAT = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/v1/chat/completions"
CF_RUN = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run"
GROQ_CHAT = "https://api.groq.com/openai/v1/chat/completions"

NOW = datetime(2026, 11, 13, 10, 30, tzinfo=UTC)

# Fallback-mechanics tests (Cloudflare 3036/3040/5035, retries, neuron metering) need a chain
# whose FIRST entry is Cloudflare. Production chains.yaml puts Groq first for agents, so these
# tests pin their own LLM chains instead of following production order. Production chain ORDER
# is asserted in tests/unit/harness/test_config.py.
CF_FIRST_CHAINS = {
    **load_chains(),
    "reasoning": [
        parse_ref("cloudflare/@cf/openai/gpt-oss-120b"),
        parse_ref("groq/openai/gpt-oss-120b"),
        parse_ref("openrouter/openai/gpt-oss-120b"),
    ],
    "fast_llm": [
        parse_ref("cloudflare/@cf/openai/gpt-oss-20b"),
        parse_ref("groq/openai/gpt-oss-20b"),
        parse_ref("openrouter/openai/gpt-oss-120b"),
    ],
}


def chat_ok(text: str = "hi") -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 5}},
    )


def cf_error(status: int, code: int, message: str = "err") -> httpx.Response:
    return httpx.Response(
        status, json={"success": False, "errors": [{"code": code, "message": message}]}
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        cloudflare_account_id=ACCOUNT,
        cloudflare_api_token="cf-test-token",
        groq_api_key="groq-test-key",
    )


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def state() -> AvailabilityState:
    return AvailabilityState(now=lambda: NOW)


@pytest.fixture
async def gateway(settings, sleeps, state):
    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    gw = Gateway(settings, chains=CF_FIRST_CHAINS, state=state, sleep=fake_sleep)
    yield gw
    await gw.aclose()
