from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from media_engine.core.config import Settings
from media_engine.gateway.fallback import AvailabilityState
from media_engine.gateway.image_registry import ImageRegistry, use_image_registry
from media_engine.gateway.openrouter_models import ImageModelInfo, ModelCache
from media_engine.gateway.router import Gateway, load_chains, parse_ref

ACCOUNT = "acct123"
CF_CHAT = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/v1/chat/completions"
CF_RUN = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run"
GROQ_CHAT = "https://api.groq.com/openai/v1/chat/completions"

NOW = datetime(2026, 11, 13, 10, 30, tzinfo=UTC)

# Fallback-MECHANICS tests (Cloudflare 3036/3040/5035, retries, neuron metering) need a chain
# whose first entry is Cloudflare. Production agent chains are Groq first, so these tests use
# SYNTHETIC chains with their own names: they can never pass for a production chain. Every
# production chain keeps its real order here; production order and the production fallback
# path are asserted in tests/unit/harness/test_config.py and tests/unit/gateway/
# test_production_chains.py.
CF_FIRST_LLM = "cf_first_llm"  # synthetic: Cloudflare 120b -> Groq 120b -> paid
CF_FIRST_FAST_LLM = "cf_first_fast_llm"  # synthetic: Cloudflare 20b -> Groq 20b -> paid
CF_FIRST_CHAINS = {
    **load_chains(),
    CF_FIRST_LLM: [
        parse_ref("cloudflare/@cf/openai/gpt-oss-120b"),
        parse_ref("groq/openai/gpt-oss-120b"),
        parse_ref("openrouter/openai/gpt-oss-120b"),
    ],
    CF_FIRST_FAST_LLM: [
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


# -- the process-wide image registry ---------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_image_registry():
    """Every test starts from the default registry and leaves it as it found it: the registry
    is process-wide (one source of truth), so a test that swaps it must not leak the swap."""
    previous = use_image_registry(None)
    yield
    use_image_registry(previous)


def entry_model(slug: str, **overrides: Any) -> dict[str, Any]:
    """A models.yaml entry with harmless defaults (entry tier, one image per call)."""
    base: dict[str, Any] = {
        "slug": slug,
        "name": slug,
        "tier": "entry",
        "billing_unit": "image",
        "price_measured_usd": 0.01,
        "supports_seed": False,
        "n_per_call": 1,
        "max_input_images": 0,
        "prompt_style": "sentences",
        "prompt_max_words": 50,
    }
    return base | overrides


def install_registry(
    tmp_path: Path,
    infos: list[ImageModelInfo],
    models: list[dict[str, Any]],
    setting_policy: dict[str, Any] | None = None,
) -> ImageRegistry:
    """Build a registry from a test models.yaml + test catalogue and make it THE registry."""
    cache = ModelCache(tmp_path / "catalogue.json")
    cache.save(infos)
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"setting_policy": setting_policy or {}, "models": models}))
    registry = ImageRegistry.load(cache, path=path)
    use_image_registry(registry)
    return registry
