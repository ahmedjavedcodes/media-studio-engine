"""Build the guardrail's providers from settings. PROVIDER_MODE picks the chain (CLAUDE.md).

One shared httpx.AsyncClient per provider with keep-alive, so calls reuse a warm TLS
connection. A cold connection costs about 870 ms, which is more than the Jev timeout.
"""

from functools import lru_cache

import httpx

from app.contracts.ports import JevPort, VisionPort
from app.core.config import Settings, get_settings, prompts_dir
from app.gateway.providers.cloudflare import (
    CloudflareClient,
    CloudflareJev,
    CloudflareVisionStandIn,
)
from app.gateway.providers.openrouter import OpenRouterLlamaGuard
from app.gateway.providers.typesafe import TypeSafeJev
from app.guardrails import policy
from app.guardrails.agent import GuardDeps, Timing
from app.local.fake_strikes import FakeStrikeLedger

_LIMITS = httpx.Limits(max_keepalive_connections=10, keepalive_expiry=60)
_clients: list[httpx.AsyncClient] = []


def _client() -> httpx.AsyncClient:
    client = httpx.AsyncClient(limits=_LIMITS)
    _clients.append(client)
    return client


async def close_guard_clients() -> None:
    for client in _clients:
        await client.aclose()
    _clients.clear()


def timing_for(s: Settings) -> Timing:
    """Mode defaults from policy.py, with any env override applied."""
    build = s.provider_mode == "build"
    vision_timeout = policy.BUILD_VISION_TIMEOUT_S if build else policy.TEST_VISION_TIMEOUT_S
    vision_retry = policy.BUILD_VISION_RETRY_DELAY_S if build else policy.TEST_VISION_RETRY_DELAY_S
    deadline_text = policy.BUILD_DEADLINE_TEXT_S if build else policy.TEST_DEADLINE_TEXT_S
    deadline_images = policy.BUILD_DEADLINE_IMAGES_S if build else policy.TEST_DEADLINE_IMAGES_S

    def pick[T](override: T | None, default: T) -> T:
        return default if override is None else override

    jev_retry = pick(s.jev_retry_delay_s, policy.JEV_RETRY_DELAY_S)
    vision_retry = pick(s.vision_retry_delay_s, vision_retry)
    return Timing(
        jev_timeout_s=pick(s.jev_timeout_s, policy.JEV_TIMEOUT_S),
        jev_retry_delay_s=jev_retry or None,  # 0 = no retry
        vision_timeout_s=pick(s.vision_timeout_s, vision_timeout),
        vision_retry_delay_s=vision_retry or None,
        deadline_text_s=pick(s.deadline_text_s, deadline_text),
        deadline_images_s=pick(s.deadline_images_s, deadline_images),
    )


def build_deps(s: Settings) -> GuardDeps:
    jev: JevPort
    vision: VisionPort
    if s.provider_mode == "test":  # paid providers, only when Ahmed asks
        jev = TypeSafeJev(_client(), api_key=s.typesafe_api_key, base_url=s.typesafe_base_url)
        vision = OpenRouterLlamaGuard(
            _client(), api_key=s.openrouter_api_key, base_url=s.openrouter_base_url
        )
    else:  # build mode: Jev through TypeSafe, vision on Cloudflare
        cf = CloudflareClient(
            _client(), account_id=s.cloudflare_account_id, api_token=s.cloudflare_api_token
        )
        if s.jev_provider == "typesafe":
            jev = TypeSafeJev(_client(), api_key=s.typesafe_api_key, base_url=s.typesafe_base_url)
        else:
            jev = CloudflareJev(cf)
        prompt = (prompts_dir() / "guardrail" / "vision_standin.md").read_text("utf-8")
        vision = CloudflareVisionStandIn(cf, system_prompt=prompt)
    return GuardDeps(
        jev=jev, vision=vision, ledger=FakeStrikeLedger(), settings=s, timing=timing_for(s)
    )


@lru_cache
def get_guard_deps() -> GuardDeps:
    return build_deps(get_settings())
