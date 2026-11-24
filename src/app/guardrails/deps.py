"""Build the guardrail's providers from settings. PROVIDER_MODE picks the chain (CLAUDE.md)."""

from functools import lru_cache

import httpx

from app.contracts.ports import JevPort, VisionPort
from app.core.config import get_settings, prompts_dir
from app.gateway.providers.cloudflare import (
    CloudflareClient,
    CloudflareJev,
    CloudflareVisionStandIn,
)
from app.gateway.providers.openrouter import OpenRouterLlamaGuard
from app.gateway.providers.typesafe import TypeSafeJev
from app.guardrails.agent import GuardDeps, Timing
from app.local.fake_strikes import FakeStrikeLedger


@lru_cache
def get_guard_deps() -> GuardDeps:
    s = get_settings()
    client = httpx.AsyncClient()
    jev: JevPort
    vision: VisionPort
    use_typesafe_jev = s.provider_mode == "test" or s.jev_provider == "typesafe"
    if use_typesafe_jev:
        jev = TypeSafeJev(client, api_key=s.typesafe_api_key, base_url=s.typesafe_base_url)
    if s.provider_mode == "test":  # paid providers, only when Ahmed asks
        vision = OpenRouterLlamaGuard(
            client, api_key=s.openrouter_api_key, base_url=s.openrouter_base_url
        )
    else:  # build mode: Cloudflare only
        cf = CloudflareClient(
            client, account_id=s.cloudflare_account_id, api_token=s.cloudflare_api_token
        )
        prompt = (prompts_dir() / "guardrail" / "vision_standin.md").read_text("utf-8")
        if not use_typesafe_jev:
            jev = CloudflareJev(cf)
        vision = CloudflareVisionStandIn(cf, system_prompt=prompt)
    timing = Timing(jev_timeout_s=s.jev_timeout_s, vision_timeout_s=s.vision_timeout_s)
    return GuardDeps(jev=jev, vision=vision, ledger=FakeStrikeLedger(), settings=s, timing=timing)
