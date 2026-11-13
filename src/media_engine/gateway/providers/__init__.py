from __future__ import annotations

import httpx

from media_engine.core.config import Settings
from media_engine.gateway.providers.base import Provider
from media_engine.gateway.providers.cloudflare import CloudflareProvider
from media_engine.gateway.providers.groq import GroqProvider


def build_providers(settings: Settings, client: httpx.AsyncClient) -> dict[str, Provider]:
    """Providers implemented so far. OpenRouter, Anthropic, fal etc. come later."""
    return {
        "cloudflare": CloudflareProvider(client, settings),
        "groq": GroqProvider(client, settings),
    }


__all__ = ["Provider", "build_providers"]
