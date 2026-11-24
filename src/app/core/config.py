"""Settings loaded from .env with pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"
    provider_mode: Literal["build", "test"] = "build"
    spend_cap_usd: float = 2.0

    groq_api_key: str = ""
    cloudflare_account_id: str = ""
    cloudflare_api_token: str = ""
    openrouter_api_key: str = ""
    fal_key: str = ""
    anthropic_api_key: str = ""
    typesafe_api_key: str = ""
    # Which Jev to use in build mode. Cloudflare has no Jev model, so "typesafe" is the only
    # one that works today. It is opt-in because TypeSafe is not a Cloudflare (free) provider.
    jev_provider: Literal["cloudflare", "typesafe"] = "cloudflare"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    typesafe_base_url: str = "https://api.typesafe.ai"
    cors_origins: str = "http://localhost:3000"

    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "app-dev"

    # Guardrail timeouts in seconds. Defaults are the spec's section 9 values.
    jev_timeout_s: float = 0.5
    vision_timeout_s: float = 1.5

    # Guardrail strikes and cooldowns (env: INSTANT_STRIKE_LIMIT, ...)
    instant_strike_limit: int = 5  # strike N of this limit bans the account
    strike_window_days: int = 30
    standard_cooldown_sec: int = 60  # first standard reject
    standard_repeat_cooldown_sec: int = 120  # another standard reject within 10 minutes

    local_output_dir: str = "./outputs"
    checkpoint_db: str = "./.checkpoints.sqlite"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def prompts_dir() -> Path:
    """Folder with system prompts. Repo root `prompts/`."""
    return Path(__file__).resolve().parents[3] / "prompts"
