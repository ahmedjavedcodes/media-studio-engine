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
    # Build mode calls Jev directly through TypeSafe (open decision 1 settled). The Cloudflare
    # route `typesafe/jev` returned "No route" on /ai/run, so it is not used.
    jev_provider: Literal["cloudflare", "typesafe"] = "typesafe"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    typesafe_base_url: str = "https://api.typesafe.ai"
    cors_origins: str = "http://localhost:3000"

    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "app-dev"

    # Guardrail timeouts in seconds. Unset (None) = the PROVIDER_MODE default in
    # guardrails/policy.py. Retry delay 0 = no retry.
    jev_timeout_s: float | None = None
    jev_retry_delay_s: float | None = None
    vision_timeout_s: float | None = None
    vision_retry_delay_s: float | None = None
    deadline_text_s: float | None = None
    deadline_images_s: float | None = None
    guard_warmup: bool = True  # warm provider connections at API startup

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
