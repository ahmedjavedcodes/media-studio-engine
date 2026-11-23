"""Settings loaded from .env with pydantic-settings."""

from functools import lru_cache
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

    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "app-dev"

    local_output_dir: str = "./outputs"
    checkpoint_db: str = "./.checkpoints.sqlite"


@lru_cache
def get_settings() -> Settings:
    return Settings()
