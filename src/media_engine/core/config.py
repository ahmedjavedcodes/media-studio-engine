"""Application settings. Reads `.env`; never log or print the values of secret fields."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_env: str = "local"
    log_level: str = "INFO"

    # Free providers
    cloudflare_account_id: str | None = None
    # CLOUDFLARE_API_KEY is still accepted as an alias of CLOUDFLARE_API_TOKEN.
    cloudflare_api_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("cloudflare_api_token", "cloudflare_api_key")
    )
    groq_api_key: SecretStr | None = None

    # Paid providers
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    fal_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    typesafe_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    siliconflow_api_key: SecretStr | None = None

    # Memory
    pinecone_api_key: SecretStr | None = None
    pinecone_index: str = "media-engine-dev"

    # Observability
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "media-engine-dev"

    # Local dev
    local_output_dir: str = "./outputs"
    checkpoint_db: str = "./.checkpoints.sqlite"

    @field_validator("*", mode="before")
    @classmethod
    def _blank_is_none(cls, value: Any) -> Any:
        # `KEY=` in .env arrives as "" — treat it as "not set".
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
