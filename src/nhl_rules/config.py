"""Settings loaded from environment variables (and an optional .env file)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://nhl:nhl@localhost:5433/nhl_rules"
    nhl_api_url: str = "http://localhost:8000"
    # The rulebook season the API answers from.
    rulebook_season: str = "2026-27"
    embedding_model: str = "BAAI/bge-small-en-v1.5"


@lru_cache
def get_settings() -> Settings:
    return Settings()
