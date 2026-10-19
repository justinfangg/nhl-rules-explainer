"""Settings loaded from environment variables (and an optional .env file)."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://nhl:nhl@localhost:5433/nhl_rules"
    nhl_api_url: str = "http://localhost:8000"
    # The rulebook season the API answers from.
    rulebook_season: str = "2026-27"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Claude model for the Understand and Generate stages.
    anthropic_api_key: SecretStr | None = None
    # Only needed when the API key isn't scoped to a workspace.
    anthropic_workspace_id: str | None = None
    llm_model: str = "claude-opus-5-5"
    understand_effort: str = "low"
    generate_effort: str = "medium"
    # Faithfulness judge (eval only).
    judge_model: str = "claude-opus-5-5"
    judge_effort: str = "medium"


@lru_cache
def get_settings() -> Settings:
    return Settings()
