"""Shared Anthropic client and usage accounting."""

from dataclasses import dataclass
from functools import lru_cache

import anthropic

from nhl_rules.config import get_settings

# USD per million tokens, for cost estimates in eval reports.
PRICES = {"claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20}}


@lru_cache
def client() -> anthropic.Anthropic:
    settings = get_settings()
    kwargs: dict = {}
    if settings.anthropic_api_key:  # else the SDK's own resolution (env var, `ant auth login`)
        kwargs["api_key"] = settings.anthropic_api_key.get_secret_value()
    if settings.anthropic_workspace_id:
        kwargs["default_headers"] = {"anthropic-workspace-id": settings.anthropic_workspace_id}
    return anthropic.Anthropic(**kwargs)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @classmethod
    def of(cls, response) -> "Usage":
        u = response.usage
        return cls(
            u.input_tokens,
            u.output_tokens,
            u.cache_read_input_tokens or 0,
            u.cache_creation_input_tokens or 0,
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    def cost(self, model: str) -> float | None:
        """Estimated USD; cache writes priced at 1.25x input (5-minute cache)."""
        p = PRICES.get(model)
        if p is None:
            return None
        return (
            self.input_tokens * p["input"]
            + self.cache_write_tokens * p["input"] * 1.25
            + self.cache_read_tokens * p["cache_read"]
            + self.output_tokens * p["output"]
        ) / 1e6
