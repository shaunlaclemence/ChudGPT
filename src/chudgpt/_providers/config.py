from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str


PROVIDER_CONFIGS: tuple[ProviderConfig, ...] = (
    ProviderConfig(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    # ProviderConfig(
    #     name="groq",
    #     base_url="https://api.groq.com/openai/v1",
    # ),
    # ProviderConfig(
    #     name="mistral",
    #     base_url="https://api.mistral.ai/v1",
    # ),
    # ProviderConfig(
    #     name="xai",
    #     base_url="https://api.x.ai/v1",
    # ),
    # ProviderConfig(
    #     name="openrouter",
    #     base_url="https://openrouter.ai/api/v1",
    # ),
)
