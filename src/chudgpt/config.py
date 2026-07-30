"""Provider registry.

Every provider here exposes an OpenAI-compatible chat-completions endpoint, so the
whole library talks one wire format through the ``openai`` client with different
base URLs. Free-tier limits drift constantly; ``known_rpd`` is a proactive-skip
hint only — the 429 response is always the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ResetPolicy = Literal["midnight_pt", "rolling"]

TIERS = ("best", "fast")


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    env_var: str
    models: dict[str, str]  # tier -> concrete model id
    priority: int = 100  # lower tries first
    known_rpd: int | None = None  # requests/day on the free tier, if known
    reset: ResetPolicy = "rolling"
    default_cooldown_s: int = (
        15 * 60
    )  # used on 429 when no Retry-After and reset is rolling
    extra_headers: dict[str, str] = field(default_factory=dict)

    def model_for(self, tier: str) -> str:
        try:
            return self.models[tier]
        except KeyError:
            raise KeyError(
                f"provider {self.name!r} has no model for tier {tier!r}; "
                f"available: {sorted(self.models)}"
            ) from None


PROVIDERS: tuple[ProviderConfig, ...] = (
    ProviderConfig(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        env_var="GEMINI_API_KEY",
        models={"best": "gemini-2.5-pro", "fast": "gemini-2.5-flash"},
        priority=10,
        known_rpd=1500,
        reset="midnight_pt",
    ),
    ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        env_var="GROQ_API_KEY",
        models={"best": "llama-3.3-70b-versatile", "fast": "llama-3.1-8b-instant"},
        priority=20,
        known_rpd=14400,
        reset="midnight_pt",
    ),
    ProviderConfig(
        name="mistral",
        base_url="https://api.mistral.ai/v1",
        env_var="MISTRAL_API_KEY",
        models={"best": "mistral-large-latest", "fast": "mistral-small-latest"},
        priority=30,
        reset="rolling",
    ),
    ProviderConfig(
        name="xai",
        base_url="https://api.x.ai/v1",
        env_var="XAI_API_KEY",
        models={"best": "grok-4", "fast": "grok-4-fast"},
        priority=40,
        reset="rolling",
    ),
    ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        env_var="OPENROUTER_API_KEY",
        models={
            "best": "meta-llama/llama-3.3-70b-instruct:free",
            "fast": "meta-llama/llama-3.1-8b-instruct:free",
        },
        priority=50,
        reset="rolling",
    ),
)


def provider_by_name(
    name: str, providers: tuple[ProviderConfig, ...] = PROVIDERS
) -> ProviderConfig:
    for p in providers:
        if p.name == name:
            return p
    raise KeyError(f"unknown provider {name!r}")
