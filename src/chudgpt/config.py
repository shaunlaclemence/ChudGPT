"""Provider registry.

Every provider here exposes an OpenAI-compatible chat-completions endpoint, so the
whole library talks one wire format through the ``openai`` client with different
base URLs. Free-tier limits drift constantly; ``known_rpd`` is a proactive-skip
hint only — the 429 response is always the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, List
import json

with open("config.json", "r") as file:
    config_data = json.load(file)


ResetPolicy = Literal["midnight_pt", "rolling"]

TIERS = ("best", "fast")

@dataclass
class KeyConfig:
    account: str
    name: str
    project_name: str
    project_number: int
    api_key: str


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    keys: List[KeyConfig]
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
        keys=json.loads(config_data)["gemini"],
        models={"best": "gemini-2.5-pro", "fast": "gemini-2.5-flash"},
        priority=10,
        known_rpd=1500,
        reset="midnight_pt",
    ),
)


def provider_by_name(
    name: str, providers: tuple[ProviderConfig, ...] = PROVIDERS
) -> ProviderConfig:
    for p in providers:
        if p.name == name:
            return p
    raise KeyError(f"unknown provider {name!r}")
