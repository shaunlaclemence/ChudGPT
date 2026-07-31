"""Provider registry.

Every provider here exposes an OpenAI-compatible chat-completions endpoint, so the
whole library talks one wire format through the ``openai`` client with different
base URLs. Free-tier limits drift constantly; ``known_rpd`` is a proactive-skip
hint only — the 429 response is always the source of truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Literal

from .errors import InvalidTierError, UnknownProviderError

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
            raise InvalidTierError(tier, list(self.models)) from None


PROVIDERS: tuple[ProviderConfig, ...] = (
    ProviderConfig(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        env_var="GEMINI_API_KEY",
        # 2.5-era models are being retired ("no longer available to new users") and
        # 2.5-pro's free RPD is tiny, so the tier defaults point at the 3.x line.
        models={"best": "gemini-3.6-flash", "fast": "gemini-3.5-flash-lite"},
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

# Providers whose OpenAI-compatible endpoint accepts audio input (as
# ``input_audio`` content parts). ChudClient.transcribe() restricts a call to
# these so a gemini-only model isn't sent to a text-only provider. Gemini is the
# one that supports it today; extend as others do.
AUDIO_PROVIDERS: tuple[str, ...] = ("gemini",)


def provider_by_name(
    name: str, providers: tuple[ProviderConfig, ...] = PROVIDERS
) -> ProviderConfig:
    for p in providers:
        if p.name == name:
            return p
    raise UnknownProviderError(name, [p.name for p in providers])


def load_model_catalog() -> dict[str, list[str]]:
    """Every known-usable model id per provider (not just the tier defaults above).

    Read from the packaged ``config.json`` (shipped alongside this module, so it's
    available whether chudgpt is run from source or installed via pip/uv) — a
    non-secret catalog of real, currently-live model ids, kept up to date by hand
    against each provider's docs. This is NOT the same file as ``secrets.json``,
    which holds your actual API keys and is never committed.
    """
    raw = resources.files(__package__).joinpath("config.json").read_text()
    return json.loads(raw)


MODEL_CATALOG: dict[str, list[str]] = load_model_catalog()
