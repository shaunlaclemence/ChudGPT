"""Discoverability enums for tier/model/temperature — what you can pass to
ChudClient.ask()/stream()/start_conversation() and Conversation.send()/ask().

GENERATED FILE — do not hand-edit Tier or Model. Regenerate with:
    uv run python scripts/generate_params.py
after changing config.TIERS or src/chudgpt/config.json (the model catalog).
"""

from __future__ import annotations

from enum import Enum


class Tier(str, Enum):
    """Quality/speed tradeoff — resolved to a concrete model per provider
    via ``ProviderConfig.model_for()``. Works anywhere a plain
    ``tier="best"`` string does."""

    BEST = "best"
    FAST = "fast"


class Model(str, Enum):
    """Every real, currently-usable model id per provider (see config.json).
    A ``model=`` override is sent as-is regardless of which provider ends up
    serving the request during rotation — only pass one of these if you also
    constrain ``providers=`` to the single provider that actually serves it,
    otherwise a request that rotates to a different provider will fail with
    an unknown-model error."""

    GEMINI_3_6_FLASH = "gemini-3.6-flash"
    GEMINI_3_5_FLASH = "gemini-3.5-flash"
    GEMINI_3_5_FLASH_LITE = "gemini-3.5-flash-lite"
    GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GROQ_LLAMA_3_3_70B_VERSATILE = "llama-3.3-70b-versatile"
    GROQ_LLAMA_3_1_8B_INSTANT = "llama-3.1-8b-instant"
    MISTRAL_LARGE_LATEST = "mistral-large-latest"
    MISTRAL_SMALL_LATEST = "mistral-small-latest"
    XAI_GROK_4 = "grok-4"
    XAI_GROK_4_FAST = "grok-4-fast"
    OPENROUTER_META_LLAMA_LLAMA_3_3_70B_INSTRUCT_FREE = "meta-llama/llama-3.3-70b-instruct:free"
    OPENROUTER_META_LLAMA_LLAMA_3_1_8B_INSTRUCT_FREE = "meta-llama/llama-3.1-8b-instruct:free"


class Temperature(float, Enum):
    """Named presets for the ``temperature=`` request kwarg (an OpenAI-compatible
    sampling parameter, typically 0.0-2.0). NOT exhaustive — pass any float for a
    value in between. Lower = more deterministic/focused; higher = more random.
    """

    PRECISE = 0.0
    BALANCED = 0.7
    CREATIVE = 1.0
    WILD = 1.4
