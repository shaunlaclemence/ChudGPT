"""Exceptions raised by chudgpt."""

from __future__ import annotations

from datetime import datetime


class ChudGPTError(Exception):
    """Base class for all chudgpt errors."""


class ConfigError(ChudGPTError):
    """Raised when configuration or key discovery is invalid (e.g. no keys found)."""


class ProviderError(ChudGPTError):
    """A single provider call failed for a non-quota reason."""

    def __init__(self, provider: str, message: str, cause: Exception | None = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.cause = cause


class AllProvidersExhausted(ChudGPTError):
    """Every configured key is rate-limited, exhausted, or failing.

    ``statuses`` maps key id -> human-readable reason.
    ``earliest_reset`` is when the first key is expected to become usable again.
    """

    def __init__(self, statuses: dict[str, str], earliest_reset: datetime | None):
        self.statuses = statuses
        self.earliest_reset = earliest_reset
        detail = (
            "; ".join(f"{k}: {v}" for k, v in statuses.items()) or "no keys configured"
        )
        when = (
            f" Earliest reset: {earliest_reset.isoformat()}." if earliest_reset else ""
        )
        super().__init__(f"All providers exhausted ({detail}).{when}")
