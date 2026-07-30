"""Exceptions raised by chudgpt.

Everything here derives from ``ChudGPTError``, so a caller that just wants
"did chudgpt fail?" can catch that one class and be exhaustive.
"""

from __future__ import annotations

from datetime import datetime


class ChudGPTError(Exception):
    """Base class for all chudgpt errors."""


class ConfigError(ChudGPTError):
    """Configuration or key discovery is invalid (e.g. no keys found)."""


class SecretsFileError(ConfigError):
    """A secrets file was named but could not be read or parsed."""

    def __init__(self, path: str, reason: str):
        super().__init__(f"could not load secrets from {path}: {reason}")
        self.path = path
        self.reason = reason


class UnknownProviderError(ConfigError):
    """A provider name was requested that isn't in the registry."""

    def __init__(self, name: str, known: list[str]):
        super().__init__(
            f"unknown provider {name!r}; known providers: {', '.join(sorted(known))}"
        )
        self.name = name
        self.known = known


class InvalidRequestError(ChudGPTError):
    """The caller passed arguments that can't form a valid request."""


class InvalidTierError(InvalidRequestError):
    """A tier was requested that no provider defines."""

    def __init__(self, tier: str, known: list[str]):
        super().__init__(
            f"unknown tier {tier!r}; available tiers: {', '.join(sorted(known))}"
        )
        self.tier = tier
        self.known = known


class ProviderError(ChudGPTError):
    """A single provider call failed for a non-quota reason."""

    def __init__(self, provider: str, message: str, cause: Exception | None = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.cause = cause


class StreamInterrupted(ProviderError):
    """A stream failed partway through, after content had already been yielded.

    Not retried automatically: the caller has already seen part of the reply, so
    transparently rotating to another provider would duplicate output.
    """

    def __init__(self, provider: str, cause: Exception | None = None):
        super().__init__(provider, "stream interrupted mid-response", cause)


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
