"""Key discovery. Keys live in env vars or ~/.chudgpt/keys.json — never in code.

Multiple comma-separated keys per env var are accepted for legitimate cases
(e.g. a work and a personal account), but note that rotating several free-tier
accounts on the *same* provider to multiply quota violates most providers'
terms of service. The intended model is one key per provider.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .config import PROVIDERS, ProviderConfig
from .errors import ConfigError

DEFAULT_KEYS_FILE = Path.home() / ".chudgpt" / "keys.json"


def key_id(provider: str, key: str) -> str:
    """Stable identifier for a key that never exposes the key itself."""
    digest = hashlib.sha256(key.encode()).hexdigest()[:8]
    return f"{provider}:{digest}"


def load_keys(
    providers: tuple[ProviderConfig, ...] = PROVIDERS,
    env: dict[str, str] | None = None,
    keys_file: Path | None = None,
) -> dict[str, list[str]]:
    """Return {provider_name: [key, ...]} from env vars, then the keys file.

    Env vars win over the file for a given provider. Raises ConfigError if no
    keys are found at all.
    """
    env = os.environ if env is None else env
    keys_file = DEFAULT_KEYS_FILE if keys_file is None else keys_file

    file_keys: dict[str, list[str]] = {}
    if keys_file.exists():
        try:
            raw = json.loads(keys_file.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ConfigError(f"could not read keys file {keys_file}: {e}") from e
        for name, value in raw.items():
            if isinstance(value, str):
                value = [value]
            file_keys[name] = [k.strip() for k in value if k and k.strip()]

    keys: dict[str, list[str]] = {}
    for cfg in providers:
        raw_env = env.get(cfg.env_var, "")
        from_env = [k.strip() for k in raw_env.split(",") if k.strip()]
        chosen = from_env or file_keys.get(cfg.name, [])
        if chosen:
            keys[cfg.name] = chosen

    if not keys:
        wanted = ", ".join(cfg.env_var for cfg in providers)
        raise ConfigError(
            f"no API keys found. Set one or more of: {wanted}, "
            f'or create {keys_file} with {{"provider": ["key"]}} entries.'
        )
    return keys
