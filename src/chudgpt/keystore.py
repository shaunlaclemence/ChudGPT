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
from dataclasses import dataclass
from pathlib import Path

from .config import PROVIDERS, ProviderConfig
from .errors import ConfigError, SecretsFileError

DEFAULT_KEYS_FILE = Path.home() / ".chudgpt" / "keys.json"


@dataclass
class KeyConfig:
    """One entry in a per-account key inventory file (see ``load_keys_from_secrets_json``)."""

    account: str
    name: str
    project_name: str
    project_number: int
    api_key: str


def load_keys_from_secrets_json(path: Path | str) -> dict[str, list[str]]:
    """Return {provider_name: [key, ...]} from a per-account key inventory file
    (conventionally named ``secrets.json`` — never commit it).

    The file maps provider name -> a list of ``KeyConfig``-shaped entries (each
    carrying bookkeeping like which account/project a key belongs to). Only the
    bare ``api_key`` strings are ever extracted and returned; the account/project
    metadata never leaves this function.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SecretsFileError(str(path), str(e)) from e

    keys: dict[str, list[str]] = {}
    try:
        for provider, entries in raw.items():
            parsed = [KeyConfig(**entry) for entry in entries]
            chosen = [kc.api_key for kc in parsed if kc.api_key]
            if chosen:
                keys[provider] = chosen
    except (AttributeError, TypeError) as e:
        raise SecretsFileError(
            str(path),
            "expected {provider: [{account, name, project_name, project_number, "
            f"api_key}}, ...]}} entries ({e})",
        ) from e

    if not keys:
        raise SecretsFileError(str(path), "no API keys found")
    return keys


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
