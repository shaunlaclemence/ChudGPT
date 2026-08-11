from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from chudgpt._schemas import ChudProvider

Secrets = dict[str, list[dict[str, Any]]]


def key_id(service: str, api_key: str) -> str:
    """Stable id for a key that never exposes it, e.g. ``gemini:1e3ff2cc``."""
    return f"{service}:{hashlib.sha256(api_key.encode()).hexdigest()[:8]}"


def load_secrets(path: Path | str) -> Secrets:
    return json.loads(Path(path).read_text())


def entries_for(secrets: Secrets, service: str) -> list[dict[str, Any]]:
    return [e for e in secrets.get(service, []) if e.get("api_key")]


def parse_providers(secrets: Secrets) -> dict[str, list[ChudProvider]]:
    return {
        service: parsed
        for service in secrets
        if (parsed := [_provider(e) for e in entries_for(secrets, service)])
    }


def _provider(entry: dict[str, Any]) -> ChudProvider:
    return ChudProvider(
        email=entry.get("account", ""),
        name=entry.get("name", ""),
        project_name=entry.get("project_name", ""),
        project_number=str(entry.get("project_number", "")),
        api_key=entry["api_key"],
    )
