"""Persist quota state between runs so daily counters survive restarts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .quota import QuotaTracker

DEFAULT_STATE_FILE = Path.home() / ".chudgpt" / "state.json"


def state_path(override: Path | str | None = None) -> Path:
    if override is not None:
        return Path(override)
    env = os.environ.get("CHUDGPT_STATE")
    return Path(env) if env else DEFAULT_STATE_FILE


def load_state(path: Path | str | None = None) -> QuotaTracker:
    p = state_path(path)
    if not p.exists():
        return QuotaTracker()
    try:
        return QuotaTracker.from_dict(json.loads(p.read_text()))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        # Corrupt state is not worth crashing over; start fresh.
        return QuotaTracker()


def save_state(tracker: QuotaTracker, path: Path | str | None = None) -> None:
    p = state_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(tracker.to_dict(), indent=2))
    tmp.replace(p)
