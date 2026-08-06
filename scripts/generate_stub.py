#!/usr/bin/env python3
"""Regenerate src/chudgpt/providers/gemini.pyi from src/chudgpt/config.json.

GeminiModel is built from the catalog at runtime, so type checkers can't see its
members. This emits a stub that declares them, keeping config.json the single
source of truth while `GeminiModel.FLASH_LITE_3_5` still resolves statically.

Run after editing config.json:
    uv run python scripts/generate_stub.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "src" / "chudgpt" / "config.json"
OUTPUT = ROOT / "src" / "chudgpt" / "providers" / "gemini.pyi"

TEMPLATE = '''# GENERATED FILE - do not edit.
# Regenerate with: uv run python scripts/generate_stub.py
from dataclasses import dataclass
from enum import Enum, Flag
from typing import Any, Self

class Modality(Flag):
    TEXT = 1
    IMAGE = 2
    AUDIO = 4
    VIDEO = 8

@dataclass(frozen=True)
class ModelSpec:
    slug: str
    rpd: int
    rpm: int
    tpm: int
    inputs: Modality

def load_catalog() -> dict[str, Any]: ...

class ModelEnum(Enum):
    @property
    def slug(self) -> str: ...
    @property
    def rpd(self) -> int: ...
    @property
    def rpm(self) -> int: ...
    @property
    def tpm(self) -> int: ...
    @property
    def inputs(self) -> Modality: ...
    def accepts(self, *modalities: Modality) -> bool: ...
    @classmethod
    def cheapest(cls) -> Self: ...
    @classmethod
    def slugs(cls) -> list[str]: ...

CATALOG: dict[str, Any]

class GeminiModel(ModelEnum):
{members}
'''


def main() -> None:
    catalog = json.loads(CATALOG.read_text())
    members = "\n".join(
        f"    {name} = {entry['slug']!r}" for name, entry in catalog["gemini"].items()
    )
    OUTPUT.write_text(TEMPLATE.format(members=members))
    print(f"wrote {OUTPUT} ({len(catalog['gemini'])} members)")


if __name__ == "__main__":
    main()
