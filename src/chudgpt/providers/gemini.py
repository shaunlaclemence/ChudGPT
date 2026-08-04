"""Gemini/Gemma models callable on the free tier.

``config.json`` is the source of truth: members below are generated from it at
import time, so adding a model there adds an enum member with no code change.

Limits verified 2026-08-04 against the account's AI Studio rate-limit dashboard;
they are per-account and Google changes them without notice, so a 429 remains the
only authoritative limit. Slugs came from models.list on this project's own key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, Flag, auto
from importlib import resources
from typing import Any, Self


class Modality(Flag):
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    rpd: int
    rpm: int
    tpm: int
    inputs: Modality


def load_catalog() -> dict[str, Any]:
    """Read the packaged model catalog.

    Packaged inside ``chudgpt`` rather than left at the repo root so it is still
    readable once the wheel is pip-installed, where no repo root exists.
    """
    raw = resources.files("chudgpt").joinpath("config.json").read_text()
    return json.loads(raw)


def _modality(names: list[str]) -> Modality:
    flag = Modality(0)
    for name in names:
        flag |= Modality[name.upper()]
    return flag


def _spec(entry: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        slug=entry["slug"],
        rpd=entry["rpd"],
        rpm=entry["rpm"],
        tpm=entry["tpm"],
        inputs=_modality(entry["inputs"]),
    )


class ModelEnum(Enum):
    """Member-less base so a concrete enum can be built from the catalog."""

    @property
    def slug(self) -> str:
        return self.value.slug

    @property
    def rpd(self) -> int:
        return self.value.rpd

    @property
    def rpm(self) -> int:
        return self.value.rpm

    @property
    def tpm(self) -> int:
        return self.value.tpm

    @property
    def inputs(self) -> Modality:
        return self.value.inputs

    def accepts(self, *modalities: Modality) -> bool:
        return all(m in self.inputs for m in modalities)

    @classmethod
    def cheapest(cls) -> Self:
        return max(cls, key=lambda m: (m.rpd, m.rpm))

    @classmethod
    def slugs(cls) -> list[str]:
        return [m.slug for m in cls]

    @classmethod
    def _missing_(cls, value: object) -> ModelEnum | None:
        """Allow lookup by slug, e.g. ``GeminiModel("gemini-3.6-flash")``."""
        for member in cls:
            if member.slug == value:
                return member
        return None


CATALOG = load_catalog()

GeminiModel = ModelEnum(
    "GeminiModel",
    {name: _spec(entry) for name, entry in CATALOG["gemini"].items()},
    module=__name__,
)
