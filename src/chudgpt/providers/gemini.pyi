# GENERATED FILE - do not edit.
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
    FLASH_3_6 = "gemini-3.6-flash"
    FLASH_3_5 = "gemini-3.5-flash"
    FLASH_LITE_3_5 = "gemini-3.5-flash-lite"
    FLASH_LITE_3_1 = "gemini-3.1-flash-lite"
    FLASH_3_PREVIEW = "gemini-3-flash-preview"
    FLASH_2_5 = "gemini-2.5-flash"
    FLASH_LITE_2_5 = "gemini-2.5-flash-lite"
    GEMMA_4_26B = "gemma-4-26b-a4b-it"
    GEMMA_4_31B = "gemma-4-31b-it"
