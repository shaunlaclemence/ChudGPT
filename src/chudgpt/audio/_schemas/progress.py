from __future__ import annotations

from enum import Enum
from typing import ClassVar

from pydantic import BaseModel

from chudgpt.audio._schemas.audio_chunk import AudioSpan
from chudgpt.audio._schemas.diarization import ChudDiarization


class ChudPhase(str, Enum):
    QUEUED = "queued"
    SENT = "sent"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    MERGING = "merging"
    DONE = "done"
    FAILED = "failed"


class ChudProgress(BaseModel):
    GLOBAL: ClassVar[str] = "__global__"

    chunk: str
    phase: ChudPhase
    detail: str = ""
    at: float = 0.0
    span: AudioSpan | None = None
    utterances: int = 0
    languages: list[str] = []
    result: ChudDiarization | None = None

    @property
    def key(self) -> tuple[str, ChudPhase, int, tuple[str, ...]]:
        return (self.chunk, self.phase, self.utterances, tuple(self.languages))

    @property
    def is_global(self) -> bool:
        return self.chunk == self.GLOBAL
