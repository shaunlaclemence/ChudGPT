from __future__ import annotations

from pydantic import BaseModel


class ChudEstimate(BaseModel):
    requests: int
    tokens: tuple[int, int]
    chunks: int
    sent: int
    utterances: int
    speech_seconds: float
    audio_seconds: float
    model: str
    translate_to: str | None = None

    @property
    def low(self) -> int:
        return self.tokens[0]

    @property
    def high(self) -> int:
        return self.tokens[1]
