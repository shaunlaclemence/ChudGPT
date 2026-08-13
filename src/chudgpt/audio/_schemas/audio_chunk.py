from __future__ import annotations

from pydantic import BaseModel

from chudgpt._schemas import ChudResponse
from chudgpt.messages import Attachment, ChudMessageBuilder


class AudioSpan(BaseModel):
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def name(self) -> str:
        return f"chunk-{self.index:03d}"


class AudioChunk(BaseModel):
    span: AudioSpan
    data: bytes
    format: str
    sample_rate: int

    def attachment(self, prompt: str | None = None) -> Attachment:
        attachment = Attachment(data=self.data, format=self.format)
        return attachment.prompt(prompt) if prompt else attachment

    def builder(self, prompt: str) -> ChudMessageBuilder:
        return ChudMessageBuilder().prompt(self.attachment(prompt))


class AudioTranscript(BaseModel):
    text: str
    segments: tuple[tuple[AudioSpan, str], ...]
    responses: dict[str, ChudResponse]
