from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from chudgpt.audio._schemas.audio_chunk import AudioChunk, AudioSpan
from chudgpt.audio._services.exceptions.audio import audio_exception_handler
from chudgpt.audio._utils.backend import AudioBackend
from chudgpt.audio._utils.chunks import AudioChunkRules
from chudgpt.messages import ChudMessageBuilder


class AudioChunker:
    def __init__(
        self,
        backend: AudioBackend | None = None,
        *,
        chunk_seconds: float = AudioChunkRules.DEFAULT_SECONDS,
        overlap_seconds: float = 0.0,
        limit_seconds: float | None = None,
        format: str | None = None,
    ) -> None:
        self._backend = backend or AudioBackend()
        self._chunk_seconds = chunk_seconds
        self._overlap_seconds = overlap_seconds
        self._limit_seconds = limit_seconds
        self._format = format

    @property
    def chunk_seconds(self) -> float:
        return self._chunk_seconds

    @property
    def overlap_seconds(self) -> float:
        return self._overlap_seconds

    @audio_exception_handler
    def stream(self, file_path: Path) -> Iterator[AudioChunk]:
        sample_rate = self._backend.sample_rate(file_path)
        channels = self._backend.channels(file_path)
        format = self._format or AudioChunkRules.encode_format(
            self._backend.writable_formats()
        )
        AudioChunkRules.guard_span(self._chunk_seconds, sample_rate, channels, format)

        frames = AudioChunkRules.frames(self._chunk_seconds, sample_rate)
        overlap = AudioChunkRules.frames(self._overlap_seconds, sample_rate)
        overlap = overlap if self._overlap_seconds > 0 else 0
        limit = (
            AudioChunkRules.frames(self._limit_seconds, sample_rate)
            if self._limit_seconds
            else -1
        )
        step = (frames - overlap) / sample_rate

        for index, block in enumerate(
            self._backend.blocks(file_path, frames, overlap, limit)
        ):
            start = index * step
            yield AudioChunk(
                span=AudioSpan(
                    index=index, start=start, end=start + len(block) / sample_rate
                ),
                data=self._backend.encode(block, sample_rate, format),
                format=format,
                sample_rate=sample_rate,
            )

    def builders(self, file_path: Path, prompt: str) -> dict[str, ChudMessageBuilder]:
        return {
            chunk.span.name: chunk.builder(prompt) for chunk in self.stream(file_path)
        }
