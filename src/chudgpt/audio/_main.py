from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from chudgpt._providers.gemini import GeminiModel
from chudgpt.audio._schemas.audio_chunk import AudioChunk, AudioSpan, AudioTranscript
from chudgpt.audio._schemas.diarization import ChudDiarization
from chudgpt.audio._schemas.progress import ChudProgress
from chudgpt.audio._services.chunker import AudioChunker
from chudgpt.audio._services.diarizer import AudioDiarizer
from chudgpt.audio._services.transcriber import AudioTranscriber
from chudgpt.audio._services.voice_activity import VoiceActivity
from chudgpt.audio._utils.chunks import AudioChunkRules
from chudgpt.audio._utils.transcription import TranscriptionRules
from chudgpt.messages import ChudMessageBuilder

if TYPE_CHECKING:
    from chudgpt import ChudGPT


class AudioService:
    def __init__(self, client: ChudGPT) -> None:
        self._client = client

    async def transcribe(
        self,
        file_path: Path,
        *,
        prompt: str = TranscriptionRules.DEFAULT_PROMPT,
        model: GeminiModel = GeminiModel.FLASH_LITE_3_5,
        chunk_seconds: float = AudioChunkRules.DEFAULT_SECONDS,
        overlap_seconds: float = 0.0,
        limit_seconds: float | None = None,
        format: str | None = None,
        concurrency: int = 8,
    ) -> AudioTranscript:
        transcriber = AudioTranscriber(
            self._client.text,
            self.__chunker(chunk_seconds, overlap_seconds, limit_seconds, format),
            concurrency=concurrency,
        )
        return await transcriber.transcribe(file_path, prompt=prompt, model=model)

    async def diarize(
        self,
        file_path: Path,
        *,
        translate_to: str | None = None,
        max_speakers: int | None = None,
        model: GeminiModel = GeminiModel.FLASH_LITE_3_5,
        chunk_seconds: float = AudioChunkRules.DEFAULT_SECONDS,
        overlap_seconds: float = 0.0,
        limit_seconds: float | None = None,
        format: str | None = None,
        concurrency: int = 8,
    ) -> ChudDiarization:
        return await self.__diarizer(
            chunk_seconds, overlap_seconds, limit_seconds, format, concurrency
        ).diarize(
            file_path,
            translate_to=translate_to,
            max_speakers=max_speakers,
            model=model,
        )

    def diarize_stream(
        self,
        file_path: Path,
        *,
        translate_to: str | None = None,
        max_speakers: int | None = None,
        model: GeminiModel = GeminiModel.FLASH_LITE_3_5,
        chunk_seconds: float = AudioChunkRules.DEFAULT_SECONDS,
        overlap_seconds: float = 0.0,
        limit_seconds: float | None = None,
        format: str | None = None,
        concurrency: int = 8,
    ) -> AsyncIterator[ChudProgress]:
        return self.__diarizer(
            chunk_seconds, overlap_seconds, limit_seconds, format, concurrency
        ).diarize_stream(
            file_path,
            translate_to=translate_to,
            max_speakers=max_speakers,
            model=model,
        )

    def voice_activity(self, file_path: Path) -> list[AudioSpan]:
        return VoiceActivity().utterances(file_path)

    def chunks(
        self,
        file_path: Path,
        *,
        chunk_seconds: float = AudioChunkRules.DEFAULT_SECONDS,
        overlap_seconds: float = 0.0,
        limit_seconds: float | None = None,
        format: str | None = None,
    ) -> Iterator[AudioChunk]:
        return self.__chunker(
            chunk_seconds, overlap_seconds, limit_seconds, format
        ).stream(file_path)

    def builders(
        self,
        file_path: Path,
        prompt: str = TranscriptionRules.DEFAULT_PROMPT,
        *,
        chunk_seconds: float = AudioChunkRules.DEFAULT_SECONDS,
        overlap_seconds: float = 0.0,
        limit_seconds: float | None = None,
        format: str | None = None,
    ) -> dict[str, ChudMessageBuilder]:
        return self.__chunker(
            chunk_seconds, overlap_seconds, limit_seconds, format
        ).builders(file_path, prompt)

    def __diarizer(
        self,
        chunk_seconds: float,
        overlap_seconds: float,
        limit_seconds: float | None,
        format: str | None,
        concurrency: int,
    ) -> AudioDiarizer:
        return AudioDiarizer(
            self._client.text,
            self.__chunker(chunk_seconds, overlap_seconds, limit_seconds, format),
            concurrency=concurrency,
        )

    @staticmethod
    def __chunker(
        chunk_seconds: float,
        overlap_seconds: float,
        limit_seconds: float | None,
        format: str | None,
    ) -> AudioChunker:
        return AudioChunker(
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            limit_seconds=limit_seconds,
            format=format,
        )
