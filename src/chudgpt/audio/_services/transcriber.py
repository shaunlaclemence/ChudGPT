from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from chudgpt._providers.gemini import GeminiModel
from chudgpt._schemas import ChudResponse
from chudgpt.audio._schemas.audio_chunk import AudioChunk, AudioSpan, AudioTranscript
from chudgpt.audio._services.chunker import AudioChunker
from chudgpt.audio._utils.chunks import AudioChunkRules
from chudgpt.audio._utils.transcription import TranscriptionRules

if TYPE_CHECKING:
    from chudgpt._services.text import TextService


class AudioTranscriber:
    def __init__(
        self,
        text: TextService,
        chunker: AudioChunker | None = None,
        *,
        concurrency: int = 8,
    ) -> None:
        self._text = text
        self._chunker = chunker or AudioChunker()
        self._concurrency = concurrency

    async def transcribe(
        self,
        file_path: Path,
        *,
        prompt: str = TranscriptionRules.DEFAULT_PROMPT,
        model: GeminiModel = GeminiModel.FLASH_LITE_3_5,
    ) -> AudioTranscript:
        AudioChunkRules.guard_model(model)
        size = AudioChunkRules.batch_size(model, self._concurrency)
        timeout = AudioChunkRules.request_timeout(self._chunker.chunk_seconds)

        segments: list[tuple[AudioSpan, str]] = []
        responses: dict[str, ChudResponse] = {}
        for batch in self.__batches(file_path, size):
            builders = {chunk.span.name: chunk.builder(prompt) for chunk in batch}
            replies = await self._text.parallel_chat(
                builders, dict.fromkeys(builders, model), timeout=timeout
            )
            responses.update(replies)
            segments.extend(
                (chunk.span, replies[chunk.span.name].data) for chunk in batch
            )

        stitched = await self.__stitch(segments, model, timeout)
        if stitched is not None:
            responses[TranscriptionRules.STITCH_KEY] = stitched

        return AudioTranscript(
            text=stitched.data if stitched else TranscriptionRules.join(segments),
            segments=tuple(segments),
            responses=responses,
        )

    async def __stitch(
        self,
        segments: Sequence[tuple[AudioSpan, str]],
        model: GeminiModel,
        timeout: float,
    ) -> ChudResponse | None:
        if not TranscriptionRules.needs_stitching(segments):
            return None
        return await self._text.chat(
            builder=TranscriptionRules.stitch_builder(segments),
            model=model,
            timeout=timeout,
        )

    def __batches(self, file_path: Path, size: int) -> Iterator[list[AudioChunk]]:
        batch: list[AudioChunk] = []
        for chunk in self._chunker.stream(file_path):
            batch.append(chunk)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch
