from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from chudgpt import ChudGPT
from chudgpt._providers.gemini import GeminiModel
from chudgpt.audio._schemas.audio_chunk import AudioChunk, AudioSpan, AudioTranscript
from chudgpt.audio._services.chunker import AudioChunker
from chudgpt.audio._utils.fanout import AudioFanOutRules


class AudioTranscriber:
    # telling it not to worry about mid sentence boundaries made it stop after
    # the first sentence; demanding the whole clip is what gets a full transcript
    DEFAULT_PROMPT = (
        "Transcribe the ENTIRE audio clip from start to finish, word for word. "
        "Do not stop early, do not summarise, do not skip any speech. "
        "Output only the transcript text."
    )

    def __init__(
        self,
        client: ChudGPT,
        chunker: AudioChunker | None = None,
        *,
        concurrency: int = 8,
    ) -> None:
        self._client = client
        self._chunker = chunker or AudioChunker()
        self._concurrency = concurrency

    async def transcribe(
        self,
        file_path: Path,
        *,
        prompt: str = DEFAULT_PROMPT,
        model: GeminiModel = GeminiModel.FLASH_LITE_3_5,
    ) -> AudioTranscript:
        AudioFanOutRules.guard_model(model)
        size = AudioFanOutRules.batch_size(model, self._concurrency)

        segments: list[tuple[AudioSpan, str]] = []
        responses = {}
        for batch in self.__batches(file_path, size):
            builders = {chunk.span.name: chunk.builder(prompt) for chunk in batch}
            replies = await self._client.parallel_chat(
                builders, dict.fromkeys(builders, model)
            )
            responses.update(replies)
            segments.extend(
                (chunk.span, replies[chunk.span.name].text) for chunk in batch
            )

        return AudioTranscript(
            text=AudioFanOutRules.stitch(segments),
            segments=tuple(segments),
            responses=responses,
        )

    # batches rather than materialising every chunk, so only one batch of
    # encoded audio is held at a time
    def __batches(self, file_path: Path, size: int) -> Iterator[list[AudioChunk]]:
        batch: list[AudioChunk] = []
        for chunk in self._chunker.stream(file_path):
            batch.append(chunk)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch
