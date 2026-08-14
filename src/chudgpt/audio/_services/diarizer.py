from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from chudgpt._providers.gemini import GeminiModel
from chudgpt._schemas import ChudChannel, ChudResponse
from chudgpt.audio._schemas.audio_chunk import AudioChunk, AudioSpan
from chudgpt.audio._schemas.diarization import (
    ChudChunkDiarization,
    ChudDiarization,
    ChudSpeaker,
    ChudSpeakerRoster,
    ChudUtterance,
)
from chudgpt.audio._schemas.progress import ChudProgress
from chudgpt.audio._services.chunker import AudioChunker
from chudgpt.audio._services.voice_activity import VoiceActivity
from chudgpt.audio._utils.chunks import AudioChunkRules
from chudgpt.audio._utils.diarization import DiarizationRules
from chudgpt.audio._utils.progress import ProgressRules
from chudgpt.exceptions import ChudGPTBadDataException, ServiceCode

if TYPE_CHECKING:
    from chudgpt._services.text import TextService


class AudioDiarizer:
    TEMPERATURE = 0.0

    def __init__(
        self,
        text: TextService,
        chunker: AudioChunker | None = None,
        voice: VoiceActivity | None = None,
        *,
        concurrency: int = 8,
    ) -> None:
        self._text = text
        self._chunker = chunker or AudioChunker()
        self._voice = voice or VoiceActivity()
        self._concurrency = concurrency

    async def diarize(
        self,
        file_path: Path,
        *,
        translate_to: str | None = None,
        max_speakers: int | None = None,
        model: GeminiModel = GeminiModel.FLASH_LITE_3_5,
    ) -> ChudDiarization:
        result: ChudDiarization | None = None
        async for update in self.diarize_stream(
            file_path,
            translate_to=translate_to,
            max_speakers=max_speakers,
            model=model,
        ):
            if update.result is not None:
                result = update.result
        if result is None:
            raise ChudGPTBadDataException(
                "diarization produced no result", ServiceCode.AUDIO_SERVICE
            )
        return result

    async def diarize_stream(
        self,
        file_path: Path,
        *,
        translate_to: str | None = None,
        max_speakers: int | None = None,
        model: GeminiModel = GeminiModel.FLASH_LITE_3_5,
    ) -> AsyncIterator[ChudProgress]:
        AudioChunkRules.guard_model(model)
        spans = self._voice.utterances(file_path)
        size = AudioChunkRules.batch_size(model, self._concurrency)
        timeout = AudioChunkRules.request_timeout(self._chunker.chunk_seconds)
        schema = DiarizationRules.provider_schema(ChudChunkDiarization)

        clock = time.monotonic()
        utterances: list[ChudUtterance] = []
        speakers: list[ChudSpeaker] = []
        responses: dict[str, ChudResponse] = {}

        for batch in self.__batches(file_path, size, spans):
            ranges = {
                c.span.name: DiarizationRules.within(spans, c.span) for c in batch
            }
            spans_by_name = {c.span.name: c.span for c in batch}
            builders = {
                chunk.span.name: chunk.builder(
                    DiarizationRules.prompt(
                        ranges[chunk.span.name],
                        chunk.span,
                        translate_to,
                        max_speakers,
                    )
                )
                for chunk in batch
            }
            for chunk in batch:
                yield ProgressRules.queued(
                    chunk.span.name,
                    chunk.span,
                    len(ranges[chunk.span.name]),
                    time.monotonic() - clock,
                )

            partial = dict.fromkeys(builders, "")
            latest: dict[str, tuple] = {}
            async for name, event in self._text.parallel_stream(
                builders,
                dict.fromkeys(builders, model),
                response_format=self.__format(schema),
                temperature=self.TEMPERATURE,
                timeout=timeout,
            ):
                span = spans_by_name[name]
                if event.channel is ChudChannel.ERROR:
                    yield ProgressRules.failed(
                        name, span, event.text, time.monotonic() - clock
                    )
                    continue
                if event.channel is ChudChannel.DONE:
                    responses[name] = event.response
                    reply = event.response.parse(ChudChunkDiarization)
                    built = DiarizationRules.utterances(
                        span, ranges[name], reply, translate_to
                    )
                    utterances.extend(built)
                    speakers.extend(
                        ChudSpeaker(
                            label=DiarizationRules.namespace(span, s.label),
                            description=s.description,
                        )
                        for s in reply.speakers
                    )
                    yield ProgressRules.finished(
                        name,
                        span,
                        len(built),
                        DiarizationRules.languages(built),
                        time.monotonic() - clock,
                    )
                    continue

                partial[name] += event.text
                update = ProgressRules.streaming(
                    name,
                    span,
                    partial[name],
                    len(ranges[name]),
                    translate_to,
                    time.monotonic() - clock,
                )
                if latest.get(name) != update.key:
                    latest[name] = update.key
                    yield update

        transcript = DiarizationRules.dedup(utterances)
        if len(speakers) > 1:
            yield ProgressRules.merging(len(speakers), time.monotonic() - clock)
        reconciled = await self.__reconcile(
            speakers, transcript, max_speakers, model, timeout
        )
        roster = reconciled.parse(ChudSpeakerRoster) if reconciled else None
        if reconciled is not None:
            responses[DiarizationRules.RECONCILE_KEY] = reconciled
        mapping = DiarizationRules.speaker_map(speakers, roster)

        yield ProgressRules.assembled(
            ChudDiarization(
                languages=DiarizationRules.languages(transcript),
                translate_to=translate_to,
                transcript=DiarizationRules.apply_map(transcript, mapping),
                speakers=DiarizationRules.global_speakers(speakers, roster, mapping),
                speaker_map=mapping,
                responses=responses,
            ),
            time.monotonic() - clock,
        )

    async def __reconcile(
        self,
        speakers: Sequence[ChudSpeaker],
        transcript: Sequence[ChudUtterance],
        max_speakers: int | None,
        model: GeminiModel,
        timeout: float,
    ) -> ChudResponse | None:
        if len(speakers) < 2:
            return None
        return await self._text.chat(
            DiarizationRules.roster_prompt(speakers, transcript, max_speakers),
            model=model,
            response_format=self.__format(
                DiarizationRules.provider_schema(ChudSpeakerRoster), "speaker_roster"
            ),
            temperature=self.TEMPERATURE,
            timeout=timeout,
        )

    @staticmethod
    def __format(schema: dict, name: str = "diarization") -> dict:
        return {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema},
        }

    def __batches(
        self, file_path: Path, size: int, spans: Sequence[AudioSpan]
    ) -> Iterator[list[AudioChunk]]:
        batch: list[AudioChunk] = []
        for chunk in self._chunker.stream(file_path):
            if not DiarizationRules.within(spans, chunk.span):
                continue
            batch.append(chunk)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch
