from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from chudgpt import ChudGPT
from chudgpt._providers.gemini import GeminiModel
from chudgpt._schemas import ChudResponse
from chudgpt.audio._schemas.audio_chunk import AudioChunk, AudioSpan
from chudgpt.audio._schemas.diarization import (
    ChudChunkDiarization,
    ChudDiarization,
    ChudSpeaker,
    ChudSpeakerRoster,
    ChudUtterance,
)
from chudgpt.audio._services.chunker import AudioChunker
from chudgpt.audio._utils.chunks import AudioChunkRules
from chudgpt.audio._utils.diarization import DiarizationRules
from chudgpt.audio._utils.vad import VoiceActivity


class AudioDiarizer:
    """Who spoke, in what language, and when.

    Timestamps come from ``VoiceActivity``, never from the model: the waveform
    decides the boundaries and the model only fills each range in. Speaker
    labels are chunk-local until a final reconciliation pass merges them, which
    is one text call comparing voice descriptions with no acoustic evidence, so
    it is best-effort and the mapping it chose is returned on
    ``ChudDiarization.speaker_map`` for auditing.
    """

    TEMPERATURE = 0.0

    def __init__(
        self,
        client: ChudGPT,
        chunker: AudioChunker | None = None,
        voice: VoiceActivity | None = None,
        *,
        concurrency: int = 8,
    ) -> None:
        self._client = client
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
        AudioChunkRules.guard_model(model)
        spans = self._voice.utterances(file_path)
        size = AudioChunkRules.batch_size(model, self._concurrency)
        timeout = AudioChunkRules.request_timeout(self._chunker.chunk_seconds)
        schema = DiarizationRules.provider_schema(ChudChunkDiarization)

        utterances: list[ChudUtterance] = []
        speakers: list[ChudSpeaker] = []
        responses: dict[str, ChudResponse] = {}
        for batch in self.__batches(file_path, size, spans):
            ranges = {
                c.span.name: DiarizationRules.within(spans, c.span) for c in batch
            }
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
            replies = await self._client.parallel_chat(
                builders,
                dict.fromkeys(builders, model),
                response_format=self.__format(schema),
                temperature=self.TEMPERATURE,
                timeout=timeout,
            )
            responses.update(replies)
            for chunk in batch:
                reply = replies[chunk.span.name].parse(ChudChunkDiarization)
                utterances.extend(
                    DiarizationRules.utterances(
                        chunk.span, ranges[chunk.span.name], reply, translate_to
                    )
                )
                speakers.extend(
                    ChudSpeaker(
                        label=DiarizationRules.namespace(chunk.span, s.label),
                        description=s.description,
                    )
                    for s in reply.speakers
                )

        transcript = DiarizationRules.dedup(utterances)
        reconciled = await self.__reconcile(
            speakers, transcript, max_speakers, model, timeout
        )
        roster = reconciled.parse(ChudSpeakerRoster) if reconciled else None
        if reconciled is not None:
            responses[DiarizationRules.RECONCILE_KEY] = reconciled
        mapping = DiarizationRules.speaker_map(speakers, roster)

        return ChudDiarization(
            languages=DiarizationRules.languages(transcript),
            translate_to=translate_to,
            transcript=DiarizationRules.apply_map(transcript, mapping),
            speakers=DiarizationRules.global_speakers(speakers, roster, mapping),
            speaker_map=mapping,
            responses=responses,
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
        # chat rather than chat_json: the roster call has to stay a ChudResponse
        # so its usage lands on ChudDiarization.responses
        return await self._client.chat(
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

    # chunks with no detected speech are never sent, so silence costs no quota
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
