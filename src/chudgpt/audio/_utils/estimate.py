from __future__ import annotations

from collections.abc import Sequence

from chudgpt.audio._schemas.audio_chunk import AudioSpan

Range = tuple[int, int]


class EstimateRules:
    # fitted against tests/outputs/test_stream_odyssey_*.json: 60s chunks bill
    # ~1500 prompt tokens of audio and carry ~1500 hidden thinking tokens each
    AUDIO_TOKENS_PER_SECOND = 25
    REQUEST_OVERHEAD = (200, 420)
    RANGE_OVERHEAD = (10, 28)
    REASONING_PER_REQUEST = (1_150, 1_950)
    SPEECH_TOKENS_PER_SECOND = (2.5, 6.0)
    UTTERANCE_OVERHEAD = (12, 35)
    TRANSLATION_FACTOR = 2.0
    RECONCILE_PROMPT = (400, 1_400)
    RECONCILE_COMPLETION = (200, 900)

    @classmethod
    def plan(
        cls,
        duration: float,
        sample_rate: int,
        chunk_seconds: float,
        overlap_seconds: float,
        limit_seconds: float | None = None,
    ) -> list[AudioSpan]:
        frames = max(1, int(chunk_seconds * sample_rate))
        overlap = int(overlap_seconds * sample_rate) if overlap_seconds > 0 else 0
        overlap = min(max(overlap, 0), frames - 1)
        step = frames - overlap
        total = int(min(duration, limit_seconds or duration) * sample_rate)

        chunks: list[AudioSpan] = []
        index = 0
        while index * step < total:
            start = index * step
            length = min(frames, total - start)
            chunks.append(
                AudioSpan(
                    index=index,
                    start=start / sample_rate,
                    end=(start + length) / sample_rate,
                )
            )
            index += 1
        return chunks

    @staticmethod
    def carried(chunk: AudioSpan, spans: Sequence[AudioSpan]) -> list[AudioSpan]:
        return [s for s in spans if chunk.start <= s.start < chunk.end]

    @classmethod
    def prompt_tokens(cls, audio_seconds: float, ranges: int, sent: int) -> Range:
        audio = int(audio_seconds * cls.AUDIO_TOKENS_PER_SECOND)
        return (
            audio + sent * cls.REQUEST_OVERHEAD[0] + ranges * cls.RANGE_OVERHEAD[0],
            audio + sent * cls.REQUEST_OVERHEAD[1] + ranges * cls.RANGE_OVERHEAD[1],
        )

    @classmethod
    def completion_tokens(
        cls, speech_seconds: float, ranges: int, translate_to: str | None
    ) -> Range:
        factor = cls.TRANSLATION_FACTOR if translate_to else 1.0
        spoken = (
            speech_seconds * cls.SPEECH_TOKENS_PER_SECOND[0],
            speech_seconds * cls.SPEECH_TOKENS_PER_SECOND[1],
        )
        return (
            int((spoken[0] + ranges * cls.UTTERANCE_OVERHEAD[0]) * factor),
            int((spoken[1] + ranges * cls.UTTERANCE_OVERHEAD[1]) * factor),
        )

    @classmethod
    def reasoning_tokens(cls, sent: int) -> Range:
        return (
            sent * cls.REASONING_PER_REQUEST[0],
            sent * cls.REASONING_PER_REQUEST[1],
        )

    @classmethod
    def reconcile_tokens(cls, sent: int) -> Range:
        if sent < 2:
            return (0, 0)
        return (
            cls.RECONCILE_PROMPT[0] + cls.RECONCILE_COMPLETION[0],
            cls.RECONCILE_PROMPT[1] + cls.RECONCILE_COMPLETION[1],
        )

    @classmethod
    def requests(cls, sent: int) -> int:
        return sent + 1 if sent > 1 else sent

    @classmethod
    def total(
        cls,
        audio_seconds: float,
        speech_seconds: float,
        ranges: int,
        sent: int,
        translate_to: str | None,
    ) -> Range:
        parts = (
            cls.prompt_tokens(audio_seconds, ranges, sent),
            cls.completion_tokens(speech_seconds, ranges, translate_to),
            cls.reasoning_tokens(sent),
            cls.reconcile_tokens(sent),
        )
        return (sum(p[0] for p in parts), sum(p[1] for p in parts))
