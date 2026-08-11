from __future__ import annotations

from chudgpt.exceptions import ChudGPTBadDataException, ServiceCode


class AudioChunkRules:
    # 300s is ~7500 input tokens and ~1100 output tokens, well clear of the
    # completion cap that truncates and derails whole file transcription
    DEFAULT_SECONDS = 300.0
    MAX_INLINE_BYTES = 20_000_000
    B64_RATIO = 4 / 3
    PCM_BYTES_PER_FRAME = 2
    # measured 5350 B/s on a 22050 mono source; 16000 leaves headroom for
    # richer sources without needing to probe the encoder
    MP3_BYTES_PER_SECOND = 16_000
    PREFERRED_FORMATS = ("mp3", "wav")

    @classmethod
    def frames(cls, seconds: float, sample_rate: int) -> int:
        return max(1, int(seconds * sample_rate))

    @classmethod
    def encode_format(cls, writable: set[str]) -> str:
        for format in cls.PREFERRED_FORMATS:
            if format in writable:
                return format
        raise ChudGPTBadDataException(
            f"audio backend cannot write any of {', '.join(cls.PREFERRED_FORMATS)}",
            ServiceCode.AUDIO_SERVICE,
        )

    @classmethod
    def bytes_per_second(cls, sample_rate: int, channels: int, format: str) -> int:
        if format == "wav":
            return sample_rate * channels * cls.PCM_BYTES_PER_FRAME
        return cls.MP3_BYTES_PER_SECOND

    @classmethod
    def max_seconds(cls, sample_rate: int, channels: int, format: str) -> float:
        budget = cls.MAX_INLINE_BYTES / cls.B64_RATIO
        return budget / cls.bytes_per_second(sample_rate, channels, format)

    @classmethod
    def guard_span(
        cls, seconds: float, sample_rate: int, channels: int, format: str
    ) -> None:
        if seconds <= 0:
            raise ChudGPTBadDataException(
                f"chunk_seconds must be positive, got {seconds}",
                ServiceCode.AUDIO_SERVICE,
            )
        limit = cls.max_seconds(sample_rate, channels, format)
        if seconds > limit:
            raise ChudGPTBadDataException(
                f"a {seconds:.0f}s {format} chunk at {sample_rate}Hz x{channels} "
                f"exceeds the {cls.MAX_INLINE_BYTES / 1e6:.0f}MB inline limit once "
                f"base64 encoded; keep chunk_seconds under {limit:.0f}",
                ServiceCode.AUDIO_SERVICE,
            )
