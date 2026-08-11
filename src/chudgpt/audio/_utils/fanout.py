from __future__ import annotations

from collections.abc import Sequence

from chudgpt._providers.gemini import GeminiModel, Modality
from chudgpt.audio._schemas.audio_chunk import AudioSpan
from chudgpt.exceptions import ChudGPTBadDataException, ServiceCode


class AudioFanOutRules:
    @classmethod
    def guard_model(cls, model: GeminiModel) -> None:
        if not model.accepts(Modality.AUDIO):
            raise ChudGPTBadDataException(
                f"{model.slug} does not accept audio input",
                ServiceCode.AUDIO_SERVICE,
            )

    @classmethod
    def batch_size(cls, model: GeminiModel, concurrency: int) -> int:
        return max(1, min(concurrency, model.rpm))

    @classmethod
    def stitch(cls, segments: Sequence[tuple[AudioSpan, str]]) -> str:
        return " ".join(text.strip() for _, text in segments if text.strip())
