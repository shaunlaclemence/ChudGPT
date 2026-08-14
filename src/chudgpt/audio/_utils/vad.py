from __future__ import annotations

from typing import Any

import numpy as np

from chudgpt.audio._schemas.audio_chunk import AudioSpan


class VoiceActivityRules:
    FRAME_SECONDS = 0.03
    SILENCE_SECONDS = 0.4
    MIN_UTTERANCE_SECONDS = 0.2
    THRESHOLD_RATIO = 0.06
    SPEECH_PERCENTILE = 95

    @classmethod
    def spans(cls, levels: Any, frame_seconds: float) -> list[AudioSpan]:
        voiced = levels > cls.threshold(levels)
        gap = max(1, round(cls.SILENCE_SECONDS / frame_seconds))
        spans = []
        for start, end in cls.runs(voiced, gap):
            span = AudioSpan(
                index=len(spans),
                start=round(start * frame_seconds, 3),
                end=round(end * frame_seconds, 3),
            )
            if span.duration >= cls.MIN_UTTERANCE_SECONDS:
                spans.append(span)
        return spans

    @classmethod
    def threshold(cls, levels: Any) -> float:
        speech = levels[levels > 0]
        if speech.size == 0:
            return float("inf")
        return float(np.percentile(speech, cls.SPEECH_PERCENTILE)) * cls.THRESHOLD_RATIO

    @staticmethod
    def runs(voiced: Any, gap: int) -> list[tuple[int, int]]:
        runs: list[list[int]] = []
        for index in np.flatnonzero(voiced).tolist():
            if runs and index - runs[-1][1] <= gap:
                runs[-1][1] = index + 1
            else:
                runs.append([index, index + 1])
        return [(start, end) for start, end in runs]

    @staticmethod
    def frame_rms(block: Any, frame: int) -> Any:
        samples = np.asarray(block, dtype=np.float64)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        usable = samples.size - samples.size % frame
        if usable == 0:
            return np.zeros(0)
        framed = samples[:usable].reshape(-1, frame)
        return np.sqrt((framed**2).mean(axis=1))
