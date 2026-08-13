from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from chudgpt.audio._schemas.audio_chunk import AudioSpan
from chudgpt.audio._utils.backend import AudioBackend


class VoiceActivity:
    """Utterance boundaries computed from the waveform, never from a model.

    A short-term RMS pass: frames above a level derived from the clip's own
    speech level are voiced, runs of voiced frames separated by less than
    SILENCE_SECONDS are one utterance. Output is a pure function of the
    samples, so the same file gives byte-identical spans on every run.
    """

    FRAME_SECONDS = 0.03
    SILENCE_SECONDS = 0.4
    MIN_UTTERANCE_SECONDS = 0.2
    # relative to the clip's own 95th-percentile frame, so quiet and loud
    # recordings threshold the same way
    THRESHOLD_RATIO = 0.06
    SPEECH_PERCENTILE = 95
    READ_SECONDS = 30.0

    def __init__(self, backend: AudioBackend | None = None) -> None:
        self._backend = backend or AudioBackend()

    def utterances(self, file_path: Path) -> list[AudioSpan]:
        sample_rate = self._backend.sample_rate(file_path)
        frame = max(1, int(self.FRAME_SECONDS * sample_rate))
        levels = self.levels(file_path, frame)
        return self.spans(levels, frame / sample_rate)

    def levels(self, file_path: Path, frame: int) -> Any:
        # a whole number of frames per block, so only the final block can drop
        # a sub-frame remainder
        read = frame * max(1, int(self.READ_SECONDS / self.FRAME_SECONDS))
        chunks = [
            self.__frame_rms(block, frame)
            for block in self._backend.blocks(file_path, read)
        ]
        return np.concatenate(chunks) if chunks else np.zeros(0)

    @classmethod
    def spans(cls, levels: Any, frame_seconds: float) -> list[AudioSpan]:
        voiced = levels > cls.threshold(levels)
        gap = max(1, round(cls.SILENCE_SECONDS / frame_seconds))
        spans = []
        for start, end in cls.__runs(voiced, gap):
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
    def __frame_rms(block: Any, frame: int) -> Any:
        samples = np.asarray(block, dtype=np.float64)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        usable = samples.size - samples.size % frame
        if usable == 0:
            return np.zeros(0)
        framed = samples[:usable].reshape(-1, frame)
        return np.sqrt((framed**2).mean(axis=1))

    @staticmethod
    def __runs(voiced: Any, gap: int) -> list[tuple[int, int]]:
        runs: list[list[int]] = []
        for index in np.flatnonzero(voiced).tolist():
            if runs and index - runs[-1][1] <= gap:
                runs[-1][1] = index + 1
            else:
                runs.append([index, index + 1])
        return [(start, end) for start, end in runs]
