from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from chudgpt.audio._schemas.audio_chunk import AudioSpan
from chudgpt.audio._services.exceptions.audio import audio_exception_handler
from chudgpt.audio._utils.backend import AudioBackend
from chudgpt.audio._utils.vad import VoiceActivityRules


class VoiceActivity:
    READ_SECONDS = 30.0

    def __init__(self, backend: AudioBackend | None = None) -> None:
        self._backend = backend or AudioBackend()
        self._rules = VoiceActivityRules()

    @audio_exception_handler
    def utterances(self, file_path: Path) -> list[AudioSpan]:
        sample_rate = self._backend.sample_rate(file_path)
        frame = max(1, int(self._rules.FRAME_SECONDS * sample_rate))
        levels = self.levels(file_path, frame)
        return self._rules.spans(levels, frame / sample_rate)

    @audio_exception_handler
    def levels(self, file_path: Path, frame: int) -> Any:
        read = frame * max(1, int(self.READ_SECONDS / self._rules.FRAME_SECONDS))
        chunks = [
            self._rules.frame_rms(block, frame)
            for block in self._backend.blocks(file_path, read)
        ]
        return np.concatenate(chunks) if chunks else np.zeros(0)
