from __future__ import annotations

import importlib
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from chudgpt.exceptions import ChudGPTAudioBackendMissingException


class AudioBackend:
    REQUIRED = ("soundfile",)
    INSTALL = (
        'uv add "chudgpt[audio] @ '
        'git+https://github.com/shaunlaclemence/ChudGPT.git@v0.4.2"'
    )

    def __init__(self, soundfile: Any | None = None) -> None:
        self._sf = soundfile if soundfile is not None else self.probe()

    @classmethod
    def probe(cls) -> Any:
        missing: list[str] = []
        error: ImportError | None = None
        for name in cls.REQUIRED:
            try:
                importlib.import_module(name)
            except ImportError as err:
                missing.append(name)
                error = err
        if missing:
            raise ChudGPTAudioBackendMissingException(
                f"chudgpt.audio needs {', '.join(missing)}, which is not installed. "
                f"install the audio extra: {cls.INSTALL}",
                error,
            )
        return importlib.import_module("soundfile")

    def writable_formats(self) -> set[str]:
        return {name.lower() for name in self._sf.available_formats()}

    def sample_rate(self, file_path: Path) -> int:
        return int(self._sf.info(str(file_path)).samplerate)

    def channels(self, file_path: Path) -> int:
        return int(self._sf.info(str(file_path)).channels)

    def blocks(
        self, file_path: Path, frames: int, overlap: int = 0, limit: int = -1
    ) -> Iterator[Any]:
        for block in self._sf.blocks(
            str(file_path), blocksize=frames, overlap=overlap, frames=limit
        ):
            if len(block) == 0:
                break
            yield block

    def encode(self, samples: Any, sample_rate: int, format: str) -> bytes:
        buffer = io.BytesIO()
        self._sf.write(buffer, samples, sample_rate, format=format.upper())
        return buffer.getvalue()
