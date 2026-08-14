from __future__ import annotations

import importlib
import io
import shutil
import tempfile
import weakref
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from chudgpt._utils.version import VersionRules
from chudgpt.exceptions import ChudGPTAudioBackendMissingException


class AudioBackend:
    REQUIRED = ("numpy", "soundfile")
    DECODER = "av"
    DECODED_RATE = 48_000
    INSTALL = VersionRules.install_command("audio")

    def __init__(self, soundfile: Any | None = None) -> None:
        self._sf = soundfile if soundfile is not None else self.probe()
        self._decoded: dict[Path, Path] = {}
        self._workspace: Path | None = None

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
        return int(self._sf.info(str(self.readable(file_path))).samplerate)

    def channels(self, file_path: Path) -> int:
        return int(self._sf.info(str(self.readable(file_path))).channels)

    def blocks(
        self, file_path: Path, frames: int, overlap: int = 0, limit: int = -1
    ) -> Iterator[Any]:
        for block in self._sf.blocks(
            str(self.readable(file_path)),
            blocksize=frames,
            overlap=overlap,
            frames=limit,
        ):
            if len(block) == 0:
                break
            yield block

    def readable(self, file_path: Path) -> Path:
        if file_path in self._decoded:
            return self._decoded[file_path]
        try:
            self._sf.info(str(file_path))
        except self._sf.LibsndfileError:
            self._decoded[file_path] = self.__decode(file_path)
            return self._decoded[file_path]
        return file_path

    def __decode(self, file_path: Path) -> Path:
        av = self.__decoder()
        target = self.__workspace() / f"{file_path.stem}.wav"
        with av.open(str(file_path)) as container:
            if not container.streams.audio:
                raise ValueError(f"{file_path.name} carries no audio stream")
            stream = container.streams.audio[0]
            rate = stream.rate or self.DECODED_RATE
            resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
            with self._sf.SoundFile(
                str(target), mode="w", samplerate=rate, channels=1, subtype="PCM_16"
            ) as out:
                for frame in container.decode(stream):
                    self.__write(out, resampler.resample(frame))
                self.__write(out, resampler.resample(None))
        return target

    @staticmethod
    def __write(out: Any, frames: Iterator[Any]) -> None:
        for frame in frames:
            out.write(frame.to_ndarray().reshape(-1))

    def __workspace(self) -> Path:
        if self._workspace is None:
            self._workspace = Path(tempfile.mkdtemp(prefix="chudgpt-audio-"))
            weakref.finalize(self, shutil.rmtree, self._workspace, True)
        return self._workspace

    @classmethod
    def __decoder(cls) -> Any:
        try:
            return importlib.import_module(cls.DECODER)
        except ImportError as err:
            raise ChudGPTAudioBackendMissingException(
                f"this container needs {cls.DECODER} to decode, which is not "
                f"installed. install the audio extra: {cls.INSTALL}",
                err,
            ) from err

    def encode(self, samples: Any, sample_rate: int, format: str) -> bytes:
        buffer = io.BytesIO()
        self._sf.write(buffer, samples, sample_rate, format=format.upper())
        return buffer.getvalue()
