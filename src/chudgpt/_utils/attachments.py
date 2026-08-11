from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import filetype

from chudgpt.exceptions import ChudGPTBadDataException, ServiceCode


class AttachmentRules:
    AUDIO_FORMATS = ("wav", "mp3")

    @classmethod
    def resolve(
        cls,
        file_path: Path | None,
        data: bytes | None,
        b64data: str | None,
        format: str | None,
    ) -> tuple[str, str, str]:
        given = [s for s in (file_path, data, b64data) if s is not None]
        if len(given) != 1:
            raise cls.__bad("pass exactly one of file_path, data, or b64data")

        if b64data is not None:
            if not format:
                raise cls.__bad("format is required when passing b64data")
            return b64data, cls.__mime(format), format

        if data is not None:
            extension = format or cls.__sniff(data)
            return base64.b64encode(data).decode(), cls.__mime(extension), extension

        path = Path(file_path)  # type: ignore[arg-type]
        kind = filetype.guess(path)
        if kind is None:
            raise cls.__bad(f"could not detect file type: {path}")
        return base64.b64encode(path.read_bytes()).decode(), kind.mime, kind.extension

    @classmethod
    def part(cls, mime: str, extension: str, b64: str) -> dict[str, Any]:
        if mime.startswith("audio/"):
            if extension not in cls.AUDIO_FORMATS:
                raise cls.__bad(
                    f"unsupported audio format {extension!r}; "
                    f"providers accept {', '.join(cls.AUDIO_FORMATS)}"
                )
            return {
                "type": "input_audio",
                "input_audio": {"data": b64, "format": extension},
            }
        if mime.startswith("image/"):
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        if mime.startswith("video/"):
            return {
                "type": "video_url",
                "video_url": {"url": f"data:{mime};base64,{b64}"},
            }
        raise cls.__bad(f"unsupported attachment type: {mime}")

    @classmethod
    def __sniff(cls, data: bytes) -> str:
        kind = filetype.guess(data)
        if kind is None:
            raise cls.__bad("could not detect type of attachment bytes; pass format")
        return kind.extension

    @classmethod
    def __mime(cls, extension: str) -> str:
        mime, _ = mimetypes.guess_type(f"attachment.{extension.lstrip('.')}")
        if mime is None:
            raise cls.__bad(f"unknown attachment format: {extension!r}")
        return mime

    @staticmethod
    def __bad(message: str) -> ChudGPTBadDataException:
        return ChudGPTBadDataException(message, ServiceCode.FILE_SERVICE)
