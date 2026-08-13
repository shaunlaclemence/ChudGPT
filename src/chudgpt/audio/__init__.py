"""Chunked audio for recordings too long to send in one turn.

    from chudgpt.audio import AudioChunker, AudioTranscriber

Needs the audio extra:

    uv add "chudgpt[audio] @ git+https://github.com/shaunlaclemence/ChudGPT.git@v0.4.2"

Importing this package without ``soundfile`` raises
``chudgpt.exceptions.ChudGPTAudioBackendMissingException`` (code 004-424), which is
also an ``ImportError``.
"""

from chudgpt.audio._schemas.audio_chunk import AudioChunk, AudioSpan, AudioTranscript
from chudgpt.audio._schemas.diarization import (
    ChudDiarization,
    ChudSpeaker,
    ChudUtterance,
)
from chudgpt.audio._services.chunker import AudioChunker
from chudgpt.audio._services.diarizer import AudioDiarizer
from chudgpt.audio._services.transcriber import AudioTranscriber
from chudgpt.audio._utils.backend import AudioBackend
from chudgpt.audio._utils.chunks import AudioChunkRules
from chudgpt.audio._utils.language import LanguageRules
from chudgpt.audio._utils.transcription import TranscriptionRules
from chudgpt.audio._utils.vad import VoiceActivity

AudioBackend.probe()

__all__ = [
    "AudioBackend",
    "AudioChunk",
    "AudioChunkRules",
    "AudioChunker",
    "AudioDiarizer",
    "AudioSpan",
    "AudioTranscriber",
    "AudioTranscript",
    "ChudDiarization",
    "ChudSpeaker",
    "ChudUtterance",
    "LanguageRules",
    "TranscriptionRules",
    "VoiceActivity",
]
