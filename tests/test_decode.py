import gc
from pathlib import Path

import numpy as np
import pytest

from chudgpt.audio._services.chunker import AudioChunker
from chudgpt.audio._services.voice_activity import VoiceActivity
from chudgpt.audio._utils.backend import AudioBackend
from chudgpt.exceptions import ChudGPTBadDataException

PIPER_FRENCH = Path(__file__).parent / "assets" / "piper_french.wav"
SAMPLE_RATE = 48_000


def encode(path: Path, codec: str, container: str) -> Path:
    av = pytest.importorskip("av")
    target = path / f"clip.{container}"
    seconds = 6.0
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    samples = (0.4 * np.sin(2 * np.pi * 220 * t)).astype("f4")
    samples[int(SAMPLE_RATE * 2.0) : int(SAMPLE_RATE * 3.0)] = 0.0

    with av.open(str(target), "w") as out:
        stream = out.add_stream(codec, rate=SAMPLE_RATE)
        stream.layout = "mono"
        size = 960
        for start in range(0, len(samples) - size + 1, size):
            frame = av.AudioFrame.from_ndarray(
                samples[start : start + size].reshape(1, -1),
                format="flt",
                layout="mono",
            )
            frame.sample_rate = SAMPLE_RATE
            frame.pts = start
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return target


@pytest.fixture()
def webm(tmp_path):
    return encode(tmp_path, "libopus", "webm")


def test_libsndfile_cannot_open_webm_alone(webm):
    import soundfile as sf

    with pytest.raises(sf.LibsndfileError):
        sf.info(str(webm))


def test_webm_is_decoded_for_reading(webm):
    backend = AudioBackend()

    assert backend.readable(webm).suffix == ".wav"
    assert backend.sample_rate(webm) == SAMPLE_RATE
    assert backend.channels(webm) == 1


def test_formats_libsndfile_handles_are_not_decoded():
    backend = AudioBackend()

    assert backend.readable(PIPER_FRENCH) == PIPER_FRENCH


def test_one_decode_is_reused_across_reads(webm):
    backend = AudioBackend()

    first = backend.readable(webm)
    backend.sample_rate(webm)
    backend.channels(webm)

    assert backend.readable(webm) == first
    assert len(backend._decoded) == 1


def test_voice_activity_finds_the_silence_in_a_webm(webm):
    spans = VoiceActivity(AudioBackend()).utterances(webm)

    assert len(spans) == 2
    assert spans[0].end == pytest.approx(2.0, abs=0.1)
    assert spans[1].start == pytest.approx(3.0, abs=0.1)


def test_a_webm_chunks_like_any_other_recording(webm):
    chunks = list(AudioChunker(AudioBackend(), chunk_seconds=2.0).stream(webm))

    assert [c.span.name for c in chunks] == ["chunk-000", "chunk-001", "chunk-002"]
    assert all(c.data for c in chunks)


def test_a_corrupt_container_still_raises_bad_data(tmp_path):
    bad = tmp_path / "corrupt.webm"
    bad.write_bytes(b"not audio at all, just bytes")

    with pytest.raises(ChudGPTBadDataException):
        VoiceActivity(AudioBackend()).utterances(bad)


def test_decoded_files_are_cleaned_up_with_the_backend(webm):
    backend = AudioBackend()
    backend.readable(webm)
    workspace = backend._workspace

    assert workspace.exists()
    del backend
    gc.collect()

    assert not workspace.exists()
