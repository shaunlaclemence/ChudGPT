"""Waveform boundaries, which must never depend on a model or a provider.

Determinism is the requirement these tests exist for: the same samples have to
give byte-identical spans on every run, since that is the only thing making the
timestamps in a diarization trustworthy.
"""

import itertools
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from chudgpt.audio import VoiceActivity

PIPER_FRENCH = Path(__file__).parent / "assets" / "piper_french.wav"
SAMPLE_RATE = 16_000


def tone(seconds: float, level: float = 0.5) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * SAMPLE_RATE), endpoint=False)
    return level * np.sin(2 * np.pi * 220 * t)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE))


def write(tmp_path: Path, *parts: np.ndarray) -> Path:
    path = tmp_path / "clip.wav"
    sf.write(path, np.concatenate(parts), SAMPLE_RATE)
    return path


def test_two_tones_split_on_a_long_silence(tmp_path):
    clip = write(tmp_path, tone(1.0), silence(1.0), tone(1.0))

    spans = VoiceActivity().utterances(clip)

    assert len(spans) == 2
    assert spans[0].start == pytest.approx(0.0, abs=0.05)
    assert spans[0].end == pytest.approx(1.0, abs=0.05)
    assert spans[1].start == pytest.approx(2.0, abs=0.05)
    assert spans[1].end == pytest.approx(3.0, abs=0.05)


def test_a_gap_shorter_than_the_silence_window_stays_one_utterance(tmp_path):
    clip = write(tmp_path, tone(1.0), silence(0.2), tone(1.0))

    spans = VoiceActivity().utterances(clip)

    assert len(spans) == 1
    assert spans[0].duration == pytest.approx(2.2, abs=0.1)


def test_pure_silence_has_no_utterances(tmp_path):
    assert VoiceActivity().utterances(write(tmp_path, silence(3.0))) == []


def test_a_blip_shorter_than_the_minimum_is_dropped(tmp_path):
    clip = write(tmp_path, tone(1.0), silence(1.0), tone(0.1), silence(1.0))

    spans = VoiceActivity().utterances(clip)

    assert len(spans) == 1, "the 0.1s blip is below MIN_UTTERANCE_SECONDS"


def test_spans_are_ordered_indexed_and_disjoint(tmp_path):
    clip = write(tmp_path, tone(0.5), silence(0.6), tone(0.5), silence(0.6), tone(0.5))

    spans = VoiceActivity().utterances(clip)

    assert [s.index for s in spans] == list(range(len(spans)))
    for earlier, later in itertools.pairwise(spans):
        assert earlier.end <= later.start


def test_the_same_file_gives_identical_spans_on_every_run():
    vad = VoiceActivity()

    first = vad.utterances(PIPER_FRENCH)
    second = vad.utterances(PIPER_FRENCH)

    assert first == second
    assert first, "the asset is speech, so it must produce at least one span"


def test_level_does_not_change_the_boundaries(tmp_path):
    loud = write(tmp_path, tone(1.0, level=0.9), silence(1.0), tone(1.0, level=0.9))
    quiet_path = tmp_path / "quiet.wav"
    sf.write(
        quiet_path,
        np.concatenate([tone(1.0, level=0.05), silence(1.0), tone(1.0, level=0.05)]),
        SAMPLE_RATE,
    )

    # the threshold is relative to the clip's own speech level, so a quiet
    # recording must segment the same way a loud one does
    assert VoiceActivity().utterances(loud) == VoiceActivity().utterances(quiet_path)
