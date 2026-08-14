import glob
import json
from pathlib import Path

import pytest

from chudgpt import ChudGPT, GeminiModel
from chudgpt.audio import ChudEstimate
from chudgpt.audio._schemas.audio_chunk import AudioSpan
from chudgpt.audio._utils.estimate import EstimateRules
from chudgpt.exceptions import ChudGPTBadDataException

ODYSSEY = Path(__file__).parent / "assets" / "odyssey05_01_homer_64kb.mp3"
RECORDED = sorted(glob.glob(str(Path(__file__).parent / "outputs" / "test_stream_odyssey_*.json")))

STREAM_ODYSSEY = {
    "chunk_seconds": 60.0,
    "overlap_seconds": 5.0,
    "limit_seconds": 600.0,
    "translate_to": "EN",
}


def recorded_usage() -> list[dict]:
    runs = []
    for path in RECORDED:
        diarization = json.loads(Path(path).read_text()).get("diarization") or {}
        if diarization.get("usage"):
            runs.append(diarization["usage"])
    return runs


@pytest.fixture(scope="module")
def estimate(chud: ChudGPT) -> ChudEstimate:
    return chud.audio.estimate(
        ODYSSEY, model=GeminiModel.FLASH_LITE_3_5, **STREAM_ODYSSEY
    )


def test_the_plan_matches_the_chunker_stepping():
    chunks = EstimateRules.plan(600.0, 1000, chunk_seconds=60.0, overlap_seconds=5.0)

    assert len(chunks) == 11
    assert chunks[0].start == 0.0
    assert chunks[1].start == pytest.approx(55.0)
    assert chunks[-1].end == pytest.approx(600.0)


def test_a_chunk_carries_the_ranges_that_start_inside_it():
    chunk = AudioSpan(index=0, start=10.0, end=20.0)
    spans = [
        AudioSpan(index=0, start=9.5, end=10.5),
        AudioSpan(index=1, start=10.0, end=11.0),
        AudioSpan(index=2, start=19.9, end=25.0),
        AudioSpan(index=3, start=20.0, end=21.0),
    ]

    assert [s.index for s in EstimateRules.carried(chunk, spans)] == [1, 2]


def test_silence_costs_nothing():
    assert EstimateRules.requests(0) == 0
    assert EstimateRules.reconcile_tokens(0) == (0, 0)


def test_a_single_chunk_skips_the_speaker_merge():
    assert EstimateRules.requests(1) == 1
    assert EstimateRules.reconcile_tokens(1) == (0, 0)


def test_translating_costs_more_output():
    plain = EstimateRules.completion_tokens(100.0, 20, None)
    translated = EstimateRules.completion_tokens(100.0, 20, "EN")

    assert translated[0] > plain[0]
    assert translated[1] > plain[1]


def test_estimate_reports_the_shape_of_the_job(estimate: ChudEstimate):
    assert estimate.chunks == 11
    assert estimate.sent == 11
    assert estimate.requests == estimate.sent + 1
    assert estimate.utterances > 0
    assert estimate.low < estimate.high
    assert estimate.model == GeminiModel.FLASH_LITE_3_5.slug


def test_a_model_without_audio_is_refused(chud: ChudGPT):
    with pytest.raises(ChudGPTBadDataException):
        chud.audio.estimate(ODYSSEY, model=GeminiModel.GEMMA_4_26B)


@pytest.mark.skipif(not recorded_usage(), reason="no recorded diarization runs")
def test_the_estimate_brackets_every_recorded_run(estimate: ChudEstimate):
    for usage in recorded_usage():
        assert usage["requests"] == estimate.requests
        assert estimate.low <= usage["total"] <= estimate.high


@pytest.mark.skipif(not recorded_usage(), reason="no recorded diarization runs")
def test_the_range_stays_useful_rather_than_merely_true(estimate: ChudEstimate):
    assert estimate.high / estimate.low < 2.0
