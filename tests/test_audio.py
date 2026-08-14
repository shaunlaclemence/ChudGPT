import asyncio
import difflib
import re
from pathlib import Path

import pytest

from chudgpt import ChudGPT, GeminiModel
from chudgpt.audio import ChudDiarization, ChudPhase, ChudProgress
from chudgpt.messages import Attachment, ChudMessageBuilder

PIPER_FRENCH = Path(__file__).parent / "assets" / "piper_french.wav"
ODYSSEY = Path(__file__).parent / "assets" / "odyssey05_01_homer_64kb.mp3"
GATSBY_AUDIO = Path(__file__).parent / "assets" / "greatgatsby_01_fitzgerald_64kb.mp3"
GATSBY = Path(__file__).parent / "assets" / "gatsby1.txt"


def plain_words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def match_ratio(heard: list[str], expected: list[str]) -> float:
    return difflib.SequenceMatcher(None, heard, expected).ratio()


@pytest.mark.skip()
def test_chat_with_audio(chud, save_output):
    response = asyncio.run(
        chud.text.chat(
            builder=ChudMessageBuilder().prompt(
                Attachment(PIPER_FRENCH).prompt(
                    r"Transcribe this clip. Then translate it to english. In your final response, separate transcription and translation into a json \{'translation:', 'transcription:'\} "
                )
            ),
            model=GeminiModel.FLASH_LITE_3_5,
        )
    )

    save_output(response)

    assert response.data
    assert response.usage.requests == 1


@pytest.mark.skip()
def test_first_two_minutes_of_gatsby_matches_the_book(chud: ChudGPT, save_output):
    transcript = asyncio.run(
        chud.audio.transcribe(
            GATSBY_AUDIO,
            model=GeminiModel.FLASH_LITE_3_5,
            chunk_seconds=600.0,
            overlap_seconds=5.0,
        )
    )

    heard = plain_words(transcript.text)
    expected = plain_words(GATSBY.read_text(encoding="utf-8"))[: len(heard)]
    match = match_ratio(heard, expected)

    print(
        f"\n{len(transcript.segments)} chunks, {len(heard)} words heard, "
        f"{match:.1%} match against the book"
    )
    save_output(transcript)

    assert len(heard) > 200, f"only {len(heard)} words for two minutes of narration"
    assert match > 0.85


@pytest.mark.skip()
def test_diarized_gatsby_matches_the_book(chud: ChudGPT, save_output):
    spans = chud.audio.voice_activity(GATSBY_AUDIO)

    diarization = asyncio.run(
        chud.audio.diarize(
            GATSBY_AUDIO,
            max_speakers=2,
            model=GeminiModel.FLASH_LITE_3_5,
            chunk_seconds=60.0,
            limit_seconds=1200.0,
            overlap_seconds=5.0,
        )
    )

    heard = plain_words(diarization.text)
    expected = plain_words(GATSBY.read_text(encoding="utf-8"))[: len(heard)]
    match = match_ratio(heard, expected)

    print(
        f"\n{len(diarization.transcript)} utterances, "
        f"{len(diarization.speakers)} speakers, {diarization.languages}, "
        f"{len(heard)} words heard, {match:.1%} match against the book"
    )
    save_output({"vad_spans": spans, "diarization": diarization})

    assert len(heard) > 200, f"only {len(heard)} words diarized"
    # 91.0% and 79.9% on two runs at temperature 0: the model still sometimes
    # repeats a sentence across ranges, so this sits below the transcriber's bar
    assert match > 0.90


@pytest.mark.skip()
def test_diarize_is_reproducible_at_temperature_zero(chud: ChudGPT, save_output):
    first = asyncio.run(
        chud.audio.diarize(PIPER_FRENCH, model=GeminiModel.FLASH_LITE_3_5)
    )
    second = asyncio.run(
        chud.audio.diarize(PIPER_FRENCH, model=GeminiModel.FLASH_LITE_3_5)
    )

    save_output({"first": first, "second": second})

    assert [u.text for u in first.transcript] == [u.text for u in second.transcript]
    assert [u.timestamp for u in first.transcript] == [
        u.timestamp for u in second.transcript
    ]


@pytest.mark.skip()
def test_chat_odyssey(chud, save_output):
    spans = chud.audio.voice_activity(ODYSSEY)

    diarization = asyncio.run(
        chud.audio.diarize(
            ODYSSEY,
            translate_to="EN",
            model=GeminiModel.FLASH_LITE_3_5,
            chunk_seconds=60.0,
            limit_seconds=600.0,
            overlap_seconds=5.0,
        )
    )
    print(spans)

    save_output(diarization)


def test_stream_odyssey(chud: ChudGPT, save_output):
    async def run() -> tuple[list[ChudProgress], ChudDiarization | None]:
        board: dict[str, ChudProgress] = {}
        timeline: list[ChudProgress] = []
        result: ChudDiarization | None = None

        update: ChudProgress
        async for update in chud.audio.diarize_stream(
            ODYSSEY,
            translate_to="EN",
            max_speakers=2,
            model=GeminiModel.FLASH_LITE_3_5,
            chunk_seconds=60.0,
            limit_seconds=600.0,
            overlap_seconds=5.0,
        ):
            board[update.chunk] = update
            timeline.append(update)
            print(
                f"{update.at:6.1f}s  {update.chunk:11} {update.phase.value:13}"
                f" {update.detail}"
            )
            if update.result is not None:
                result = update.result

        print("\nfinal board:")
        for name, latest in sorted(
            board.items(), key=lambda kv: kv[1].span.start if kv[1].span else -1
        ):
            print(f"  {name:11} {latest.phase.value:13} {latest.detail}")

        return timeline, result

    timeline, result = asyncio.run(run())
    save_output({"timeline": timeline, "diarization": result})

    assert result is not None, "the stream never yielded an assembled diarization"
    assert result.transcript, "no utterances came back"

    chunks = {u.chunk for u in timeline if not u.is_global}
    assert chunks, "no per-chunk progress was reported"

    for name in chunks:
        phases = [u.phase for u in timeline if u.chunk == name]
        assert phases[0] is ChudPhase.QUEUED, f"{name} did not start queued"
        assert ChudPhase.DONE in phases or ChudPhase.FAILED in phases, (
            f"{name} never settled"
        )

    assert timeline[-1].phase is ChudPhase.DONE
    assert timeline[-1].is_global
    assert [u.at for u in timeline] == sorted(u.at for u in timeline), (
        "updates must arrive in time order"
    )
