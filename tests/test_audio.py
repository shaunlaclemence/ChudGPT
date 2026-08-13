import asyncio
import difflib
import re
from pathlib import Path

import pytest

from chudgpt import ChudGPT, GeminiModel
from chudgpt.audio import AudioChunker, AudioDiarizer, AudioTranscriber, VoiceActivity
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
        chud.chat(
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
    two_minutes = AudioChunker(chunk_seconds=600.0, overlap_seconds=5.0)

    transcript = asyncio.run(
        AudioTranscriber(chud, two_minutes).transcribe(
            GATSBY_AUDIO, model=GeminiModel.FLASH_LITE_3_5
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
    spans = VoiceActivity().utterances(GATSBY_AUDIO)

    chunker = AudioChunker(
        chunk_seconds=60.0, limit_seconds=1200.0, overlap_seconds=5.0
    )

    diarization = asyncio.run(
        AudioDiarizer(chud, chunker).diarize(
            GATSBY_AUDIO, max_speakers=2, model=GeminiModel.FLASH_LITE_3_5
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
    diarizer = AudioDiarizer(chud)

    first = asyncio.run(
        diarizer.diarize(PIPER_FRENCH, model=GeminiModel.FLASH_LITE_3_5)
    )
    second = asyncio.run(
        diarizer.diarize(PIPER_FRENCH, model=GeminiModel.FLASH_LITE_3_5)
    )

    save_output({"first": first, "second": second})

    assert [u.text for u in first.transcript] == [u.text for u in second.transcript]
    assert [u.timestamp for u in first.transcript] == [
        u.timestamp for u in second.transcript
    ]


@pytest.mark.live()
def test_chat_odyssey(chud, save_output):
    spans = VoiceActivity().utterances(ODYSSEY)

    chunker = AudioChunker(chunk_seconds=60.0, limit_seconds=600.0, overlap_seconds=5.0)

    # diarization = asyncio.run(
    #     AudioDiarizer(chud, chunker).diarize(
    #         ODYSSEY, translate_to="EN", model=GeminiModel.FLASH_LITE_3_5
    #     )
    # )
    print(chunker, "\n\n", spans)

    # save_output(diarization)
