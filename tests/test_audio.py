import asyncio
import difflib
import re
from pathlib import Path

import pytest

from chudgpt import ChudGPT, GeminiModel
from chudgpt.audio import AudioChunker, AudioTranscriber
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
def test_chat_with_audio(chud):
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

    print(response.text)

    assert response.text
    assert response.usage.requests == 1


@pytest.mark.live
def test_first_two_minutes_of_gatsby_matches_the_book(chud: ChudGPT):
    two_minutes = AudioChunker(chunk_seconds=600.0, limit_seconds=1200.0, overlap_seconds=10.0)

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

    print(transcript.responses)

    assert len(heard) > 200, f"only {len(heard)} words for two minutes of narration"
    assert match > 0.85


@pytest.mark.skip()
def test_chat_odyssey(chud):
    response = asyncio.run(
        chud.chat(
            builder=ChudMessageBuilder().prompt(
                Attachment(ODYSSEY).prompt(
                    "Transcribe and translate this clip. It is mostly in greek and some in english"
                )
            ),
            model=GeminiModel.FLASH_LITE_3_5,
        )
    )

    print(response)
