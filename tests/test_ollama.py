"""Transcribe locally with ollama, the way ChudGPT does it against Gemini.

gemma4 reports an ``audio`` capability and takes a clip on the ``images`` field,
which is a generic media blob rather than pictures only. So the whole pipeline
runs on the machine: chunk, transcribe each chunk in turn, then hand every
chunk to one more call that stitches them into a single transcript.
"""

import base64
import difflib
import re
import time
from pathlib import Path

import ollama
import pytest

from chudgpt.audio import AudioSpan
from chudgpt.audio._services.chunker import AudioChunker
from chudgpt.audio._utils.transcription import TranscriptionRules

GATSBY_AUDIO = Path(__file__).parent / "assets" / "greatgatsby_01_fitzgerald_64kb.mp3"
GATSBY = Path(__file__).parent / "assets" / "gatsby1.txt"

MODEL = "gemma4"
CHUNK_SECONDS = 15.0
OVERLAP_SECONDS = 5.0
LIMIT_SECONDS = 45.0

# flip these to change what the stitch call does and how it reports
THINK = True
STREAM = True


def plain_words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def match_ratio(heard: list[str], expected: list[str]) -> float:
    return difflib.SequenceMatcher(None, heard, expected).ratio()


def against_book(text: str) -> float:
    heard = plain_words(text)
    return match_ratio(
        heard, plain_words(GATSBY.read_text(encoding="utf-8"))[: len(heard)]
    )


def say(messages: list[dict], *, think: bool, stream: bool) -> str:
    """One ollama turn, printing thinking and answer as they arrive."""
    reply = ollama.chat(model=MODEL, messages=messages, think=think, stream=stream)

    if not stream:
        if reply.message.thinking:
            print(f"\n--- thinking ---\n{reply.message.thinking}")
        print(f"\n--- answer ---\n{reply.message.content}")
        return reply.message.content

    answer: list[str] = []
    channel = None
    for chunk in reply:
        for name, piece in (
            ("thinking", chunk.message.thinking),
            ("answer", chunk.message.content),
        ):
            if not piece:
                continue
            if channel != name:
                print(f"\n--- {name} ---")
                channel = name
            if name == "answer":
                answer.append(piece)
            print(piece, end="", flush=True)
    print()
    return "".join(answer)


def transcribe(chunk_data: bytes) -> str:
    reply = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": TranscriptionRules.DEFAULT_PROMPT,
                "images": [base64.b64encode(chunk_data).decode()],
            }
        ],
    )
    return reply.message.content.strip()


@pytest.mark.skip()
def test_gatsby_transcribed_and_stitched_by_ollama(save_output):
    chunker = AudioChunker(
        chunk_seconds=CHUNK_SECONDS,
        overlap_seconds=OVERLAP_SECONDS,
        limit_seconds=LIMIT_SECONDS,
    )

    print(
        f"\ntranscribing sequentially on {MODEL}: "
        f"{CHUNK_SECONDS:.0f}s chunks, {OVERLAP_SECONDS:.0f}s overlap"
    )
    segments: list[tuple[AudioSpan, str]] = []
    started = time.time()
    for chunk in chunker.stream(GATSBY_AUDIO):
        at = time.time()
        text = transcribe(chunk.data)
        segments.append((chunk.span, text))
        print(
            f"  {chunk.span.name} [{chunk.span.start:6.1f}s -{chunk.span.end:7.1f}s]"
            f"  {time.time() - at:5.1f}s  {len(text.split()):4d} words"
        )
    transcribed = time.time() - started

    print(f"\nstitching {len(segments)} chunks with one more call")
    stitched = say(
        [
            {"role": "system", "content": TranscriptionRules.STITCH_PROMPT},
            {"role": "user", "content": TranscriptionRules.render(segments)},
        ],
        think=THINK,
        stream=STREAM,
    )

    joined = TranscriptionRules.join(segments)
    print(
        f"\n{len(segments)} chunks in {transcribed:.0f}s"
        f" | {len(plain_words(joined))} words joined ({against_book(joined):.1%})"
        f" | {len(plain_words(stitched))} stitched ({against_book(stitched):.1%})"
    )
    save_output(
        {
            "model": MODEL,
            "chunk_seconds": CHUNK_SECONDS,
            "overlap_seconds": OVERLAP_SECONDS,
            "seconds_to_transcribe": round(transcribed, 1),
            "match_joined": against_book(joined),
            "match_stitched": against_book(stitched),
            "segments": [
                {"name": s.name, "start": s.start, "end": s.end, "text": t}
                for s, t in segments
            ],
            "stitched": stitched,
        }
    )

    assert len(plain_words(stitched)) > 200, "the stitch dropped almost everything"
    assert against_book(stitched) > 0.5
