from __future__ import annotations

from collections.abc import Sequence

from chudgpt.audio._schemas.audio_chunk import AudioSpan
from chudgpt.messages import ChudMessageBuilder


class TranscriptionRules:
    # telling it not to worry about mid sentence boundaries made it stop after
    # the first sentence; demanding the whole clip is what gets a full transcript
    DEFAULT_PROMPT = (
        "Transcribe the ENTIRE audio clip from start to finish, word for word. "
        "Do not stop early, do not summarise, do not skip any speech. "
        "Output only the transcript text."
    )

    STITCH_PROMPT = (
        "Several chunks of text have been provided to you, they are rough "
        "transcriptions of a single audio file. They each overlap by a few "
        "seconds. Stitch the chunks together into one single text, trim away "
        "the repeated segments in the overlapping regions between chunks and "
        "decide which is the truth to keep. Do not modify any other words in "
        "the text. Output only the stitched transcript."
    )

    STITCH_KEY = "stitch"

    @classmethod
    def join(cls, segments: Sequence[tuple[AudioSpan, str]]) -> str:
        return " ".join(text.strip() for _, text in segments if text.strip())

    @classmethod
    def render(cls, segments: Sequence[tuple[AudioSpan, str]]) -> str:
        return "\n\n".join(
            f"[chunk {span.index}, {span.start:.1f}s to {span.end:.1f}s]\n"
            f"{text.strip()}"
            for span, text in segments
            if text.strip()
        )

    @classmethod
    def stitch_builder(
        cls, segments: Sequence[tuple[AudioSpan, str]]
    ) -> ChudMessageBuilder:
        return (
            ChudMessageBuilder().system(cls.STITCH_PROMPT).prompt(cls.render(segments))
        )

    @classmethod
    def needs_stitching(cls, segments: Sequence[tuple[AudioSpan, str]]) -> bool:
        # one chunk has no seam, so a stitch call would only risk the model
        # rewriting a transcript that is already whole
        return len([text for _, text in segments if text.strip()]) > 1
