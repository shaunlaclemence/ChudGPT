from __future__ import annotations

import re

from chudgpt.audio._schemas.audio_chunk import AudioSpan
from chudgpt.audio._schemas.diarization import ChudDiarization
from chudgpt.audio._schemas.progress import ChudPhase, ChudProgress
from chudgpt.audio._utils.language import LanguageRules


class ProgressRules:
    GLOBAL = ChudProgress.GLOBAL

    TEXT = re.compile(r'"text"\s*:')
    LANGUAGE = re.compile(r'"language"\s*:\s*"([^"]{1,24})"')
    TRANSLATION = re.compile(r'"translation"\s*:\s*"([^"]*)"')

    @classmethod
    def scan(cls, partial: str) -> tuple[int, list[str], int]:
        spoken = len(cls.TEXT.findall(partial))
        languages = list(
            dict.fromkeys(
                LanguageRules.code(found) for found in cls.LANGUAGE.findall(partial)
            )
        )
        translated = sum(
            1 for found in cls.TRANSLATION.findall(partial) if found.strip()
        )
        return spoken, languages, translated

    @classmethod
    def phase(cls, spoken: int, translated: int, translate_to: str | None) -> ChudPhase:
        if translate_to and translated:
            return ChudPhase.TRANSLATING
        if spoken:
            return ChudPhase.TRANSCRIBING
        return ChudPhase.SENT

    @classmethod
    def detail(
        cls,
        phase: ChudPhase,
        spoken: int,
        expected: int,
        languages: list[str],
        translated: int,
    ) -> str:
        spoken_of = f"{min(spoken, expected)}/{expected}"
        heard = ", ".join(languages) or "?"
        if phase is ChudPhase.TRANSLATING:
            return f"Translating {heard}, {translated}/{expected} done"
        if phase is ChudPhase.TRANSCRIBING:
            return f"Transcribing {heard}, {spoken_of} utterances"
        return f"Waiting on {expected} utterances"

    @classmethod
    def streaming(
        cls,
        chunk: str,
        span: AudioSpan,
        partial: str,
        expected: int,
        translate_to: str | None,
        at: float,
    ) -> ChudProgress:
        spoken, languages, translated = cls.scan(partial)
        phase = cls.phase(spoken, translated, translate_to)
        return ChudProgress(
            chunk=chunk,
            phase=phase,
            detail=cls.detail(phase, spoken, expected, languages, translated),
            at=at,
            span=span,
            utterances=min(spoken, expected),
            languages=languages,
        )

    @classmethod
    def queued(
        cls, chunk: str, span: AudioSpan, expected: int, at: float
    ) -> ChudProgress:
        return ChudProgress(
            chunk=chunk,
            phase=ChudPhase.QUEUED,
            detail=f"Queued, {expected} utterances to fill",
            at=at,
            span=span,
        )

    @classmethod
    def finished(
        cls, chunk: str, span: AudioSpan, spoken: int, languages: list[str], at: float
    ) -> ChudProgress:
        return ChudProgress(
            chunk=chunk,
            phase=ChudPhase.DONE,
            detail=f"Done, {spoken} utterances, {', '.join(languages) or '?'}",
            at=at,
            span=span,
            utterances=spoken,
            languages=languages,
        )

    @classmethod
    def failed(
        cls, chunk: str, span: AudioSpan, reason: str, at: float
    ) -> ChudProgress:
        return ChudProgress(
            chunk=chunk, phase=ChudPhase.FAILED, detail=reason, at=at, span=span
        )

    @classmethod
    def merging(cls, labels: int, at: float) -> ChudProgress:
        return ChudProgress(
            chunk=cls.GLOBAL,
            phase=ChudPhase.MERGING,
            detail=f"Merging {labels} speaker labels",
            at=at,
        )

    @classmethod
    def assembled(cls, result: ChudDiarization, at: float) -> ChudProgress:
        return ChudProgress(
            chunk=cls.GLOBAL,
            phase=ChudPhase.DONE,
            detail=(
                f"{len(result.transcript)} utterances, "
                f"{len(result.speakers)} speakers, {', '.join(result.languages)}"
            ),
            at=at,
            utterances=len(result.transcript),
            languages=result.languages,
            result=result,
        )
