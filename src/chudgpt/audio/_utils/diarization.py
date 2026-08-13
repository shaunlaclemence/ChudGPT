from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from chudgpt.audio._schemas.audio_chunk import AudioSpan
from chudgpt.audio._schemas.diarization import (
    ChudChunkDiarization,
    ChudChunkUtterance,
    ChudSpeaker,
    ChudSpeakerRoster,
    ChudUtterance,
)
from chudgpt.audio._utils.language import LanguageRules


class DiarizationRules:
    DROPPED_KEYS = ("title", "default", "$defs")
    LABEL_SEPARATOR = "/"
    SAMPLE_CHARS = 140
    RECONCILE_KEY = "reconcile"

    @classmethod
    def provider_schema(cls, model: type[BaseModel]) -> dict[str, Any]:
        """A pydantic schema rewritten into the subset providers reliably accept.

        Inlines every $ref, collapses ``str | None`` unions to a plain string,
        and marks every property required. anyOf-with-null and $ref are the same
        class of hazard as prefixItems: commonly ignored or rejected.
        """
        schema = model.model_json_schema()
        return cls.__flatten(schema, schema.get("$defs", {}))

    @classmethod
    def __flatten(cls, node: Any, defs: dict[str, Any]) -> Any:
        if isinstance(node, list):
            return [cls.__flatten(item, defs) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return cls.__flatten(defs[node["$ref"].rsplit("/", 1)[-1]], defs)
        if "anyOf" in node:
            real = [b for b in node["anyOf"] if b.get("type") != "null"]
            merged = {k: v for k, v in node.items() if k != "anyOf"} | real[0]
            return cls.__flatten(merged, defs)

        flat = {
            key: cls.__flatten(value, defs)
            for key, value in node.items()
            if key not in cls.DROPPED_KEYS
        }
        if flat.get("type") == "object" and "properties" in flat:
            flat["required"] = list(flat["properties"])
        return flat

    @classmethod
    def within(cls, spans: Sequence[AudioSpan], chunk: AudioSpan) -> list[AudioSpan]:
        return [s for s in spans if chunk.start <= s.start < chunk.end]

    @classmethod
    def prompt(
        cls,
        spans: Sequence[AudioSpan],
        chunk: AudioSpan,
        translate_to: str | None = None,
        max_speakers: int | None = None,
    ) -> str:
        ranges = "\n".join(
            f"{index + 1}. [{s.start - chunk.start:.2f}, "
            f"{min(s.end, chunk.end) - chunk.start:.2f}]"
            for index, s in enumerate(spans)
        )
        lines = [
            (
                f"This clip contains {len(spans)} utterances at exactly these "
                "ranges, in seconds from the start of THIS clip:"
            ),
            ranges,
            (
                f"Return exactly {len(spans)} transcript entries, one per numbered "
                "range, in the same order. For each, give the speaker, the spoken "
                "language, and the words spoken in that range."
            ),
            (
                "Use the exact range you were given as the timestamp. Never invent "
                "or adjust a time."
            ),
            (
                "Give language as the ISO 639-1 two-letter code in UPPERCASE, for "
                "example EN, FR, ES, DE, ZH. Never a language name, never three "
                "letters, never lowercase."
            ),
            (
                "Label speakers 'Speaker A', 'Speaker B' and so on, in the order "
                "they first speak. One person reading dialogue aloud for several "
                "characters is ONE speaker, not one per character."
            ),
            (
                "Also return one roster entry per speaker, describing the voice in "
                "exactly this form and nothing else: "
                "<gender>, <low|medium|high> pitch, <slow|measured|fast> pace, "
                "<accent>. Example: 'male, low pitch, measured pace, American'."
            ),
        ]
        if max_speakers:
            lines.append(
                f"This recording has at most {max_speakers} distinct "
                f"{'speaker' if max_speakers == 1 else 'speakers'}. Never report more."
            )
        if translate_to:
            lines.append(
                f"Translate each utterance into {translate_to}. When an utterance "
                f"is already in {translate_to}, return an empty string for "
                "translation rather than copying the text."
            )
        else:
            lines.append("Return an empty string for every translation.")
        return "\n".join(lines)

    @classmethod
    def utterances(
        cls,
        chunk: AudioSpan,
        spans: Sequence[AudioSpan],
        reply: ChudChunkDiarization,
        translate_to: str | None = None,
    ) -> list[ChudUtterance]:
        return [
            ChudUtterance(
                speaker=cls.namespace(chunk, entry.speaker),
                language=LanguageRules.code(entry.language),
                text=entry.text,
                translation=cls.translation(entry, translate_to),
                timestamp=[span.start, span.end],
            )
            for span, entry in cls.align(chunk, spans, reply.transcript)
        ]

    @classmethod
    def align(
        cls,
        chunk: AudioSpan,
        spans: Sequence[AudioSpan],
        entries: Sequence[ChudChunkUtterance],
    ) -> list[tuple[AudioSpan, ChudChunkUtterance]]:
        """Pair each reply entry with the range it belongs to.

        The echoed range decides, so one skipped or invented entry cannot shift
        every following utterance onto the wrong timestamp. Position is the
        fallback when nothing was echoed, and entries past the last free range
        are dropped.
        """
        taken: set[int] = set()
        paired: list[tuple[int, ChudChunkUtterance]] = []
        for position, entry in enumerate(entries):
            index = cls.__match(chunk, spans, entry, position, taken)
            if index is None:
                continue
            taken.add(index)
            paired.append((index, entry))
        return [(spans[index], entry) for index, entry in sorted(paired)]

    @classmethod
    def __match(
        cls,
        chunk: AudioSpan,
        spans: Sequence[AudioSpan],
        entry: ChudChunkUtterance,
        position: int,
        taken: set[int],
    ) -> int | None:
        free = [i for i in range(len(spans)) if i not in taken]
        if not free:
            return None
        if entry.timestamp:
            echoed = entry.timestamp[0]
            return min(free, key=lambda i: abs(spans[i].start - chunk.start - echoed))
        return position if position in free else None

    @classmethod
    def namespace(cls, chunk: AudioSpan, label: str) -> str:
        return f"{chunk.name}{cls.LABEL_SEPARATOR}{label.strip()}"

    @classmethod
    def translation(
        cls, entry: ChudChunkUtterance, translate_to: str | None
    ) -> str | None:
        if not translate_to or not entry.translation.strip():
            return None
        if LanguageRules.same(entry.language, translate_to):
            return None
        return entry.translation

    @classmethod
    def dedup(cls, utterances: Sequence[ChudUtterance]) -> list[ChudUtterance]:
        seen: set[tuple[float, float]] = set()
        ordered: list[ChudUtterance] = []
        for utterance in sorted(utterances, key=lambda u: (u.start, u.end)):
            key = (round(utterance.start, 3), round(utterance.end, 3))
            if key not in seen:
                seen.add(key)
                ordered.append(utterance)
        return ordered

    @classmethod
    def languages(cls, utterances: Sequence[ChudUtterance]) -> list[str]:
        return LanguageRules.union([u.language for u in utterances])

    @classmethod
    def samples(cls, utterances: Sequence[ChudUtterance]) -> dict[str, tuple[str, int]]:
        spoken: dict[str, list[str]] = {}
        for utterance in utterances:
            spoken.setdefault(utterance.speaker, []).append(utterance.text)
        return {
            label: (texts[0][: cls.SAMPLE_CHARS], len(texts))
            for label, texts in spoken.items()
        }

    @classmethod
    def roster_prompt(
        cls,
        speakers: Sequence[ChudSpeaker],
        utterances: Sequence[ChudUtterance] = (),
        max_speakers: int | None = None,
    ) -> str:
        samples = cls.samples(utterances)
        listing = "\n".join(cls.__entry(speaker, samples) for speaker in speakers)
        lines = [
            (
                "Every label below came from a different chunk of ONE recording, "
                "and each chunk described the voices independently, so the SAME "
                "person is written up in different words from chunk to chunk:"
            ),
            listing,
            (
                "Merge aggressively. Treat two labels as the same person unless "
                "the descriptions plainly conflict, such as a different gender or "
                "a different accent. Different wording for the same voice is not a "
                "conflict, and neither is a different pitch or pace word."
            ),
            (
                "One narrator reading dialogue aloud for several characters is ONE "
                "speaker, however many voices they put on."
            ),
        ]
        if max_speakers:
            lines.append(
                f"This recording has at most {max_speakers} distinct "
                f"{'speaker' if max_speakers == 1 else 'speakers'}. Returning more "
                "is wrong."
            )
        lines.append(
            "Return a global roster using 'Speaker A', 'Speaker B' and so on, and "
            "one link for EVERY label listed above, mapping it to its global speaker."
        )
        return "\n".join(lines)

    @classmethod
    def __entry(cls, speaker: ChudSpeaker, samples: dict[str, tuple[str, int]]) -> str:
        said, count = samples.get(speaker.label, ("", 0))
        heard = f'\n    says: "{said}"' if said else ""
        return f"- {speaker.label} ({count} utterances): {speaker.description}{heard}"

    @classmethod
    def speaker_map(
        cls, speakers: Sequence[ChudSpeaker], roster: ChudSpeakerRoster | None
    ) -> dict[str, str]:
        links = (
            {link.chunk_label: link.speaker for link in roster.links} if roster else {}
        )
        return {s.label: links.get(s.label, s.label) for s in speakers}

    @classmethod
    def apply_map(
        cls, utterances: Sequence[ChudUtterance], mapping: dict[str, str]
    ) -> list[ChudUtterance]:
        return [
            u.model_copy(update={"speaker": mapping.get(u.speaker, u.speaker)})
            for u in utterances
        ]

    @classmethod
    def global_speakers(
        cls,
        speakers: Sequence[ChudSpeaker],
        roster: ChudSpeakerRoster | None,
        mapping: dict[str, str],
    ) -> list[ChudSpeaker]:
        if roster and roster.speakers:
            return list(roster.speakers)
        described = {mapping.get(s.label, s.label): s.description for s in speakers}
        return [ChudSpeaker(label=k, description=v) for k, v in described.items()]
