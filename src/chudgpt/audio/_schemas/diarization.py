from __future__ import annotations

from functools import reduce
from operator import add
from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator

from chudgpt._schemas import ChudResponse, Usage


class ChudUtterance(BaseModel):
    speaker: str = Field(description="Who spoke, as a label.")
    language: str = Field(
        description="ISO 639-1 code of the spoken language, uppercase."
    )
    text: str = Field(description="Exactly what was said in this range.")
    translation: str | None = Field(
        default=None,
        description="Translation of text. Empty when none was asked for or the "
        "utterance is already in the target language.",
    )
    # list, never tuple: tuple emits prefixItems, which providers reject
    timestamp: list[float] = Field(
        min_length=2,
        max_length=2,
        description="Start and end of this utterance, in seconds.",
    )

    @field_validator("translation", mode="before")
    @classmethod
    def blank_to_none(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    @property
    def start(self) -> float:
        return self.timestamp[0]

    @property
    def end(self) -> float:
        return self.timestamp[1]


class ChudSpeaker(BaseModel):
    label: str = Field(description="Speaker label, e.g. 'Speaker A'.")
    description: str = Field(
        description="Short description of the voice: pitch, pace, accent, "
        "gender impression. Used to match this speaker across chunks."
    )


class ChudSpeakerLink(BaseModel):
    chunk_label: str = Field(
        description="A chunk-local label, e.g. 'chunk-000/Speaker A'."
    )
    speaker: str = Field(description="The global speaker it belongs to.")


# the wire shape, so every docstring stays out of the schema the provider is sent.
# timestamp is unconstrained here while the public shape pins it to two floats: it
# is only ever used to align entries against the ranges the model was given, never
# trusted as a time, so a malformed one must not fail the parse
class ChudChunkUtterance(BaseModel):
    speaker: str = Field(description="Who spoke, as a label.")
    language: str = Field(
        description="ISO 639-1 code of the spoken language, uppercase: EN, FR, ES."
    )
    text: str = Field(description="Exactly what was said in this range.")
    translation: str = Field(
        default="",
        description="Translation of text, or an empty string when none is "
        "wanted or the utterance is already in the target language.",
    )
    timestamp: list[float] = Field(
        default_factory=list,
        description="The exact range you were given for this entry.",
    )


class ChudChunkDiarization(BaseModel):
    transcript: list[ChudChunkUtterance]
    speakers: list[ChudSpeaker]


class ChudSpeakerRoster(BaseModel):
    speakers: list[ChudSpeaker]
    links: list[ChudSpeakerLink]


class ChudDiarization(BaseModel):
    languages: list[str]
    translate_to: str | None = None
    transcript: list[ChudUtterance]
    speakers: list[ChudSpeaker]
    speaker_map: dict[str, str]
    responses: dict[str, ChudResponse] = Field(default_factory=dict)

    @computed_field
    @property
    def text(self) -> str:
        return " ".join(u.text.strip() for u in self.transcript if u.text.strip())

    @computed_field
    @property
    def usage(self) -> Usage:
        spent = [response.usage for response in self.responses.values()]
        return reduce(add, spent) if spent else Usage(requests=0)
