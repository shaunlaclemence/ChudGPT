"""The decisions in DiarizationRules, none of which need a provider.

Everything here is the half of diarization that must be right before a request
is worth making: the schema the provider is sent, and what is done with a reply
that does not come back in the shape that was asked for.
"""

import json

import pytest

from chudgpt.audio import AudioSpan, ChudSpeaker, ChudUtterance
from chudgpt.audio._schemas.diarization import (
    ChudChunkDiarization,
    ChudChunkUtterance,
    ChudSpeakerLink,
    ChudSpeakerRoster,
)
from chudgpt.audio._utils.diarization import DiarizationRules
from chudgpt.audio._utils.language import LanguageRules

CHUNK = AudioSpan(index=0, start=0.0, end=30.0)
SPANS = [
    AudioSpan(index=0, start=1.0, end=2.0),
    AudioSpan(index=1, start=5.0, end=6.5),
    AudioSpan(index=2, start=9.0, end=11.0),
]


def entry(
    speaker="Speaker A", language="french", text="bonjour", translation="", at=None
):
    return ChudChunkUtterance(
        speaker=speaker,
        language=language,
        text=text,
        translation=translation,
        timestamp=list(at) if at else [],
    )


def reply(*entries, speakers=()):
    return ChudChunkDiarization(
        transcript=list(entries),
        speakers=[ChudSpeaker(label=label, description=d) for label, d in speakers],
    )


def utterance(speaker="Speaker A", language="english", start=0.0, end=1.0):
    return ChudUtterance(
        speaker=speaker, language=language, text="x", timestamp=[start, end]
    )


### provider schema, R3 and R7 ###


@pytest.mark.parametrize("model", [ChudChunkDiarization, ChudSpeakerRoster])
def test_the_wire_schema_carries_no_construct_providers_reject(model):
    text = json.dumps(DiarizationRules.provider_schema(model))

    for hazard in ("$ref", "$defs", "anyOf", "prefixItems", "null"):
        assert hazard not in text, f"{hazard} reached the provider schema"


def test_every_property_is_required_on_the_wire():
    schema = DiarizationRules.provider_schema(ChudChunkDiarization)
    item = schema["properties"]["transcript"]["items"]

    assert set(item["required"]) == set(item["properties"])
    assert "translation" in item["required"], "the empty-string convention needs it"


def test_the_public_timestamp_stays_a_two_item_array():
    timestamp = ChudUtterance.model_json_schema()["properties"]["timestamp"]

    assert timestamp["type"] == "array"
    assert timestamp["items"] == {"type": "number"}
    assert (timestamp["minItems"], timestamp["maxItems"]) == (2, 2)
    assert "prefixItems" not in timestamp


### alignment, R18 ###


def test_every_range_filled_pairs_in_order():
    paired = DiarizationRules.align(CHUNK, SPANS, [entry(), entry(), entry()])

    assert [span for span, _ in paired] == SPANS


def test_fewer_entries_than_ranges_fills_what_it_can():
    paired = DiarizationRules.align(CHUNK, SPANS, [entry(at=(1.0, 2.0))])

    assert [span for span, _ in paired] == [SPANS[0]]


def test_extra_entries_are_dropped_rather_than_crashing():
    paired = DiarizationRules.align(CHUNK, SPANS, [entry() for _ in range(6)])

    assert len(paired) == len(SPANS)


def test_a_skipped_range_does_not_shift_the_rest():
    # the model answered ranges 1 and 3, echoing their times: without the echo
    # this would put range 3's text on range 2
    paired = DiarizationRules.align(
        CHUNK,
        SPANS,
        [entry(text="first", at=(1.0, 2.0)), entry(text="third", at=(9.0, 11.0))],
    )

    assert [(span.start, e.text) for span, e in paired] == [
        (1.0, "first"),
        (9.0, "third"),
    ]


def test_entries_arriving_out_of_order_are_sorted_back_onto_their_ranges():
    paired = DiarizationRules.align(
        CHUNK,
        SPANS,
        [entry(text="third", at=(9.0, 11.0)), entry(text="first", at=(1.0, 2.0))],
    )

    assert [e.text for _, e in paired] == ["first", "third"]


def test_timestamps_come_from_the_spans_not_from_the_model():
    lying = entry(at=(999.0, 1000.0))

    built = DiarizationRules.utterances(CHUNK, SPANS[:1], reply(lying))

    assert built[0].timestamp == [SPANS[0].start, SPANS[0].end]


def test_chunk_local_timestamps_are_offset_by_the_chunk_start():
    chunk = AudioSpan(index=1, start=30.0, end=60.0)
    spans = [AudioSpan(index=0, start=31.5, end=33.0)]

    built = DiarizationRules.utterances(chunk, spans, reply(entry(at=(1.5, 3.0))))

    assert built[0].timestamp == [31.5, 33.0]


### translation, R8 ###


def test_translation_is_dropped_when_none_was_asked_for():
    built = DiarizationRules.utterances(
        CHUNK, SPANS[:1], reply(entry(translation="hi"))
    )

    assert built[0].translation is None


def test_a_redundant_translation_is_dropped():
    same = entry(language="english", translation="hello")

    built = DiarizationRules.utterances(CHUNK, SPANS[:1], reply(same), "english")

    assert built[0].translation is None, "echoing text back doubles output for nothing"


def test_a_real_translation_survives():
    built = DiarizationRules.utterances(
        CHUNK, SPANS[:1], reply(entry(translation="hello")), "english"
    )

    assert built[0].translation == "hello"


def test_the_empty_string_convention_becomes_none():
    assert (
        ChudUtterance(
            speaker="A",
            language="french",
            text="x",
            translation="",
            timestamp=[0.0, 1.0],
        ).translation
        is None
    )


### assembly, R13, R15, R19 ###


def test_speaker_labels_are_namespaced_per_chunk():
    built = DiarizationRules.utterances(CHUNK, SPANS[:1], reply(entry()))

    assert built[0].speaker == "chunk-000/Speaker A"


def test_dedup_drops_a_repeated_range_and_orders_the_rest():
    duplicated = [
        utterance(start=5.0, end=6.0),
        utterance(start=1.0, end=2.0),
        utterance(start=5.0, end=6.0),
    ]

    deduped = DiarizationRules.dedup(duplicated)

    assert [(u.start, u.end) for u in deduped] == [(1.0, 2.0), (5.0, 6.0)]


def test_languages_are_the_deduplicated_union():
    mixed = [
        utterance(language="french"),
        utterance(language="english"),
        utterance(language="french"),
    ]

    assert DiarizationRules.languages(mixed) == ["FR", "EN"]


def test_the_speaker_map_uses_the_links_the_model_chose():
    speakers = [
        ChudSpeaker(label="chunk-000/Speaker A", description="low"),
        ChudSpeaker(label="chunk-001/Speaker A", description="low"),
    ]
    roster = ChudSpeakerRoster(
        speakers=[ChudSpeaker(label="Speaker A", description="low")],
        links=[
            ChudSpeakerLink(chunk_label="chunk-000/Speaker A", speaker="Speaker A"),
            ChudSpeakerLink(chunk_label="chunk-001/Speaker A", speaker="Speaker A"),
        ],
    )

    mapping = DiarizationRules.speaker_map(speakers, roster)

    assert mapping == {
        "chunk-000/Speaker A": "Speaker A",
        "chunk-001/Speaker A": "Speaker A",
    }


def test_a_label_the_reconciler_ignored_keeps_its_own_identity():
    speakers = [ChudSpeaker(label="chunk-000/Speaker B", description="high")]
    roster = ChudSpeakerRoster(speakers=[], links=[])

    mapping = DiarizationRules.speaker_map(speakers, roster)

    assert mapping == {"chunk-000/Speaker B": "chunk-000/Speaker B"}


def test_applying_the_map_rewrites_the_transcript():
    mapped = DiarizationRules.apply_map(
        [utterance(speaker="chunk-001/Speaker B")],
        {"chunk-001/Speaker B": "Speaker A"},
    )

    assert mapped[0].speaker == "Speaker A"


### guards ###


def test_an_utterance_seen_by_two_overlapping_chunks_survives_once():
    # the vad span is absolute, so both chunks report the identical range and
    # dedup is what makes overlap_seconds safe
    from_first = utterance(speaker="chunk-000/Speaker A", start=57.0, end=59.0)
    from_second = utterance(speaker="chunk-001/Speaker A", start=57.0, end=59.0)

    deduped = DiarizationRules.dedup([from_first, from_second])

    assert len(deduped) == 1
    assert deduped[0].speaker == "chunk-000/Speaker A", "the earlier chunk wins"


@pytest.mark.parametrize(
    ("said", "code"),
    [
        ("en", "EN"),
        ("eng", "EN"),
        ("English", "EN"),
        ("ENGLISH", "EN"),
        ("en-US", "EN"),
        ("French", "FR"),
        ("fra", "FR"),
        ("français", "FR"),
        ("chinese pinyin", "ZH"),
        ("Spanish (Latin America)", "ES"),
        ("", "UND"),
    ],
)
def test_language_is_standardised_to_one_code(said, code):
    assert LanguageRules.code(said) == code


def test_the_three_spellings_of_english_collapse_to_one_language():
    mixed = [
        utterance(language="en"),
        utterance(language="eng"),
        utterance(language="English"),
    ]

    assert DiarizationRules.languages(mixed) == ["EN"]


def test_utterances_come_back_with_a_code_not_a_name():
    built = DiarizationRules.utterances(
        CHUNK, SPANS[:1], reply(entry(language="English"))
    )

    assert built[0].language == "EN"


def test_a_translation_into_a_language_named_differently_is_still_redundant():
    already = entry(language="en", translation="hello")

    built = DiarizationRules.utterances(CHUNK, SPANS[:1], reply(already), "English")

    assert built[0].translation is None, "en and English are the same language"


def test_the_speaker_ceiling_reaches_both_prompts():
    chunk_prompt = DiarizationRules.prompt(SPANS, CHUNK, None, max_speakers=1)
    roster = DiarizationRules.roster_prompt(
        [ChudSpeaker(label="chunk-000/Speaker A", description="low")],
        [],
        max_speakers=1,
    )

    assert "at most 1 distinct speaker" in chunk_prompt
    assert "at most 1 distinct speaker" in roster


def test_the_reconciler_is_told_to_merge_and_shown_what_was_said():
    speakers = [ChudSpeaker(label="chunk-000/Speaker A", description="male, low")]
    said = [utterance(speaker="chunk-000/Speaker A")]

    text = DiarizationRules.roster_prompt(speakers, said)

    assert "Merge aggressively" in text
    assert "1 utterances" in text, "counts are evidence for the merge"
    assert "ONE speaker" in text, "a narrator doing voices is not many speakers"


def test_the_prompt_states_the_count_and_the_chunk_local_ranges():
    text = DiarizationRules.prompt(SPANS, AudioSpan(index=1, start=1.0, end=31.0))

    assert "3 utterances" in text
    assert "1. [0.00, 1.00]" in text
    assert "empty string for every translation" in text


def test_asking_for_a_translation_changes_the_prompt():
    text = DiarizationRules.prompt(SPANS, CHUNK, "english")

    assert "Translate each utterance into english" in text
