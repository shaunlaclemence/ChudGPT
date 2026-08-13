# Diarized, timestamped, multi-language transcription

## Context

`AudioTranscriber.transcribe()` returns flat text. The goal is structured output identifying who
spoke, in what language, and when:

```json
{ "languages": ["english", "french"],
  "translate_to": "english",
  "transcript": [{ "speaker": "Speaker C", "language": "french",
                   "text": "...", "translation": "...", "timestamp": [40.0, 45.9] }] }
```

`translation` is populated only when a segment's language differs from `translate_to`, and is `null`
otherwise. Translating in the same request as transcription is the right call: the model already holds
the audio and the detected language, so a second pass would re-pay the audio token cost for nothing.

Decisions taken: a new `AudioDiarizer` service (leaving `transcribe()` alone), parallel chunk fan-out
plus a reconciliation pass to merge per-chunk speaker labels, and **timestamps must be deterministic.**

## Answering the determinism question

"Deterministic" splits into two separate problems, and only one of them is solvable by a flag.

**1. Reproducible output** (same input, same result): pass `temperature=0`. This makes the model's
text and speaker assignment stable across runs. It does **not** make timestamps correct.

**2. Correct timestamps: do not let the model produce them at all.** LLMs have no clock; they
estimate offsets and the numbers are plausible-looking guesses. The fix is to compute boundaries from
the waveform and let the model only fill in content.

A plain RMS voice-activity pass was verified against the existing assets, using the `numpy` that
`soundfile` already pulls in (no new dependency):

| Asset | Duration | Utterances found | Identical on rerun |
| --- | --- | --- | --- |
| `piper_french.wav` | 13.4s | 2 | **yes** |
| `greatgatsby_01_fitzgerald_64kb.mp3` | first 120s | 31 (1.4s to 4.3s each) | **yes** |

Boundaries are a pure function of the samples, so they are bit-identical every run, independent of
model, temperature, or provider. That is real determinism.

**So the flow inverts.** Instead of asking the model "when did each speaker talk", we tell it the
exact time ranges and ask it to fill each one in:

```
Audio + "This clip contains 7 utterances at exactly these ranges:
         1. [0.84, 2.61]   2. [3.30, 4.74]   ...
         For each numbered range give speaker, language, and text."
```

Timestamps come from the signal. The model never invents a number. Its remaining job (transcribe,
identify speaker, identify language) is what it is actually good at.

Trade-off to accept: VAD detects *when someone is speaking*, not *who*. Two speakers with no pause
between them land in one range. That is a real limitation, and honest, and the model can still report
a speaker change inside a range if we let it split.

## Requirements

### Core blockers, must be fixed first

**R1. `parallel_chat` cannot request structured output.** [_main.py](../_main.py) `parallel_chat`
hardcodes `self.__require().chat(messages=..., model=...)` with **no `**request_kwargs`**. Both
`response_format` and `temperature=0` are therefore unreachable on the fan-out path, which is the path
this feature uses. `RotorService.chat` already accepts and forwards `**request_kwargs` untouched
([rotor.py](../_services/rotor.py)), so only `parallel_chat` needs the passthrough added.

**R2. The eager-parse guard in `chat_json` is a silent no-op.** [_main.py](../_main.py) reads
`_ = response.json`, but `ChudResponse` has no `json` property; it is named `parsed_json`
(deliberately, since `BaseModel` already has a deprecated `json()` method, see
[chud_response.py](../_schemas/chud_response.py)). So `.json` binds pydantic's deprecated method and
discards it. Malformed JSON does **not** fail at the `chat_json` boundary as the comment claims, and
nothing is cached. One-word fix, and it matters here because diarization depends entirely on
well-formed JSON.

### Schema requirements

**R3. `timestamp` must be `list[float]` with `min_length=2, max_length=2`, never `tuple[float, float]`.**
Verified locally: the tuple form makes pydantic emit `prefixItems` (JSON Schema 2020-12), which
provider structured-output validators commonly ignore or reject. `list[float]` emits the portable
`items` + `minItems`/`maxItems` form and produces the exact `[a, b]` wire shape requested.

**R4. The existing `Language` enum cannot be reused.** [structured.py](../_schemas/structured.py)
defines `Language` as *programming* languages (`python`, `swift`, `C++`, and so on). Spoken language
needs its own type. Recommend a plain `str` field rather than an enum, since the goal is "any audio"
and a closed set would reject languages the model can legitimately identify.

**R5. Diarization schemas must live in `audio/_schemas/`, not `chudgpt/_schemas/`.** The latter is a
barrel eagerly imported by [__init__.py](../__init__.py), so anything placed there loads on plain
`import chudgpt` and would undo the lean-install work. Also note `scripts/generate_schemas_init.py` is
stale, pointing at a pre-rename `schemas/` path, and its hook guard tests `'schemas' in parts` which
never matches `_schemas`, so that barrel is hand-maintained and the generator will not pick anything
up. Leave both alone.

**R6. Reconciliation needs voice descriptions in the per-chunk schema.** Merging "Speaker A" across
chunks is impossible from labels alone. Each chunk response must also return a roster entry per
speaker with a short description (pitch, pace, accent, gender impression) for the merge pass to have
any evidence at all.

**R7. Nullable fields must be flattened before the schema reaches the provider.** Verified locally:
`translation: str | None` makes pydantic emit `{"anyOf": [{"type": "string"}, {"type": "null"}]}`.
`anyOf` with a `null` branch is the same class of hazard as `prefixItems` in R3 and is commonly
ignored or rejected by provider structured-output validators. Both `translation` and `translate_to`
hit this.

The codebase already has the pattern for exactly this problem: `ChudSchema.pin()` in
[structured.py](../_schemas/structured.py) rewrites a pydantic schema to work around a Gemini
limitation, with the reason recorded inline ("gemini silently ignores const"). Extend that approach
rather than inventing a new one:

- **On the wire**, send plain `{"type": "string"}` and instruct the model to return `""` when no
  translation is needed. Provider-safe with no `anyOf` anywhere.
- **In Python**, keep the field as `str | None` and normalise `""` to `None` with a pydantic
  validator, so callers and any JSON they dump see `"translation": null` as specified.

That keeps one source of truth for the shape while giving the provider a schema it will definitely
accept.

**R8. Translation must be conditional, not unconditional.** When a segment's language already equals
`translate_to`, `translation` must be empty rather than a copy of `text`. Echoing the text back would
roughly double output tokens for no information, and output is the binding constraint. When
`translate_to` is `None`, translation is skipped entirely.

**R9. Translation raises the output token cost, so chunk size may need to drop.** A 300s chunk is
about 700 words of speech, which as diarized JSON is roughly 2.5k output tokens; if most segments also
need translating that pushes toward 5k. Still inside the cap measured earlier (866 completion tokens
for plain 300s text, against a limit that truncated a 40-minute single request), but the margin
shrinks. Verify against a real translated chunk before settling on `chunk_seconds` and be ready to
lower it when `translate_to` is set.

### Determinism requirements

- **R10.** Timestamps derived from waveform VAD, never from the model (verified deterministic above).
- **R11.** `temperature=0` on every diarization call, for reproducible text, speaker assignment, and
  translation. Depends on R1.
- **R12.** VAD runs over the whole file, so boundaries are already absolute. Where a chunk-local value
  does appear it must be offset by `AudioSpan.start`, which already exists.

### Behavioural requirements

- **R13. Speaker labels are chunk-local until reconciled.** Namespace them (`chunk-000/Speaker A`)
  through the pipeline so they can never be silently mistaken for global identity, and keep the raw
  per-chunk labels on the result after merging.
- **R14. Reconciliation is best-effort and must be documented as such.** It is one LLM call comparing
  text descriptions with no acoustic evidence; it will sometimes be wrong. Expose the mapping it chose
  so a caller can audit or override it.
- **R15. Overlap must be de-duplicated.** `AudioChunker` supports `overlap_seconds`, and overlapping
  chunks will report the same utterance twice. Either force `overlap=0` for diarization or drop
  utterances whose time range is already covered. Recommend forcing 0 and documenting why.
- **R16. Model must accept `Modality.AUDIO`.** 7 of the 9 catalog models do; both Gemma members do
  not. `AudioFanOutRules.guard_model` already enforces this and is reusable as-is. Related trap
  already handled: `GeminiModel.cheapest()` returns a Gemma model, so the rotor's implicit default is
  audio-incapable and `model=` must always be passed.
- **R17. Free-tier budget.** The audio-capable models cap at `rpm 15 / rpd 500`. Diarization costs
  one request per chunk plus one reconciliation call, so a long file plus iteration burns rpd quickly.
  `AudioFanOutRules.batch_size` already clamps to `model.rpm`.
- **R18. Utterance-count mismatch must be handled.** Given N ranges the model may return fewer, more,
  or misaligned entries. Rules must reconcile by index and drop or flag extras rather than crash.
- **R19. `languages` is a union, `translate_to` is not.** The top-level `languages` list is the union
  of per-segment languages across every chunk, deduplicated. `translate_to` is caller-supplied and
  passes through unchanged, so it must not be inferred from the audio.

### Unverified risk, resolve first

**Does `response_format: json_schema` actually work alongside `input_audio` on Gemini's
OpenAI-compat endpoint?** Nothing in the repo does audio + structured output together today
(`chat_json` has exactly one test, in `tests/test_chat.py`, and it is `@pytest.mark.skip`). This was
not verified, because doing so writes a usage record to the local database. **This is step 1.** If the
combination is rejected, the fallback is prompt-only JSON plus `ChudResponse.parsed_json`, which needs
no schema support from the provider.

## Design

New files under this package:

```
_schemas/diarization.py       ChudUtterance, ChudSpeaker, ChudDiarization
_utils/vad.py                 VoiceActivity          (deterministic boundaries)
_utils/diarization.py         DiarizationRules       (validation, offsets, dedup, merge map)
_services/diarizer.py         AudioDiarizer
```

Core changes, both small:

```
_main.py    parallel_chat gains **request_kwargs               (R1)
_main.py    _ = response.json  ->  _ = response.parsed_json    (R2)
```

Shapes:

```python
class ChudUtterance(BaseModel):
    speaker: str
    language: str
    text: str
    translation: str | None = None  # "" on the wire, None here, R7/R8
    timestamp: list[float] = Field(min_length=2, max_length=2)


class ChudSpeaker(BaseModel):
    label: str
    description: str  # feeds reconciliation, R6


class ChudDiarization(BaseModel):
    languages: list[str]  # union across chunks, R19
    translate_to: str | None = None  # caller supplied, never inferred
    transcript: list[ChudUtterance]
    speakers: list[ChudSpeaker]
    speaker_map: dict[str, str]  # chunk-local label -> global, R14
```

`AudioDiarizer.diarize(path, *, translate_to=None, model=FLASH_LITE_3_5) -> ChudDiarization`.
Passing `translate_to` switches the prompt to ask for translation and turns on the empty-string
convention; omitting it keeps output minimal.

`VoiceActivity.utterances(path) -> list[AudioSpan]` holds the RMS pass, reusing `AudioBackend` for IO
so `soundfile` stays behind the one wall. Tunables as class constants (frame 30ms, silence gap 400ms,
threshold relative to the 95th-percentile speech level) since those are the values validated above.

`DiarizationRules` owns every decision per `logic-belongs-in-utils`: grouping utterance spans into
request-sized chunks, rendering the numbered-range prompt, flattening the provider schema (R7),
clamping and ordering, dedup, normalising redundant translations to `None` (R8), unioning languages,
and building the global speaker map.

`AudioDiarizer` orchestrates only: VAD, group, fan out via `parallel_chat` with `response_format` +
`temperature=0`, reconcile, assemble. Reuses `AudioFanOutRules.guard_model` and `batch_size`.

## Verification

All commands run from the repo root. Step 1 gates everything, so run it before writing the service:

```bash
# does audio + json_schema coexist on the compat endpoint? (see unverified risk above)
uv run python -c "
import asyncio; from pathlib import Path
from chudgpt import ChudGPT
from chudgpt.audio import AudioChunker
c = ChudGPT().app(app_name='chudgpt-pytests')
ch = next(iter(AudioChunker(chunk_seconds=15.0, limit_seconds=15.0).stream(Path('tests/assets/piper_french.wav'))))
r = asyncio.run(c.chat_json('List each utterance.', builder=ch.builder('List each utterance.'),
                            schema={'type':'object','properties':{'transcript':{'type':'array','items':{'type':'string'}}},'required':['transcript']}))
print(r.text[:300]); print(r.parsed_json)"
```

If that returns parseable JSON, also confirm the flattened schema survives a nullable field, since R7
is the other thing that can sink this: send a schema with `translation` as a plain string and check
the model returns `""` rather than omitting the key or emitting literal `null`.

Then the deterministic layers, which need no quota:

```bash
uv run pytest tests/test_vad.py -q          # same file -> identical spans, twice
uv run pytest tests/test_diarization.py -q  # clamping, dedup, merge map, count mismatch,
                                            # redundant translation -> None, language union
uv run ruff check src/
```

End-to-end:

```bash
uv run pytest -m live -k diarize -s
```

Assert: `languages` is the deduplicated union across chunks; `translate_to` echoes back exactly what
was passed; every `timestamp` pair exactly matches a `VoiceActivity` span, which is the proof that the
model contributed no numbers; segments strictly ordered; and two runs at `temperature=0` yield
identical text.

Translation needs a genuinely multi-language clip to test, and `piper_french.wav` is single-speaker
French only, so `diarize(translate_to="english")` against it verifies translation but not the
`translation is None` branch. Both gaps in the current fixtures are worth naming plainly:

- **No multi-speaker asset**, so speaker reconciliation (R13/R14) cannot be honestly tested at all
  until one is added.
- **No mixed-language asset**, so R8's conditional-population rule only gets half exercised.

A single short clip with two speakers and two languages would cover both, and until it exists those
requirements are unverified rather than working.