# ChudGPT

One client over a pool of free-tier Gemini keys. It rotates keys when a limit is hit,
keeps a per-app quota ledger in SQLite, and ships an audio package that transcribes and
diarizes recordings too long to send in a single turn.

Everything goes through Gemini's OpenAI-compatible endpoint, so there is one code path.

## Install

```bash
uv add "chudgpt @ git+https://github.com/shaunlaclemence/ChudGPT.git@v0.4.2"

# with transcription and diarization
uv add "chudgpt[audio] @ git+https://github.com/shaunlaclemence/ChudGPT.git@v0.4.2"
```

The `audio` extra pulls in `soundfile` and `numpy`. Without it `client.audio` is simply
absent, and importing `chudgpt.audio` raises `ChudGPTAudioBackendMissingException`,
which is also an `ImportError`.

## Keys

Keys live in a `secrets.json` at your project root, found by walking up from the running
process so it works from a script, a test, or a nested package. Keep it out of git.

```json
{
  "gemini": [
    {
      "account": "you@example.com",
      "name": "Personal key",
      "project_name": "projects/803725075154",
      "project_number": "803725075154",
      "api_key": "AIza..."
    }
  ]
}
```

Each entry is one place the rotor can move to when a key is rate limited, so a
single-entry file cannot rotate. Only a masked key ever reaches the database
(`**48**C2D9g` records the length and last five characters); the live key stays in
`secrets.json`.

## Quickstart

```python
import asyncio
from chudgpt import ChudGPT, GeminiModel

ChudGPT().initialise(app_name="My App")     # once, creates my-app.db
client = ChudGPT().app(app_name="My App")   # every run after that

reply = asyncio.run(client.text.chat("Explain monads in one sentence."))
print(reply.data)
print(reply.model, reply.usage, f"{reply.duration:.2f}s")
```

The app name is kebab-cased into its own SQLite file, so `"My App"`, `"my-app"` and
`"My_App"` all resolve to `my-app.db` and two apps never share one quota ledger.
`app()` raises `ChudGPTNotFoundException` when no database exists yet, so a typo fails
loudly instead of silently starting a second ledger.

---

# Public API

The client is the only entry point. Every capability hangs off a namespace on it, so
nothing but `ChudGPT` itself needs importing:

```python
client.text.chat / chat_json / parallel_chat / stream / parallel_stream
client.audio.diarize / diarize_stream / transcribe / voice_activity / chunks / builders
```

`text` is always there. `audio` is a plugin: it exists only when the `audio` extra is
installed, so `hasattr(client, "audio")` is the feature check, and reaching for it
without the extra raises an `AttributeError` naming the install command.

Four modules are importable: `chudgpt` (the client, its types, `GeminiModel`),
`chudgpt.messages`, `chudgpt.audio` (audio types only) and `chudgpt.exceptions`. No
service is importable from any of them. Everything else (services, ORM models, the key
store, the rules classes) is internal and may change without notice.

## `ChudGPT`

```python
ChudGPT(timeout: float = 30.0)
    .initialise(app_name) -> ChudGPT
    .app(app_name) -> ChudGPT
    .db_path -> Path
    .scheduler -> SchedulerService
    .text -> the chat namespace
    .audio -> the audio namespace, when the extra is installed
```

The default 30s timeout suits text. Long audio needs far more; the audio namespace sets
its own per-request timeout from the chunk length, but a direct `text.chat()` with a
large attachment needs `ChudGPT(timeout=300)` or `timeout=` on the call.

Both `text` and `audio` need a bound client, so call `initialise()` or `app()` first;
touching `client.text` before that raises `ChudGPTNotFoundException`.

## `client.text`

### chat

```python
await client.text.chat(
    prompt=None, *, messages=None, system=None, builder=None, model=None, **request_kwargs
) -> ChudResponse
```

Pass exactly one of `prompt`, `messages`, or `builder`. Anything in `request_kwargs` goes
straight to the provider, which is how you reach `temperature`, `timeout`, `top_p` and
friends. Usage is recorded against the key that served the call.

```python
reply = await client.text.chat(
    "Rewrite this changelog entry.",
    system="You are a terse release engineer.",
    model=GeminiModel.FLASH_LITE_3_5,
    temperature=0,
)
```

### chat_json

```python
await client.text.chat_json(
    prompt=None, *, schema, messages=None, system=None, builder=None,
    model=None, schema_name=None, **request_kwargs
) -> dict[str, Any]
```

Same call, constrained to a JSON schema. Pass a Pydantic model and it is converted for
you, or a raw schema dict when you need control over the wire shape. Returns the whole
response envelope with the model's reply already parsed under `data`:

```python
class Recipe(BaseModel):
    title: str
    minutes: int

envelope = await client.text.chat_json("A quick pasta recipe.", schema=Recipe)

envelope["data"]      # {'title': ..., 'minutes': ...}
envelope["usage"]     # what the call cost
envelope["provider"]  # the key that served it

recipe = Recipe.model_validate(envelope["data"])
```

Gemini's validator ignores or rejects some JSON Schema constructs. Avoid
`tuple[float, float]`, which emits `prefixItems`, and `str | None`, which emits an
`anyOf` with a null branch. Prefer `list[float]` with length bounds, and a plain string
with an empty-string convention for "absent".

### stream

```python
client.text.stream(
    prompt=None, *, messages=None, system=None, builder=None, model=None,
    think=False, **request_kwargs
) -> AsyncIterator[ChudStreamEvent]
```

Token streaming with thinking and answer on separate channels. Gemini's compat endpoint
has no separate reasoning field; it wraps thoughts in tags inside the ordinary content,
so ChudGPT splits them for you.

```python
from chudgpt import ChudChannel

async for event in client.text.stream("Why is the sky blue?", model=GeminiModel.FLASH_3_6, think=True):
    if event.channel is ChudChannel.DONE:
        print(event.response.usage, event.response.duration)
    else:
        print(f"[{event.channel.value}] {event.text}", end="", flush=True)
```

| channel | meaning |
| --- | --- |
| `THINKING` | reasoning text, only when `think=True` on a model that reasons |
| `ANSWER` | the reply |
| `DONE` | final event, carries the assembled `ChudResponse` on `.response` |
| `ERROR` | a failed stream, message on `.text` |

`think=True` on a model without reasoning (any `*-lite`) simply yields no thinking
events. Usage arrives only on the final chunk, so it is recorded when the stream drains,
and `DONE.response` carries it along with duration and provider. Streams translate
errors but do **not** rotate or retry: once text has been yielded, replaying the call
would duplicate it.

### parallel_chat

```python
await client.text.parallel_chat(
    builders: dict[str, ChudMessageBuilder],
    models: dict[str, GeminiModel],
    *, return_exceptions=False, **request_kwargs
) -> dict[str, ChudResponse]
```

Fan several turns out at once, keyed by names you choose. Every key in `builders` needs a
model in `models`, or the call raises before spending anything.

```python
builders = {
    "summary": ChudMessageBuilder().prompt("Summarise this release."),
    "risks": ChudMessageBuilder().prompt("List the rollout risks."),
}
replies = await client.text.parallel_chat(builders, dict.fromkeys(builders, GeminiModel.FLASH_LITE_3_5))
print(replies["risks"].data)
```

With the default `return_exceptions=False`, one failure discards every sibling reply, and
those siblings have already been billed. Pass `True` on expensive fan-outs and check each
value before use.

### parallel_stream

```python
client.text.parallel_stream(
    builders, models, *, think=False, return_exceptions=True, **request_kwargs
) -> AsyncIterator[tuple[str, ChudStreamEvent]]
```

The same fan-out with the intermediate steps exposed, merged into one labelled stream so
you cannot accidentally consume the branches serially.

```python
speaking = None
async for name, event in client.text.parallel_stream(builders, models):
    if event.channel is ChudChannel.DONE:
        print(f"\n[{name}] done, {event.response.usage.total} tokens")
        continue
    if name != speaking:
        print(f"\n\n[{name}] ", end="")
        speaking = name
    print(event.text, end="", flush=True)
```

`return_exceptions` defaults to `True` here, the opposite of `parallel_chat`: streams do
not retry, and when four of five branches have already put text on screen, killing them
because the fifth failed throws away work the user can see. Collecting the `DONE` events
gives you exactly what `parallel_chat` would have returned.

## `GeminiModel`

Members are generated at import time from the packaged `config.json`, so adding a model
there adds an enum member with no code change. Each exposes `slug`, `rpd`, `rpm`, `tpm`,
`inputs` and `accepts(Modality.AUDIO)`. Look one up by slug with
`GeminiModel("gemini-3.6-flash")`.

| Member | Slug | Req/day | Req/min | Audio |
| --- | --- | ---: | ---: | --- |
| `FLASH_3_6` | `gemini-3.6-flash` | 20 | 5 | yes |
| `FLASH_3_5` | `gemini-3.5-flash` | 20 | 5 | yes |
| `FLASH_LITE_3_5` | `gemini-3.5-flash-lite` | 500 | 15 | yes |
| `FLASH_LITE_3_1` | `gemini-3.1-flash-lite` | 500 | 15 | yes |
| `FLASH_3_PREVIEW` | `gemini-3-flash-preview` | 20 | 5 | yes |
| `FLASH_2_5` | `gemini-2.5-flash` | 20 | 5 | yes |
| `FLASH_LITE_2_5` | `gemini-2.5-flash-lite` | 20 | 10 | yes |
| `GEMMA_4_26B` | `gemma-4-26b-a4b-it` | 14400 | 30 | no |
| `GEMMA_4_31B` | `gemma-4-31b-it` | 14400 | 30 | no |

Limits are per account and Google changes them without notice, so a 429 remains the only
authoritative limit. The rotor rotates to the next key on 429, retries with backoff on
503, and does not retry a timeout, since repeating an identical call against the same
deadline only spends quota to fail the same way.

## `ChudResponse`

| Member | Type | What it is |
| --- | --- | --- |
| `data` | `str` | the reply body, as the model returned it |
| `service` | `str` | provider family, currently `gemini` |
| `model` | `str` | slug that actually served the call |
| `usage` | `Usage` | prompt, completion, total, requests, reasoning |
| `provider` | `ChudProvider` | the key that served it, masked |
| `duration` | `float` | wall-clock seconds |
| `message` | `ChudMessage` | the reply as an assistant turn, ready to feed back |
| `parsed_json` | `dict` | the whole response with `data` embedded as an object |
| `parse(Model)` | `Model` | validate the embedded `data` into a Pydantic model |

The property is `parsed_json`, not `json`, because `BaseModel` already defines a
deprecated `json()` method that would silently shadow it.

## `chudgpt.messages`

```python
from chudgpt.messages import Attachment, ChudMessageBuilder

convo = (
    ChudMessageBuilder()
    .system("You answer in one sentence.")   # first message only
    .prompt("What is a rotor?")
    .assistant("A component that turns.")
    .prompt("And in this library?")
)
reply = await client.text.chat(builder=convo, model=GeminiModel.FLASH_LITE_3_5)
```

Every method returns the builder, so turns chain. Pass the finished builder as
`builder=`, on its own and never alongside `prompt`/`messages`/`system`.
`.system()` raises `ValueError` if anything has already been added, since a system turn
is only meaningful first. Turns are `ChudMessage` objects carrying a `ChudMessageRole`
(`SYSTEM`, `USER`, `ASSISTANT`) and content that is either a string or a
`MessageContent` such as an `Attachment`.

`Attachment(file_path=None, *, data=None, b64data=None, format=None)` carries media from
a path, raw bytes, or base64, with the MIME type resolved for you. Attach the caption
with `.prompt()` so the file and its instruction travel as one message:

```python
clip = Attachment(Path("interview.wav")).prompt("Transcribe this.")
reply = await client.text.chat(
    builder=ChudMessageBuilder().prompt(clip),
    model=GeminiModel.FLASH_LITE_3_5,   # required: audio
)
```

Inline media is capped around 20 MB once base64 encoded. For anything longer than a few
minutes use `client.audio.transcribe()` rather than one large attachment.

## Structured output

`ChudSchema` is a `BaseModel` with one extra classmethod, `pin()`, which emits the JSON
schema with chosen fields locked to a single value. It writes a one-value enum rather
than `const`, because Gemini silently ignores `const`.

```python
from chudgpt import GeneratedCode, Language

schema = GeneratedCode.pin(language=Language.PYTHON)
envelope = await client.text.chat_json("A function that flattens a list.", schema=schema)
code = GeneratedCode.model_validate(envelope["data"])
```

| Preset | Fields |
| --- | --- |
| `GeneratedCode` | `language`, `code`, `entrypoint`, `imports`, `dependencies`, `explanation` |
| `Classification` | `label`, `confidence`, `reasoning` |
| `SentimentAnalysis` | `sentiment`, `confidence`, `rationale` |

Supporting enums: `Language` (programming languages, not spoken), `Confidence`,
`Sentiment`.

## Usage and quota

```python
from chudgpt import UsagePeriod

client.get_requests(per=UsagePeriod.ONE_DAY)   # -> {ChudProvider: {model: count}}
client.get_tokens(per=UsagePeriod.ONE_MIN)     # -> {ChudProvider: {model: tokens}}
client.get_usage_summary()                     # -> ChudUsageSummary
```

Every served call writes a row against the serving key. `UsagePeriod` offers `ONE_DAY`,
`FIVE_HOUR`, `ONE_HOUR`, `FIVE_MIN` and `ONE_MIN`; pair `ONE_MIN` with a model's `rpm`
and `ONE_DAY` with its `rpd` to see how close you are to a limit.

| Type | Fields |
| --- | --- |
| `ChudUsageRecord` | `model`, `provider`, `created_at`, `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `total_tokens` |
| `ChudQuota` | `slug`, `rpd`, `rpm`, `tpm`, `inputs` |
| `ChudProvider` | `id`, `email`, `name`, `project_name`, `project_number`, `api_key` (masked) |
| `Usage` | `prompt`, `completion`, `total`, `requests`, `reasoning` |

`ChudProvider` hashes and compares on `id` alone, which is what lets it key the
dictionaries above. `created_at` comes back timezone-aware in UTC.

### Daily reset

Free-tier daily quotas roll over at midnight America/Los_Angeles. `client.scheduler`
clears the usage ledger at that boundary, but does not start itself, because a library
must not own a background thread your app did not ask for.

```python
client.scheduler.start()      # idempotent, call on every launch
client.scheduler.shutdown()   # on exit

client.scheduler.running      # bool
client.scheduler.is_due       # missed the last boundary?
client.scheduler.last_run     # datetime | None
client.scheduler.next_run     # datetime | None
```

If the app was closed when a reset was due, the next `start()` runs it immediately.

## `client.audio`

Needs the `audio` extra. Present on the client only when that extra is installed.

Every method takes the same optional chunking arguments, so nothing has to be
constructed by hand: `chunk_seconds=300.0`, `overlap_seconds=0.0`, `limit_seconds=None`,
`format=None`. `limit_seconds` stops early, which is how you work on the first few
minutes of a long file while iterating.

### transcribe

```python
await client.audio.transcribe(
    file_path, *, prompt=..., model=GeminiModel.FLASH_LITE_3_5, concurrency=8,
    chunk_seconds=300.0, overlap_seconds=0.0, limit_seconds=None, format=None
) -> AudioTranscript
```

Chunks the recording, fans the chunks out in batches sized to the model's rate limit,
then makes one final call that stitches the pieces into a single text, trimming the
repeated words in each overlap. Single-chunk files skip the stitch.

```python
transcript = await client.audio.transcribe(
    Path("lecture.mp3"),
    model=GeminiModel.FLASH_LITE_3_5,
    chunk_seconds=60.0,
    overlap_seconds=5.0,
)

transcript.text        # the stitched transcript
transcript.segments    # ((AudioSpan, str), ...) per chunk, before stitching
transcript.responses   # {"chunk-000": ChudResponse, ..., "stitch": ChudResponse}
```

Chunk length dominates accuracy. Measured against the book text, 600s chunks scored 59.6%
because the model began repeating passages, while 60s chunks with the stitch pass scored
98.1%.

### diarize

```python
await client.audio.diarize(
    file_path, *, translate_to=None, max_speakers=None,
    model=GeminiModel.FLASH_LITE_3_5, concurrency=8,
    chunk_seconds=300.0, overlap_seconds=0.0, limit_seconds=None, format=None
) -> ChudDiarization
```

Who spoke, in what language, and when. The waveform decides the boundaries and the model
only fills each range in, so **no timestamp you receive was invented by a model**.
Speaker labels are chunk-local until one final text call merges them across chunks.

```python
result = await client.audio.diarize(
    Path("interview.mp3"),
    translate_to="EN",   # optional, adds a translation per utterance
    max_speakers=2,      # optional ceiling, strongly recommended
    chunk_seconds=180.0,
)

for u in result.transcript:
    print(f"[{u.start:.2f}-{u.end:.2f}] {u.speaker} ({u.language}): {u.text}")

result.text          # every utterance collated
result.languages     # ['EN', 'FR'] deduplicated ISO 639-1 union
result.speaker_map   # {'chunk-000/Speaker A': 'Speaker A', ...}
result.usage         # totalled across every call it made
```

Deterministic: every timestamp, the ordering, the deduplication, the language codes.
Best effort: the transcription, the speaker split within a chunk, and above all the
cross-chunk merge, which compares written voice descriptions with no acoustic evidence.
That is why `speaker_map` is returned rather than hidden, so you can audit or override
the merge it chose.

`translate_to` passes through unchanged and is never inferred from the audio. A
translation is filled in only when the utterance is in a different language, so
`translation` is `None` rather than a copy of `text`.

### diarize_stream

```python
client.audio.diarize_stream(
    file_path, *, translate_to=None, max_speakers=None,
    model=GeminiModel.FLASH_LITE_3_5, concurrency=8,
    chunk_seconds=300.0, overlap_seconds=0.0, limit_seconds=None, format=None
) -> AsyncIterator[ChudProgress]
```

The same pipeline with per-chunk progress exposed, for a UI that wants a live board
rather than a spinner. `diarize()` is implemented on top of this, so there is one
pipeline and the two cannot drift.

```python
board: dict[str, ChudProgress] = {}

async for update in client.audio.diarize_stream(path, translate_to="EN"):
    board[update.chunk] = update            # last write wins per chunk
    if update.result is not None:
        diarization = update.result         # the assembled ChudDiarization
```

```
  0.6s  chunk-000   queued        Queued, 18 utterances to fill
  5.8s  chunk-004   transcribing  Transcribing EL, 1/6 utterances
  6.6s  chunk-002   translating   Translating EL, 2/6 done
  7.2s  chunk-005   done          Done, 2 utterances, EL
 48.8s  __global__  merging       Merging 12 speaker labels
 50.8s  __global__  done          61 utterances, 2 speakers, EL, EN
```

| Field | What it is |
| --- | --- |
| `chunk` | chunk name, or `ChudProgress.GLOBAL` (`__global__`) for whole-job steps; test it with `update.is_global` |
| `phase` | a `ChudPhase`: `QUEUED`, `SENT`, `TRANSCRIBING`, `TRANSLATING`, `MERGING`, `DONE`, `FAILED` |
| `detail` | a line ready to render, e.g. `Transcribing FR, 3/9 utterances` |
| `at` | seconds elapsed since the call started |
| `span` | the slice of audio this chunk covers, for laying the board out by timeline |
| `utterances`, `languages` | counts and ISO codes so far |
| `result` | only on the final event, the assembled `ChudDiarization` |

Status is derived locally from pipeline state and the partially-arrived JSON, never
asked of the model, so it costs no tokens and cannot be hallucinated. Events are
coalesced on `(chunk, phase, utterances, languages)`, so hundreds of deltas produce only
the handful of updates where the summary actually changed, and a slow consumer can drop
intermediate states safely because the next event for that chunk supersedes them.

Chunks are dispatched in rate-limit-sized batches and arrive out of order, so key the
board by `chunk` and sort by `span.start` for display.

### voice_activity

```python
client.audio.voice_activity(file_path) -> list[AudioSpan]
```

The deterministic half of diarization, usable alone. A short-term RMS pass over 30ms
frames, thresholded against the clip's own speech level, with runs separated by less than
400ms of silence merged into one utterance and anything under 200ms discarded. Because it
is a pure function of the samples, the same file gives byte-identical spans on every run.
It costs no quota and makes no provider call.

It detects when *someone* is speaking, not *who*, so two people with no pause between
them land in one span.

### chunks and builders

```python
client.audio.chunks(file_path, *, chunk_seconds=300.0, ...) -> Iterator[AudioChunk]
client.audio.builders(file_path, prompt=..., *, chunk_seconds=300.0, ...) -> dict[str, ChudMessageBuilder]
```

The chunking layer on its own, for pipelines the three methods above do not cover. Cuts a
recording into inline-sized pieces and encodes each one, streaming so only the current
chunk is held in memory. `builders` returns the same chunks already wrapped as prompts
keyed by chunk name, ready to hand straight to `client.text.parallel_chat`:

```python
builders = client.audio.builders(Path("lecture.mp3"), chunk_seconds=60.0)
replies = await client.text.parallel_chat(
    builders, dict.fromkeys(builders, GeminiModel.FLASH_LITE_3_5)
)
```

### Result types

| Type | Members |
| --- | --- |
| `AudioSpan` | `index`, `start`, `end`, `duration`, `name` |
| `AudioChunk` | `span`, `data`, `format`, `sample_rate`, `attachment()`, `builder()` |
| `AudioTranscript` | `text`, `segments`, `responses` |
| `ChudUtterance` | `speaker`, `language`, `text`, `translation`, `timestamp`, `start`, `end` |
| `ChudSpeaker` | `label`, `description` |
| `ChudDiarization` | `languages`, `translate_to`, `transcript`, `speakers`, `speaker_map`, `responses`, `text`, `usage` |
| `ChudProgress` | `chunk`, `phase`, `detail`, `at`, `span`, `utterances`, `languages`, `result` |

`chudgpt.audio` exports these types and nothing else. The classes that produce them
(the chunker, the voice-activity pass, the transcriber, the diarizer) are services and
are reachable only through `client.audio`.

## `chudgpt.exceptions`

Everything raised by this library derives from `chudgpt.exceptions.BaseException`, so one
`except` clause is exhaustive. Each error carries a message, an `error_code`, a
`service_code`, and the original error on `.error`, giving a full code like `003-429`.

```python
from chudgpt.exceptions import (
    BaseException,              # anything from this library
    RotorServiceException,      # one subsystem
    ChudGPTRateLimitException,  # one condition
)

try:
    reply = await client.text.chat("hello")
except ChudGPTRateLimitException as err:
    print(err.error_code, err.service_code, err.error)
```

| Code | Service | Base class |
| ---: | --- | --- |
| 001 | Files | `FileServiceException` |
| 002 | Database | `DBServiceException` |
| 003 | Rotor and providers | `RotorServiceException` |
| 004 | Audio | `AudioServiceException` |
| 999 | Unknown | `BaseException` |

| Code | Exception | Raised when |
| ---: | --- | --- |
| 401 | `ChudGPTUnauthorizedException` | credentials missing, invalid, or rejected |
| 403 | `ChudGPTForbiddenException` | access denied, usually file permissions |
| 404 | `ChudGPTNotFoundException` | record, file, or unbound app not found |
| 400 | `ChudGPTBadDataException` | payload corrupt, malformed, or the wrong type |
| 409 | `ChudGPTConflictException` | write rejected by a constraint or duplicate |
| 422 | `ChudGPTInvalidPathException` | path is a directory or unusable segment |
| 422 | `ChudGPTDBConfigException` | config or secrets JSON invalid or incomplete |
| 424 | `ChudGPTAudioBackendMissingException` | the audio extra is not installed |
| 429 | `ChudGPTRateLimitException` | provider refused; quota or RPM exhausted |
| 500 | `ChudGPTInternalServerException` | unhandled failure, treat as a bug |
| 503 | `ChudGPTServiceUnavailableException` | resource unreachable or model overloaded |
| 504 | `ChudGPTTimeoutException` | provider did not answer inside the timeout |

503 and 504 differ deliberately. A 503 means the model was busy, and the rotor retries it
with doubling backoff. A 504 means your timeout was too short, and it is not retried,
since the identical call would reach the same deadline again at full quota cost.

---

## Known traps

- **The implicit model cannot hear.** Omitting `model=` gives you `GeminiModel.cheapest()`,
  a Gemma model with no audio support. Always name a model for audio.
- **30s is a text timeout.** Direct `chat()` calls with long attachments need a larger
  one. The audio services already scale theirs to the chunk.
- **One fan-out failure discards the batch.** `parallel_chat` defaults to
  `return_exceptions=False`, and the siblings were already billed. `parallel_stream`
  defaults the other way.
- **Rotation needs somewhere to go.** A `secrets.json` with one key cannot rotate on a 429.
- **Provider ids are reassigned.** Provider and quota rows are replaced on every bind, so
  ids track the order of `secrets.json`. Reordering it repoints older usage rows.
- **Temperature 0 is not determinism.** Three diarization runs of identical input scored
  91.0%, 79.9% and 92.6%. Only the timestamps are reproducible.

## Audio

Rotating across different providers, one free key each, is ordinary failover and is what
this library is for. Multiple keys for one provider is supported for legitimate cases
(a work and a personal account), but creating multiple free-tier accounts on one provider
to multiply your quota violates most providers' terms of service and can get all of those
accounts banned. Don't do that.

## Development

```bash
uv sync --extra audio
uv run pytest -m "not live"   # offline suite, no keys or quota needed
uv run ruff check
uv version --bump patch    # bumps pyproject.toml and re-locks
```

Test markers: `live` hits a real provider and burns quota, `audio` needs the audio extra,
`ollama` needs a local ollama server with the model pulled.
