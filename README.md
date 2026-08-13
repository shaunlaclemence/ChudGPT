# ChudGPT

Route chat prompts across free-tier provider API keys through one client. When a key hits
its daily limit (HTTP 429), ChudGPT rotates to the next key and retries, so your code
never sees the rotation.

Every provider is called through its OpenAI-compatible endpoint, so there is a single code
path. **Gemini is the only provider wired up today**; the others are stubbed out in
`src/chudgpt/_providers/config.py`.

## Install

Consumed as a git dependency, not published to PyPI:

```bash
uv add "chudgpt @ git+https://github.com/shaunlaclemence/ChudGPT.git@v0.5.1"
uv add "chudgpt[audio] @ git+https://github.com/shaunlaclemence/ChudGPT.git@v0.5.1"
```

Keys come from a `secrets.json` at your project root, found by walking up from the running
process. It is gitignored and must never be committed:

```json
{
  "gemini": [
    { "account": "you@example.com", "name": "Key 1",
      "project_name": "projects/123", "project_number": "123", "api_key": "..." }
  ]
}
```

Only `api_key` is required. Pass several entries to rotate across them.

## Quickstart

```python
import asyncio
from chudgpt import ChudGPT, GeminiModel

ChudGPT().initialise(app_name="My App")     # once, creates my-app.db
client = ChudGPT().app(app_name="My App")   # thereafter, attaches to it

reply = asyncio.run(client.chat("Explain monads in one sentence.",
                                model=GeminiModel.FLASH_LITE_3_5))
print(reply.data)                            # the answer text
print(reply.model, reply.duration)           # who served it, and how long it took
print(reply.usage.total)                     # tokens
```

The app name is kebab-cased into its own database file, so "My App", "my-app" and "My_App"
all resolve to `my-app.db`, and two apps never share a quota ledger. The database lives in
the platform data directory, overridable with `CHUDGPT_HOME`.

### Multi-turn and attachments

```python
from chudgpt.messages import Attachment, ChudMessageBuilder

builder = (ChudMessageBuilder()
           .system("Answer in one sentence.")
           .prompt(Attachment("diagram.png").prompt("What is this?")))

reply = await client.chat(builder=builder, model=GeminiModel.FLASH_3_6)
```

`Attachment` takes a path, or `data=`/`b64data=` with `format=` for in-memory bytes. It
detects the type and emits the right content part for audio, images, or video.

### Structured output

```python
from chudgpt import GeneratedCode, Language

answer = await client.chat_json(
    "Write a function that reverses a string.",
    schema=GeneratedCode.pin(language=Language.PYTHON),
    model=GeminiModel.FLASH_LITE_3_5,
)
```

`chat_json` returns the parsed dict. `ChudSchema.pin(field=value)` forces a field to a
single value, which is how enums are constrained (Gemini ignores JSON Schema `const`).
`ChudResponse.parse(Model)` validates a reply against a pydantic model.

### Running turns in parallel

```python
replies = await client.parallel_chat(
    {"a": builder_a, "b": builder_b},
    {"a": GeminiModel.FLASH_LITE_3_5, "b": GeminiModel.FLASH_LITE_3_1},
)
```

One model per builder, keyed the same. Pass `return_exceptions=True` to collect failures
instead of raising. Extra kwargs (`temperature`, `response_format`) pass through.

## Public API

Three modules are supported:

| Module | Contents |
| --- | --- |
| `chudgpt` | `ChudGPT`, `GeminiModel`, the response and usage types, the structured schemas, `__version__` |
| `chudgpt.exceptions` | every error |
| `chudgpt.messages` | `ChudMessageBuilder`, `Attachment`, `MessageContent` |
| `chudgpt.audio` | chunking, transcription, diarization (needs the `audio` extra) |

Everything else is underscore-prefixed (`_services`, `_schemas`, `_providers`, `_utils`,
`_db`) and free to change between versions.

## Models

`GeminiModel` is generated at import time from the packaged `src/chudgpt/config.json`
catalog, so adding a model there adds an enum member with no code change. Each member
carries `slug`, `rpd`, `rpm`, `tpm`, and `inputs`:

```python
GeminiModel.FLASH_LITE_3_5.rpd            # 500
GeminiModel.FLASH_LITE_3_5.accepts(Modality.AUDIO)   # True
GeminiModel("gemini-3.5-flash-lite")      # lookup by slug
```

The lite models allow 500 requests/day at 15/min; the rest allow 20/day. Both Gemma
members are text, image, and video only, so they reject audio. Note `GeminiModel.cheapest()`
returns a Gemma model, so audio calls must always pass `model=` explicitly.

Limits are per-account and change without notice, so a 429 remains the only authoritative
limit. Regenerate the typing stub after editing the catalog:
`uv run python scripts/generate_stub.py`.

## Usage tracking

```python
from chudgpt import UsagePeriod

client.get_requests(per=UsagePeriod.ONE_DAY)   # {provider: {model: count}}
client.get_tokens(per=UsagePeriod.ONE_HOUR)
client.get_usage_summary()                     # every record plus quotas
client.version                                 # installed version
```

Usage is recorded against the serving key on every call.

`client.scheduler` resets the daily quota at midnight America/Los_Angeles. Your app owns
its lifetime:

```python
client.scheduler.start()      # idempotent, call on every launch
client.scheduler.shutdown()   # on exit
```

If the app was closed when a reset was due, the next `start()` runs it immediately. Also
exposes `running`, `is_due`, `last_run`, and `next_run`.

## Errors

Every failure derives from `chudgpt.exceptions.BaseException`, so one `except` clause is
exhaustive. Each carries `error_code`, `service_code`, and the original exception on
`.error`. Codes read `<service>-<error>`, for example `002-404`:

```
001 FILE_SERVICE   002 DB_SERVICE       003 ROTOR_SERVICE
004 AUDIO_SERVICE  005 EXECUTOR_SERVICE 999 UNKOWN_SERVICE
```

Catch `BaseException` for anything from the library, a service base
(`FileServiceException`, `DBServiceException`, `RotorServiceException`,
`AudioServiceException`, `ExecutorServiceException`) for one subsystem, or a leaf class
such as `ChudGPTRateLimitException` for one condition.

## Audio

Needs the `audio` extra, which adds `soundfile`. Importing `chudgpt.audio` without it
raises `ChudGPTAudioBackendMissingException` (`004-424`), which is also an `ImportError`
and carries the exact install command.

A whole long recording sent as one request truncates: the audio costs roughly 25 input
tokens per second, and the reply hits the completion cap and starts repeating itself.
Chunking is the fix.

```python
from chudgpt.audio import AudioChunker, AudioTranscriber

transcript = await AudioTranscriber(client).transcribe(Path("interview.mp3"))
print(transcript.text)        # stitched, in order
print(transcript.segments)    # (span, text) per chunk
```

`AudioChunker(chunk_seconds=300.0, overlap_seconds=0.0, limit_seconds=None)` controls the
split. Chunks are encoded as MP3 when the backend can write it, because a 300s WAV chunk is
about 17.6MB of base64 against a 20MB inline limit, while the same span as MP3 is about
2.4MB.

`AudioDiarizer(client).diarize(path, translate_to="english")` returns `ChudDiarization`
with `languages`, `transcript` (speaker, language, text, translation, timestamp per
utterance), `speakers`, and `speaker_map`. Timestamps come from a voice-activity pass over
the waveform rather than from the model, so they are reproducible.

## Development

```bash
uv sync --extra audio
uv run pytest              # -m "not live" to skip tests that call the API
uv run ruff check
uv version --bump patch    # bumps pyproject.toml and re-locks
```

Tests marked `live` spend real quota; `audio` needs the extra. `pyproject.toml` is the
single source of truth for the version, read back at runtime through
`importlib.metadata`.

## Terms of service

Rotating across different providers, one free key each, is ordinary failover and is what
this library is for. Multiple keys for one provider is supported for legitimate cases (a
work and a personal account, say), but creating multiple free-tier accounts on one provider
to multiply your quota violates most providers' terms and can get all of them banned.
