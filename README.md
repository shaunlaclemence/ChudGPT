# ChudGPT

Route chat prompts across the free tiers of multiple AI providers through one client.
When a provider's daily free limit is hit (HTTP 429), ChudGPT cools that key down and
transparently retries the next provider — your code never sees the rotation.

Supported out of the box (all via their OpenAI-compatible endpoints, so there is a
single code path): **Gemini, Groq, Mistral, xAI (Grok), OpenRouter**.

## Quickstart

```bash
uv add chudgpt  # or: pip install -e .
cp .env.example .env  # fill in whichever keys you have
```

```python
from chudgpt import ChudClient, AllProvidersExhausted

client = ChudClient()             # discovers keys from env vars

reply = client.ask("Explain monads in one paragraph.")
print(reply.text)                 # the answer
print(reply.provider, reply.model)  # who actually served it
print(reply.key_id)               # which key served it — index into client.usage()

reply = client.ask(
    messages=[{"role": "user", "content": "hi"}],
    tier="best",                  # "best" | "fast" (default), or pass model="..."
    temperature=0.2,              # extra kwargs pass through to the API
)

try:
    client.ask("...")
except AllProvidersExhausted as e:
    print(e.statuses)             # why each key is unavailable
    print(e.earliest_reset)       # when to try again

print(client.status())            # health of every key right now
print(client.usage())             # requests/tokens used today, % of daily cap
```

## Public API

`ChudClient` is the entry point — you shouldn't need anything else. Also exported:
the `Tier`/`Model`/`Temperature` enums, the result types (`Response`, `StreamChunk`,
`KeyUsage`, `Conversation`), and the exception hierarchy below.

Everything else (`Rotor`, `PROVIDERS`, `ProviderConfig`, `QuotaTracker`, keystore
and state helpers) is internal — still reachable via submodules if you need it, but
not part of the supported surface and free to change between versions.

### Errors

Every failure derives from `ChudGPTError`, so one `except` clause is exhaustive:

```
ChudGPTError
├── ConfigError               bad/missing configuration
│   ├── SecretsFileError      secrets file unreadable, malformed, or empty
│   └── UnknownProviderError  provider name not in the registry
├── InvalidRequestError       caller error — rotating wouldn't help
│   └── InvalidTierError      tier isn't "best" or "fast"
├── ProviderError             a single provider call failed
│   └── StreamInterrupted     stream died after content was already yielded
└── AllProvidersExhausted     every key rate-limited, exhausted, or failing
```

`InvalidRequestError` covers caller mistakes that no amount of failover can fix —
an unknown pinned model, an out-of-range `temperature`, passing both `prompt` and
`messages`. These raise immediately rather than burning through your key pool.

## How it works

1. Providers are tried in priority order (Gemini → Groq → Mistral → xAI → OpenRouter).
2. Each provider is called through the `openai` client pointed at its base URL.
3. On **429**: the key is cooled down — honouring `Retry-After` when present, otherwise
   until the provider's daily reset (midnight Pacific for Gemini, a 15-minute default
   cooldown for rolling-window providers) — and the next candidate is tried.
4. On transient 5xx/network errors: 60-second cooldown, next candidate.
5. Requests and tokens are counted per key per day and persisted to
   `~/.chudgpt/state.json` (override with `CHUDGPT_STATE`), so known daily caps are
   skipped proactively across restarts. The 429 remains the source of truth — free-tier
   limits drift and the `known_rpd` values in `config.py` are only hints.

Keys are read from `GEMINI_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `XAI_API_KEY`,
`OPENROUTER_API_KEY` (any subset), or from `~/.chudgpt/keys.json`. The state file only
ever stores hashed key ids, never the keys.

Free key signup: [Gemini](https://aistudio.google.com/apikey) ·
[Groq](https://console.groq.com/keys) · [Mistral](https://console.mistral.ai/api-keys) ·
[xAI](https://console.x.ai) · [OpenRouter](https://openrouter.ai/keys)

## Streaming client

For a conversational, token-streaming interface on top of the same rotation logic:

```python
import asyncio
from chudgpt import ChudClient

async def main():
    client = ChudClient(secrets_path="secrets.json")  # or: ChudClient(keys={"gemini": [...]})
    convo = client.start_conversation()

    async for token in convo.send("Explain monads in one paragraph."):
        print(token, end="", flush=True)
    print()

    # continues the same conversation — full history is sent each turn
    async for token in convo.send("Now in one sentence."):
        print(token, end="", flush=True)

asyncio.run(main())
```

`secrets_path` accepts a per-account key inventory file (conventionally named `secrets.json`,
gitignored, never committed — a JSON map of provider name to a list of `{account, name,
project_name, project_number, api_key}` entries). Only the bare `api_key` values are ever read
out of it. `Conversation.ask(prompt)` is a non-streaming alternative that blocks for the full
`Response`.

## Discovering models

`chudgpt.Model` is a generated enum of every real, currently-usable model id per provider
(kept in `src/chudgpt/config.json` — a non-secret, packaged catalog, not to be confused with
your own `secrets.json`). `chudgpt.Tier` (`BEST`/`FAST`) picks the provider's default for a
quality/speed tradeoff instead of naming an exact model:

```python
from chudgpt import ChudClient, Model, Tier

client = ChudClient(secrets_path="secrets.json", providers=["gemini"])
reply = client.ask("hi", model=Model.GEMINI_3_6_FLASH)  # exact model
reply = client.ask("hi", tier=Tier.BEST)                # provider's default "best" model
```

Pin `model=` only alongside `providers=["<that provider>"]` — a model id is only valid for
the provider that defines it, so a request that rotates elsewhere raises
`InvalidRequestError`. Pass `providers=` plain names; it also sets failover order.

Model ids drift: Google is retiring the `gemini-2.5-*` line ("no longer available to new
users"), which is why the Gemini tier defaults point at `gemini-3.6-flash` and
`gemini-3.5-flash-lite`. If a pinned model starts raising `InvalidRequestError`, check
`src/chudgpt/config.json` against the provider's current docs and regenerate:
`uv run python scripts/generate_params.py`.

## Speed benchmark

Times every model in the catalog against the same prompt, one streaming call each,
reporting time-to-first-token, total latency, tokens/sec and usage. Opt-in, since it
costs one real request per model:

```bash
CHUDGPT_SPEED=1 uv run pytest tests/test_live_speed.py -s
```

Models your account can't reach are listed as unavailable rather than failing the run.

## Terms-of-service note

Rotating across **different providers** — one free key each — is ordinary failover and
is what this library is for. Comma-separating **multiple keys for the same provider**
is supported for legitimate cases (e.g. a work and a personal account), but creating
multiple free-tier accounts on one provider to multiply your quota violates most
providers' terms of service and can get all of those accounts banned. Don't do that.

## Development

```bash
uv sync
uv run pytest       # mocked test suite, no keys needed
uv run ruff check
```

## Not yet implemented

Embeddings/vision, agentic tool-calling, a local proxy server.
