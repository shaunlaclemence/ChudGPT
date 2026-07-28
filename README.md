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
from chudgpt import Rotor, AllProvidersExhausted

rotor = Rotor.from_env()          # discovers keys from env vars

reply = rotor.chat("Explain monads in one paragraph.")
print(reply.text)                 # the answer
print(reply.provider, reply.model)  # who actually served it

reply = rotor.chat(
    messages=[{"role": "user", "content": "hi"}],
    tier="best",                  # "best" | "fast" (default), or pass model="..."
    temperature=0.2,              # extra kwargs pass through to the API
)

try:
    rotor.chat("...")
except AllProvidersExhausted as e:
    print(e.statuses)             # why each key is unavailable
    print(e.earliest_reset)       # when to try again

print(rotor.status())             # health of every key right now
```

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

Streaming, async client, embeddings/vision, a local proxy server.
