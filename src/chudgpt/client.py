"""ChudClient: a conversational, streaming wrapper around Rotor."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from .config import AUDIO_PROVIDERS, PROVIDERS, TIERS, ProviderConfig, provider_by_name
from .errors import InvalidRequestError, InvalidTierError
from .keystore import load_keys_from_secrets_json
from .rotor import KeyUsage, Response, Rotor, StreamChunk


def _audio_message(
    audio: bytes | str | Path, audio_format: str | None, prompt: str
) -> dict[str, Any]:
    """Build one user chat message carrying ``prompt`` plus an audio clip as an
    ``input_audio`` content part (base64) — the shape an OpenAI-compatible endpoint
    expects for audio input. ``audio`` may be raw bytes or a path; with bytes,
    ``audio_format`` (e.g. ``"mp3"``) is required since there's no filename to read
    it from."""
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        data = path.read_bytes()
        fmt = (audio_format or path.suffix).lstrip(".").lower()
    else:
        data = audio
        fmt = (audio_format or "").lstrip(".").lower()
    if not fmt:
        raise InvalidRequestError(
            "audio_format is required when passing raw audio bytes"
        )
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"data": encoded, "format": fmt}},
        ],
    }


class Conversation:
    """A multi-turn chat session. History lives in memory for the life of this object."""

    def __init__(
        self,
        rotor: Rotor,
        *,
        tier: str = "fast",
        model: str | None = None,
        system: str | None = None,
    ):
        self._rotor = rotor
        self._tier = tier
        self._model = model
        self.messages: list[dict[str, str]] = []
        if system:
            self.messages.append({"role": "system", "content": system})
        self.last_meta: dict[str, Any] | None = None

    async def send(
        self,
        prompt: str,
        *,
        tier: str | None = None,
        model: str | None = None,
        **request_kwargs: Any,
    ) -> AsyncIterator[str]:
        """Send a prompt and stream back the reply, extending the conversation history."""
        self.messages.append({"role": "user", "content": prompt})
        chunks: list[str] = []
        async for chunk in self._rotor.chat_stream(
            messages=self.messages,
            tier=tier or self._tier,
            model=model or self._model,
            **request_kwargs,
        ):
            if chunk.done:
                self.last_meta = {
                    "provider": chunk.provider,
                    "model": chunk.model,
                    "key_id": chunk.key_id,
                    "usage": chunk.usage,
                }
                continue
            chunks.append(chunk.delta)
            yield chunk.delta
        self.messages.append({"role": "assistant", "content": "".join(chunks)})

    def ask(
        self,
        prompt: str,
        *,
        tier: str | None = None,
        model: str | None = None,
        **request_kwargs: Any,
    ) -> Response:
        """Non-streaming turn: blocks for the full reply, still extends history."""
        self.messages.append({"role": "user", "content": prompt})
        reply = self._rotor.chat(
            messages=self.messages,
            tier=tier or self._tier,
            model=model or self._model,
            **request_kwargs,
        )
        self.messages.append({"role": "assistant", "content": reply.text})
        self.last_meta = {
            "provider": reply.provider,
            "model": reply.model,
            "key_id": reply.key_id,
            "usage": reply.usage,
        }
        return reply


class ChudClient:
    """ChudGPT v0.3.0

    Keys are resolved in this order: an explicit ``keys`` dict, a ``secrets_path``
    pointing at a per-account key inventory file (e.g. ``secrets.json`` — never
    commit it), or — if neither is given — environment variables /
    ``~/.chudgpt/keys.json`` via ``Rotor.from_env()``.

    ``providers`` narrows and orders which providers may serve a request. Pass
    plain names (``providers=["gemini"]``) — needed when pinning ``model=``, since
    a model id is only valid for the provider that defines it.

    Every failure raised from here derives from ``ChudGPTError``.
    """

    def __init__(
        self,
        keys: dict[str, list[str]] | None = None,
        *,
        secrets_path: Path | str | None = None,
        providers: Sequence[str | ProviderConfig] = PROVIDERS,
        state_file: Path | str | None = None,
        tier: str = "fast",
        model: str | None = None,
        **rotor_kwargs: Any,
    ):
        resolved = tuple(
            p if isinstance(p, ProviderConfig) else provider_by_name(p)
            for p in providers
        )
        self._validate_tier(tier)
        self._tier = tier
        self._model = model
        if keys is None and secrets_path is not None:
            keys = load_keys_from_secrets_json(secrets_path)
        if keys is not None:
            self._rotor = Rotor(
                keys, providers=resolved, state_file=state_file, **rotor_kwargs
            )
        else:
            self._rotor = Rotor.from_env(
                providers=resolved, state_file=state_file, **rotor_kwargs
            )

    @staticmethod
    def _validate_tier(tier: str | None) -> None:
        if tier is not None and tier not in TIERS:
            raise InvalidTierError(tier, list(TIERS))

    def status(self) -> dict[str, str]:
        return self._rotor.status()

    def usage(self) -> dict[str, KeyUsage]:
        """Today's request/token count per key, and % of the known daily cap used."""
        return self._rotor.usage()

    def ask(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        tier: str | None = None,
        model: str | None = None,
        **request_kwargs: Any,
    ) -> Response:
        """One-shot, non-streaming turn. Mirrors Rotor.chat()'s full signature."""
        self._validate_tier(tier)
        return self._rotor.chat(
            prompt,
            messages=messages,
            tier=tier or self._tier,
            model=model or self._model,
            **request_kwargs,
        )

    def transcribe(
        self,
        audio: bytes | str | Path,
        *,
        prompt: str,
        audio_format: str | None = None,
        providers: Sequence[str] | None = None,
        tier: str | None = None,
        model: str | None = None,
        **request_kwargs: Any,
    ) -> Response:
        """Send an audio clip plus ``prompt`` as one chat turn and return the reply.

        What comes back is whatever ``prompt`` asks for — a plain transcript, a
        diarized JSON structure, a summary — because this is an ordinary chat
        completion with the audio attached, not a fixed transcription endpoint. The
        call is restricted to audio-capable providers (``config.AUDIO_PROVIDERS``,
        Gemini by default) so a gemini-only model is never sent to a text-only key;
        pass ``providers`` to override that set.

        ``audio`` is raw bytes or a path (pass ``audio_format`` like ``"mp3"`` with
        bytes). The clip is inlined as base64 over the provider's OpenAI-compatible
        endpoint, so keep it within that provider's request-size limit — chunk long
        recordings.
        """
        self._validate_tier(tier)
        message = _audio_message(audio, audio_format, prompt)
        return self._rotor.chat(
            messages=[message],
            tier=tier or self._tier,
            model=model,
            providers=providers or AUDIO_PROVIDERS,
            **request_kwargs,
        )

    def stream(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        tier: str | None = None,
        model: str | None = None,
        **request_kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """One-shot streaming turn (no history kept). Mirrors Rotor.chat_stream()."""
        self._validate_tier(tier)
        return self._rotor.chat_stream(
            prompt,
            messages=messages,
            tier=tier or self._tier,
            model=model or self._model,
            **request_kwargs,
        )

    def start_conversation(
        self,
        system: str | None = None,
        *,
        tier: str | None = None,
        model: str | None = None,
    ) -> Conversation:
        self._validate_tier(tier)
        return Conversation(
            self._rotor,
            tier=tier or self._tier,
            model=model or self._model,
            system=system,
        )
