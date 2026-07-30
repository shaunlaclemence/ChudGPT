"""ChudClient: a conversational, streaming wrapper around Rotor."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .config import PROVIDERS, ProviderConfig
from .keystore import load_keys_from_secrets_json
from .rotor import KeyUsage, Response, Rotor, StreamChunk


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
            "usage": reply.usage,
        }
        return reply


class ChudClient:
    """Entry point: wraps Rotor with a conversation-friendly API.

    Keys are resolved in this order: an explicit ``keys`` dict, a ``secrets_path``
    pointing at a per-account key inventory file (e.g. ``secrets.json`` — never
    commit it), or — if neither is given — environment variables /
    ``~/.chudgpt/keys.json`` via ``Rotor.from_env()``.
    """

    def __init__(
        self,
        keys: dict[str, list[str]] | None = None,
        *,
        secrets_path: Path | str | None = None,
        providers: tuple[ProviderConfig, ...] = PROVIDERS,
        state_file: Path | str | None = None,
        tier: str = "fast",
        model: str | None = None,
        **rotor_kwargs: Any,
    ):
        if keys is None and secrets_path is not None:
            keys = load_keys_from_secrets_json(secrets_path)
        self._tier = tier
        self._model = model
        if keys is not None:
            self._rotor = Rotor(
                keys, providers=providers, state_file=state_file, **rotor_kwargs
            )
        else:
            self._rotor = Rotor.from_env(
                providers=providers, state_file=state_file, **rotor_kwargs
            )

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
        return self._rotor.chat(
            prompt,
            messages=messages,
            tier=tier or self._tier,
            model=model or self._model,
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
        return Conversation(
            self._rotor,
            tier=tier or self._tier,
            model=model or self._model,
            system=system,
        )
