from __future__ import annotations

from pathlib import Path

from .db.models import ModelQuota
from .providers.gemini import GeminiModel
from .rotor import Rotor
from .schemas.chat import Message, Response
from .utils.keys import load_secrets


class ChudGPT:
    def __init__(self, secrets_path: Path | str, timeout: float = 30.0):
        self._rotor = Rotor(load_secrets(secrets_path), timeout=timeout)

    async def chat(
        self,
        prompt: str | None = None,
        *,
        messages: list[Message] | None = None,
        system: str | None = None,
        model: GeminiModel | None = None,
    ) -> Response:
        return await self._rotor.chat(
            prompt, messages=messages, system=system, model=model
        )

    async def usage(self, model: GeminiModel | None = None) -> ModelQuota | None:
        """Today's stored spend for the current provider and model.

        @param model: which model's row to read; defaults to the model ``chat()``
            would pick, so a bare call describes the next request.
        """
        return await self._rotor.usage(model)
