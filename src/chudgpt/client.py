from __future__ import annotations

from pathlib import Path
from typing import Any

from chudgpt.scheduler import DailyScheduler

from .providers.gemini import GeminiModel
from .rotor import Rotor
from .schemas.chat import Message, MessageRole, Response
from .utils.keys import load_secrets


class MessageBuilder:
    def __init__(self) -> None:
        self.messages_list: list[Message] = []

    def system(self, content: Any):
        if len(self.messages_list) > 0:
            raise ValueError("System must be the first message")
        self.messages_list.append(Message(role=MessageRole.SYSTEM, content=content))
        return self

    def prompt(self, content: Any):
        self.messages_list.append(Message(role=MessageRole.USER, content=content))
        return self

    def assistant(self, content: Any):
        self.messages_list.append(Message(role=MessageRole.ASSISTANT, content=content))
        return self

    def messages(self, messages: list[Message]):
        self.messages_list += messages
        return self


class ChudGPT:
    def __init__(self, secrets_path: Path | str, timeout: float = 30.0):
        self._rotor = Rotor(load_secrets(secrets_path), timeout=timeout)
        self.scheduler = DailyScheduler()

    async def chat(
        self,
        prompt: str | None = None,
        *,
        messages: list[Message] | None = None,
        system: str | None = None,
        builder: MessageBuilder | None = None,
        model: GeminiModel | None = None,
    ) -> Response:
        if builder:
            if prompt is not None or messages is not None or system is not None:
                raise ValueError(
                    "pass builder on its own, not alongside prompt/messages/system"
                )
            return await self._rotor.chat(messages=builder.messages_list, model=model)
        else:
            return await self._rotor.chat(
                prompt, messages=messages, system=system, model=model
            )
