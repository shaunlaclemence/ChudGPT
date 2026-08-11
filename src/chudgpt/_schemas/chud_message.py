from __future__ import annotations

from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel


class MessageContent:
    def __init__(self, prompt: str | None = None) -> None:
        self.text = prompt

    def build(self):
        return [{"type": "text", "text": self.text}]

ModelT = TypeVar("ModelT", bound=BaseModel)


class ChudMessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChudMessage(BaseModel):
    role: ChudMessageRole
    content: Any

    def build(self):
        if isinstance(self.content, MessageContent):
            c = self.content.build()
        else:
            c = self.content

        return ChudMessage(role=self.role, content=c)