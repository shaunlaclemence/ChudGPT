from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from .chud_response import ChudResponse


class ChudChannel(str, Enum):
    THINKING = "thinking"
    ANSWER = "answer"
    DONE = "done"
    ERROR = "error"


class ChudStreamEvent(BaseModel):
    channel: ChudChannel
    text: str = ""
    response: ChudResponse | None = None
