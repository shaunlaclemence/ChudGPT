"""ChudGPT Messages Package

    from chudgpt.messages import ChudMessageBuilder

Use this micro-package to build messages to send through ChudGPT's public API
"""

from pathlib import Path

from chudgpt._schemas import ChudMessage, ChudMessageRole, MessageContent
from chudgpt._utils.attachments import AttachmentRules


class Attachment(MessageContent):
    def __init__(
        self,
        file_path: Path | None = None,
        *,
        data: bytes | None = None,
        b64data: str | None = None,
        format: str | None = None,
    ) -> None:
        self.data, self.mime, self.extension = AttachmentRules.resolve(
            file_path, data, b64data, format
        )
        super().__init__()

    def prompt(self, prompt: str):
        self.text = prompt
        return self

    def build(self):
        res = [AttachmentRules.part(self.mime, self.extension, self.data)]
        if self.text:
            res.append({"type": "text", "text": self.text})
        return res


class ChudMessageBuilder:
    def __init__(self) -> None:
        self.messages_list: list[ChudMessage] = []

    def system(self, content: str):
        if len(self.messages_list) > 0:
            raise ValueError("System must be the first message")
        self.messages_list.append(
            ChudMessage(role=ChudMessageRole.SYSTEM, content=content)
        )
        return self

    def prompt(self, content: str | MessageContent):
        self.messages_list.append(
            ChudMessage(role=ChudMessageRole.USER, content=content).build()
        )
        return self

    def assistant(self, content: str | MessageContent):
        self.messages_list.append(
            ChudMessage(role=ChudMessageRole.ASSISTANT, content=content).build()
        )
        return self

    def messages(self, messages: list[ChudMessage]):
        self.messages_list += messages
        return self


__all__ = [
    "Attachment",
    "ChudMessage",
    "ChudMessageBuilder",
    "ChudMessageRole",
    "MessageContent",
]
