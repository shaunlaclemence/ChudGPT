from __future__ import annotations

from chudgpt._schemas.chud_stream import ChudChannel, ChudStreamEvent


class StreamRules:
    OPEN = "<thought>"
    CLOSE = "</thought>"

    def __init__(self) -> None:
        self._buffer = ""
        self._thinking = False

    @property
    def channel(self) -> ChudChannel:
        return ChudChannel.THINKING if self._thinking else ChudChannel.ANSWER

    def feed(self, piece: str) -> list[ChudStreamEvent]:
        self._buffer += piece
        events: list[ChudStreamEvent] = []

        while True:
            tag = self.CLOSE if self._thinking else self.OPEN
            found = self._buffer.find(tag)
            if found == -1:
                break
            events.extend(self.__emit(self._buffer[:found]))
            self._buffer = self._buffer[found + len(tag) :]
            self._thinking = not self._thinking

        held = self.partial(self._buffer, self.CLOSE if self._thinking else self.OPEN)
        cut = len(self._buffer) - held
        events.extend(self.__emit(self._buffer[:cut]))
        self._buffer = self._buffer[cut:]
        return events

    def flush(self) -> list[ChudStreamEvent]:
        events = self.__emit(self._buffer)
        self._buffer = ""
        return events

    @classmethod
    def partial(cls, text: str, tag: str) -> int:
        for size in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:size]):
                return size
        return 0

    def __emit(self, text: str) -> list[ChudStreamEvent]:
        if not text:
            return []
        return [ChudStreamEvent(channel=self.channel, text=text)]
