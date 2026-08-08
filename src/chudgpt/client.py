from __future__ import annotations

from typing import Any

from chudgpt.services.db import DBService
from chudgpt.services.files import FilesService
from chudgpt.services.scheduler import SchedulerService

from .providers.gemini import GeminiModel
from .schemas.chat import Message, MessageRole, Response
from .services.rotor import RotorService
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
    """ChudGPT 0.4.1

        from chudgpt import ChudGPT
        from chudgpt.exceptions import ChudGPTRateLimitException

        client = ChudGPT(app_name="my-app")
        reply = await client.chat("Explain monads in one sentence.")

    Keys are read from ``secrets.json`` at your project root. ``app_name``
    namespaces the local database so two apps never share one quota ledger;
    set it before anything touches the db.

    Methods
        await chat(prompt=None, *, messages=None, system=None, builder=None,
                   model=None) -> Response
            Send a turn. Pass exactly one of ``prompt``, ``messages``, or
            ``builder``. ``model`` pins a ``GeminiModel``; omit it to let the
            rotor choose. Usage is recorded against the serving key.

    Attributes
        scheduler -> SchedulerService
            Daily quota reset, fired at midnight America/Los_Angeles. Not
            started for you -- your app owns its lifetime:

                client.scheduler.start()      # idempotent; call on every launch
                client.scheduler.shutdown()   # on exit

            If the app was closed when a reset was due, the next ``start()``
            runs it immediately. Also exposes ``running``, ``is_due``,
            ``last_run``, and ``next_run``.

    Every failure raised from here derives from ``chudgpt.exceptions.BaseException``.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        app_name: str | None = None,
    ):
        files = FilesService()
        files.set_app_name(app_name)

        self._db = DBService(files)
        self._rotor = RotorService(
            db_service=self._db,
            secrets=load_secrets(files.secrets_path()),
            timeout=timeout,
        )
        self.scheduler = SchedulerService(
            controller=self._db, func=self._db.flush_usage
        )

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
