from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from chudgpt._providers.gemini import GeminiModel
from chudgpt._schemas import ChudMessage, ChudResponse, ChudStreamEvent
from chudgpt._services.exceptions.main import main_exception_handler
from chudgpt._services.rotor import RotorService
from chudgpt._utils.chat import ChatRules
from chudgpt.messages import ChudMessageBuilder


class TextService:
    def __init__(self, rotor: RotorService) -> None:
        self._rotor = rotor

    @main_exception_handler
    async def chat(
        self,
        prompt: str | None = None,
        *,
        messages: list[ChudMessage] | None = None,
        system: str | None = None,
        builder: ChudMessageBuilder | None = None,
        model: GeminiModel | None = None,
        **request_kwargs: Any,
    ) -> ChudResponse:
        prompt, messages, system = ChatRules.turn(prompt, messages, system, builder)
        return await self._rotor.chat(
            prompt,
            messages=messages,
            system=system,
            model=model,
            **request_kwargs,
        )

    @main_exception_handler
    async def chat_json(
        self,
        prompt: str | None = None,
        *,
        schema: type[BaseModel] | dict[str, Any],
        messages: list[ChudMessage] | None = None,
        system: str | None = None,
        builder: ChudMessageBuilder | None = None,
        model: GeminiModel | None = None,
        schema_name: str | None = None,
        **request_kwargs: Any,
    ) -> dict[str, Any]:
        prompt, messages, system = ChatRules.turn(prompt, messages, system, builder)
        response = await self._rotor.chat(
            prompt,
            messages=messages,
            system=system,
            model=model,
            response_format=ChatRules.response_format(schema, schema_name),
            **request_kwargs,
        )
        return response.parsed_json

    @main_exception_handler
    async def parallel_chat(
        self,
        builders: dict[str, ChudMessageBuilder],
        models: dict[str, GeminiModel],
        *,
        return_exceptions: bool = False,
        **request_kwargs: Any,
    ) -> dict[str, ChudResponse]:
        ChatRules.guard_models(builders, models)
        names = list(builders)
        responses = await asyncio.gather(
            *(
                self._rotor.chat(
                    messages=builders[name].messages_list,
                    model=models[name],
                    **request_kwargs,
                )
                for name in names
            ),
            return_exceptions=return_exceptions,
        )
        return dict(zip(names, responses))

    @main_exception_handler
    def stream(
        self,
        prompt: str | None = None,
        *,
        messages: list[ChudMessage] | None = None,
        system: str | None = None,
        builder: ChudMessageBuilder | None = None,
        model: GeminiModel | None = None,
        think: bool = False,
        **request_kwargs: Any,
    ) -> AsyncIterator[ChudStreamEvent]:
        prompt, messages, system = ChatRules.turn(prompt, messages, system, builder)
        return self._rotor.stream(
            prompt,
            messages=messages,
            system=system,
            model=model,
            think=think,
            **request_kwargs,
        )

    @main_exception_handler
    def parallel_stream(
        self,
        builders: dict[str, ChudMessageBuilder],
        models: dict[str, GeminiModel],
        *,
        think: bool = False,
        return_exceptions: bool = True,
        **request_kwargs: Any,
    ) -> AsyncIterator[tuple[str, ChudStreamEvent]]:
        ChatRules.guard_models(builders, models)
        return self._rotor.parallel_stream(
            builders,
            models,
            think=think,
            return_exceptions=return_exceptions,
            **request_kwargs,
        )
