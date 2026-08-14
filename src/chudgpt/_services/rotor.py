from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

import openai

from chudgpt._schemas import (
    ChudChannel,
    ChudMessage,
    ChudMessageRole,
    ChudProvider,
    ChudResponse,
    ChudStreamEvent,
    Usage,
)
from chudgpt._services.db import DBService
from chudgpt._services.exceptions.rotor import (
    rotor_exception_handler,
    rotor_retry_handler,
    rotor_rotation_handler,
)
from chudgpt._services.timer import timer
from chudgpt.messages import ChudMessageBuilder

from .._providers.config import PROVIDER_CONFIGS
from .._providers.gemini import GeminiModel
from .._utils.keys import Secrets, parse_providers
from .._utils.stream import StreamRules

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


class RotorService:
    def __init__(self, db_service: DBService, secrets: Secrets, timeout: float = 120.0):
        self.secrets = parse_providers(secrets)

        self.provider_config = PROVIDER_CONFIGS[0]
        self.provider_list = self.secrets[self.provider_config.name]

        self.provider_index = 0

        self._timeout = timeout
        self.db_service = db_service

    def __client(self):
        return openai.AsyncOpenAI(
            api_key=self._provider().api_key,
            base_url=self.provider_config.base_url,
            timeout=self._timeout,
        )

    def _rotate_provider(self):
        i = self.provider_index

        if i < len(self.provider_list) - 1:
            i += 1
        elif i == 0:
            raise ValueError("No additional providers available")
        else:
            i = 0
        self.provider_index = i

    def _provider(self) -> ChudProvider:
        return self.provider_list[self.provider_index]

    def __request(
        self,
        prompt: str | None,
        messages: list[ChudMessage] | None,
        system: str | None,
        model: GeminiModel | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        if (prompt is None) == (messages is None):
            raise ValueError("pass exactly one of prompt or messages")
        turns = (
            [ChudMessage(role=ChudMessageRole.USER, content=prompt)]
            if messages is None
            else list(messages)
        )
        if system is not None:
            turns.insert(0, ChudMessage(role=ChudMessageRole.SYSTEM, content=system))
        return (
            (model or GeminiModel.cheapest()).slug,
            [{"role": t.role.value, "content": t.content} for t in turns],
        )

    @rotor_exception_handler
    async def stream(
        self,
        prompt: str | None = None,
        *,
        messages: list[ChudMessage] | None = None,
        system: str | None = None,
        model: GeminiModel | None = None,
        think: bool = False,
        **request_kwargs: Any,
    ) -> AsyncIterator[ChudStreamEvent]:
        slug, payload = self.__request(prompt, messages, system, model)
        if think:
            request_kwargs["extra_body"] = {
                "extra_body": {
                    "google": {"thinking_config": {"include_thoughts": True}}
                }
            }

        rules = StreamRules()
        answer = []
        usage = Usage(requests=1)
        served = slug

        with timer() as timing:
            async with self.__client() as client:
                chunks = await client.chat.completions.create(
                    model=slug,
                    messages=cast("list[ChatCompletionMessageParam]", payload),
                    stream=True,
                    stream_options={"include_usage": True},
                    **request_kwargs,
                )
                async for chunk in chunks:
                    if chunk.usage:
                        usage = Usage.from_completion(chunk)
                    served = chunk.model or served
                    for choice in chunk.choices:
                        for event in rules.feed(choice.delta.content or ""):
                            if event.channel is ChudChannel.ANSWER:
                                answer.append(event.text)
                            yield event
                for event in rules.flush():
                    if event.channel is ChudChannel.ANSWER:
                        answer.append(event.text)
                    yield event

        self.db_service.create_usage_record(
            usage, slug, self._provider().project_number
        )
        yield ChudStreamEvent(
            channel=ChudChannel.DONE,
            response=ChudResponse(
                data="".join(answer),
                service=self.provider_config.name,
                model=served,
                usage=usage,
                provider=self._provider(),
                duration=timing.duration,
            ),
        )

    @rotor_exception_handler
    async def parallel_stream(
        self,
        builders: dict[str, ChudMessageBuilder],
        models: dict[str, GeminiModel],
        *,
        think: bool = False,
        return_exceptions: bool = True,
        **request_kwargs: Any,
    ) -> AsyncIterator[tuple[str, ChudStreamEvent]]:
        missing = sorted(set(builders) - set(models))
        if missing:
            raise ValueError(f"no model given for sub-agent(s): {missing}")

        queue: asyncio.Queue[tuple[str, ChudStreamEvent | Exception]] = asyncio.Queue()

        async def pump(name: str) -> None:
            try:
                async for event in self.stream(
                    messages=builders[name].messages_list,
                    model=models[name],
                    think=think,
                    **request_kwargs,
                ):
                    await queue.put((name, event))
            except Exception as err:  # noqa: BLE001
                await queue.put((name, err))

        tasks = [asyncio.create_task(pump(name)) for name in builders]
        pending = len(tasks)
        try:
            while pending:
                name, item = await queue.get()
                if isinstance(item, Exception):
                    pending -= 1
                    if not return_exceptions:
                        raise item
                    yield (
                        name,
                        ChudStreamEvent(channel=ChudChannel.ERROR, text=str(item)),
                    )
                    continue
                yield name, item
                if item.channel is ChudChannel.DONE:
                    pending -= 1
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    @rotor_rotation_handler
    @rotor_retry_handler
    @rotor_exception_handler
    async def chat(
        self,
        prompt: str | None = None,
        *,
        messages: list[ChudMessage] | None = None,
        system: str | None = None,
        model: GeminiModel | None = None,
        **request_kwargs: Any,
    ) -> ChudResponse:
        """
        @param prompt: standalone prompt
        @param messages: full turn history, sent verbatim
        @param system: standing instruction prepended as the first turn
        @param request_kwargs: passed straight to the provider call, e.g.
            response_format, temperature, tools
        """

        ## Prepare request
        slug, payload = self.__request(prompt, messages, system, model)

        ## Make the API call
        with timer() as timing:
            async with self.__client() as client:
                result = await client.chat.completions.create(
                    model=slug,
                    messages=cast("list[ChatCompletionMessageParam]", payload),
                    **request_kwargs,
                )

        ## Record usage
        usage = Usage.from_completion(result)
        self.db_service.create_usage_record(
            usage, slug, self._provider().project_number
        )

        return ChudResponse(
            data=result.choices[0].message.content or "",
            service=self.provider_config.name,
            model=result.model,
            usage=usage,
            provider=self._provider(),
            duration=timing.duration,
        )
