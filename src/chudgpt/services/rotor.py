from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import openai

from chudgpt.schemas.chat import Provider
from chudgpt.services.db import DBService
from chudgpt.services.exceptions.rotor import (
    rotor_exception_handler,
    rotor_rotation_handler,
)
from chudgpt.services.timer import timer

from ..providers.config import PROVIDER_CONFIGS
from ..providers.gemini import GeminiModel
from ..schemas.chat import ChudMessage, ChudMessageRole, ChudResponse, Usage
from ..utils.keys import Secrets, parse_providers

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

    def _provider(self) -> Provider:
        return self.provider_list[self.provider_index]

    @rotor_rotation_handler
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
        if (prompt is None) == (messages is None):
            raise ValueError("pass exactly one of prompt or messages")
        turns: list[ChudMessage] = (
            [ChudMessage(role=ChudMessageRole.USER, content=prompt)]
            if messages is None
            else list(messages)
        )
        if system is not None:
            turns.insert(0, ChudMessage(role=ChudMessageRole.SYSTEM, content=system))

        ## Make the API call
        slug = (model or GeminiModel.cheapest()).slug
        payload = [{"role": t.role.value, "content": t.content} for t in turns]
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
            text=result.choices[0].message.content or "",
            service=self.provider_config.name,
            model=result.model,
            usage=usage,
            provider=self._provider(),
            duration=timing.duration,
        )
