from __future__ import annotations

from typing import TYPE_CHECKING, cast

import openai

from chudgpt.db.db_controller import DBController

from .providers.config import PROVIDERS
from .providers.gemini import GeminiModel
from .schemas.chat import Message, MessageRole, Response, Usage
from .utils.keys import Secrets, parse_providers

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


class Rotor:
    def __init__(self, secrets: Secrets, timeout: float = 120.0):
        self.secrets = parse_providers(secrets)
        for config in PROVIDERS:
            if self.secrets.get(config.name):
                self.config = config
                self.provider = self.secrets[config.name][0]
                break
        else:
            raise ValueError(f"no api_key for any of: {[p.name for p in PROVIDERS]}")
        self._timeout = timeout
        self.db_controller = DBController()

    def __client(self):
        return openai.AsyncOpenAI(
            api_key=self.provider.api_key,
            base_url=self.config.base_url,
            timeout=self._timeout,
        )

    async def chat(
        self,
        prompt: str | None = None,
        *,
        messages: list[Message] | None = None,
        system: str | None = None,
        model: GeminiModel | None = None,
    ) -> Response:
        """
        @param prompt: standalone prompt
        @param messages: full turn history, sent verbatim
        @param system: standing instruction prepended as the first turn
        """

        ## Prepare request
        if (prompt is None) == (messages is None):
            raise ValueError("pass exactly one of prompt or messages")
        turns: list[Message] = (
            [Message(role=MessageRole.USER, content=prompt)]
            if messages is None
            else list(messages)
        )
        if system is not None:
            turns.insert(0, Message(role=MessageRole.SYSTEM, content=system))

        ## Make the API call
        slug = (model or GeminiModel.cheapest()).slug
        payload = [{"role": t.role.value, "content": t.content} for t in turns]
        async with self.__client() as client:
            result = await client.chat.completions.create(
                model=slug,
                messages=cast("list[ChatCompletionMessageParam]", payload),
            )

        ## Record usage
        usage = Usage.from_completion(result)
        self.db_controller.create_usage_record(usage, slug)

        return Response(
            text=result.choices[0].message.content or "",
            service=self.config.name,
            model=result.model,
            usage=usage,
            provider=self.provider,
        )
