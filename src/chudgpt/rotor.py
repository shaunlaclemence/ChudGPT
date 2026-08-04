from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import openai

from .db.models import ModelQuota
from .db.quota import QuotaDB, day_key
from .providers.config import PROVIDERS
from .providers.gemini import GeminiModel
from .schemas.chat import Message, Response, Usage
from .utils.keys import Secrets, parse_providers

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


class Rotor:
    def __init__(
        self,
        secrets: Secrets,
        timeout: float = 120.0,
        db: QuotaDB | None = None,
    ):
        self.secrets = parse_providers(secrets)
        for config in PROVIDERS:
            if self.secrets.get(config.name):
                self.config = config
                self.provider = self.secrets[config.name][0]
                break
        else:
            raise ValueError(f"no api_key for any of: {[p.name for p in PROVIDERS]}")
        self._timeout = timeout
        self.db = db if db is not None else QuotaDB()
        self._synced_day: str | None = None

    def _ensure_ready(self, now: datetime | None = None) -> None:
        """Resync providers and reset quotas, lazily: once on first use, then
        again the first time a call lands on a new Pacific day."""
        now = now or datetime.now(UTC)
        today = day_key(now)
        if self._synced_day == today:
            return
        self.db.ensure_ready(self.secrets[self.config.name], now)
        self._synced_day = today

    def __client(self):
        return openai.AsyncOpenAI(
            api_key=self.provider.api_key,
            base_url=self.config.base_url,
            timeout=self._timeout,
        )

    async def usage(self, model: GeminiModel | None = None) -> ModelQuota | None:
        """The stored quota row for the current provider and model.

        @param model: which model's row to read; defaults to the same model
            ``chat()`` would pick, so ``usage()`` describes the next request.
        """
        self._ensure_ready()
        chosen = model or GeminiModel.cheapest()
        return self.db.quota(self.provider.project_number, chosen.slug)

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

        self._ensure_ready()

        ## Prepare request
        if (prompt is None) == (messages is None):
            raise ValueError("pass exactly one of prompt or messages")
        turns: list[Message] = (
            [{"role": "user", "content": prompt}]
            if messages is None
            else list(messages)
        )
        if system is not None:
            turns.insert(0, {"role": "system", "content": system})

        ## Make the API call
        slug = (model or GeminiModel.cheapest()).slug
        async with self.__client() as client:
            result = await client.chat.completions.create(
                model=slug,
                messages=cast("list[ChatCompletionMessageParam]", turns),
            )

        ## Bank what it cost against the requested slug, which is what the
        ## quota row is keyed on -- providers may echo a versioned name back.
        usage = Usage.from_completion(result)
        self.db.record(self.provider.project_number, slug, usage)

        return Response(
            text=result.choices[0].message.content or "",
            service=self.config.name,
            model=result.model,
            usage=usage,
            provider=self.provider,
        )
