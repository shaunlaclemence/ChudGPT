from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from chudgpt._schemas import (
    ChudMessage,
    ChudProvider,
    ChudResponse,
    ChudUsageSummary,
    UsagePeriod,
)
from chudgpt._services.db import DBService
from chudgpt._services.exceptions.main import main_exception_handler
from chudgpt._services.files import FilesService
from chudgpt._services.scheduler import SchedulerService
from chudgpt.exceptions import ChudGPTNotFoundException, ServiceCode
from chudgpt.messages import ChudMessageBuilder

from ._providers.gemini import GeminiModel
from ._services.rotor import RotorService
from ._utils.keys import load_secrets
from ._utils.usage import UsageRules


class ChudGPT:
    """ChudGPT 0.4.2

        from chudgpt import ChudGPT
        from chudgpt.exceptions import ChudGPTRateLimitException

        ChudGPT().initialise(app_name="My App")      # once, creates my-app.db
        client = ChudGPT().app(app_name="My App")    # thereafter, attaches to it
        reply = await client.chat("Explain monads in one sentence.")

    Keys are read from ``secrets.json`` at your project root. The app name is
    kebab-cased into its own database file, so "My App", "my-app" and "My_App"
    all resolve to my-app.db and two apps never share one quota ledger.

    Methods
        await chat(prompt=None, *, messages=None, system=None, builder=None,
                   model=None) -> ChudResponse
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

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout
        self._files = FilesService()
        self.app_name: str | None = None
        self._db: DBService | None = None
        self._rotor: RotorService | None = None
        self.scheduler: SchedulerService | None = None

    ChudUsageResponse = dict[ChudProvider, dict[str, int]]

    @main_exception_handler
    def initialise(self, app_name: str) -> ChudGPT:
        return self.__bind(app_name)

    @main_exception_handler
    def app(self, app_name: str) -> ChudGPT:
        if not self._files.db_exists(app_name):
            raise ChudGPTNotFoundException(
                f"no database for {app_name!r} at "
                f"{self._files.db_path(app_name)}. call initialise() first",
                ServiceCode.DB_SERVICE,
            )
        return self.__bind(app_name)

    def __bind(self, app_name: str) -> ChudGPT:
        self.app_name = app_name
        self._db = DBService(self._files, app_name)
        self._rotor = RotorService(
            db_service=self._db,
            secrets=load_secrets(self._files.secrets_path()),
            timeout=self._timeout,
        )
        self.scheduler = SchedulerService(
            controller=self._db, func=self._db.flush_usage
        )
        return self

    @property
    @main_exception_handler
    def db_path(self) -> Path:
        self.__require()
        return self._files.db_path(str(self.app_name))

    def __require(self) -> RotorService:
        self.__require_db()
        if self._rotor is None:
            raise self.__unbound()
        return self._rotor

    def __require_db(self) -> DBService:
        if self._db is None:
            raise self.__unbound()
        return self._db

    def __unbound(self) -> ChudGPTNotFoundException:
        return ChudGPTNotFoundException(
            "client is not bound to an app. call initialise(app_name) or "
            "app(app_name) first",
            ServiceCode.DB_SERVICE,
        )

    @main_exception_handler
    def get_requests(self, per: UsagePeriod = UsagePeriod.ONE_DAY) -> ChudUsageResponse:
        db = self.__require_db()
        usages = db.get_usage()

        return UsageRules.collate_usage(
            usages, lambda x: UsageRules.is_recent(x.created_at, per.value)
        )

    @main_exception_handler
    def get_tokens(self, per: UsagePeriod = UsagePeriod.ONE_DAY) -> ChudUsageResponse:
        db = self.__require_db()
        usages = db.get_usage()

        return UsageRules.collate_usage(
            usages,
            lambda x: UsageRules.is_recent(x.created_at, per.value),
            "total_tokens",
        )

    @main_exception_handler
    def get_usage_summary(self) -> ChudUsageSummary:
        db = self.__require_db()
        return ChudUsageSummary(usage=db.get_usage(), quotas=db.get_quota())

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
        if builder:
            if prompt is not None or messages is not None or system is not None:
                raise ValueError(
                    "pass builder on its own, not alongside prompt/messages/system"
                )
            return await self.__require().chat(
                messages=builder.messages_list, model=model, **request_kwargs
            )
        else:
            return await self.__require().chat(
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
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_name = schema_name or schema.__name__
            schema = schema.model_json_schema()
        schema_name = schema_name or "response"

        if builder:
            if prompt is not None or messages is not None or system is not None:
                raise ValueError(
                    "pass builder on its own, not alongside prompt/messages/system"
                )
            prompt, messages, system = None, builder.messages_list, None

        response = await self.__require().chat(
            prompt,
            messages=messages,
            system=system,
            model=model,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema},
            },
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
        missing = sorted(set(builders) - set(models))
        if missing:
            raise ValueError(f"no model given for sub-agent(s): {missing}")

        names = list(builders)
        chud_responses = await asyncio.gather(
            *(
                self.__require().chat(
                    messages=builders[name].messages_list,
                    model=models[name],
                    **request_kwargs,
                )
                for name in names
            ),
            return_exceptions=return_exceptions,
        )
        return dict(zip(names, chud_responses))
