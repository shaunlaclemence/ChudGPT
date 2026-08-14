from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from chudgpt._schemas import ChudProvider, ChudUsageSummary, UsagePeriod
from chudgpt._services.db import DBService
from chudgpt._services.exceptions.main import main_exception_handler
from chudgpt._services.files import FilesService
from chudgpt._services.scheduler import SchedulerService
from chudgpt._services.text import TextService
from chudgpt.exceptions import ChudGPTNotFoundException, ServiceCode

from ._services.rotor import RotorService
from ._utils.keys import load_secrets
from ._utils.plugins import PluginRegistry
from ._utils.usage import UsageRules
from ._utils.version import VersionRules

if TYPE_CHECKING:
    from chudgpt.audio._main import AudioService


class ChudGPT:
    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout
        self._files = FilesService()
        self._plugins = PluginRegistry()
        self.app_name: str | None = None
        self._db: DBService | None = None
        self._text: TextService | None = None
        self.scheduler: SchedulerService | None = None

    ChudUsageResponse = dict[ChudProvider, dict[str, int]]

    if TYPE_CHECKING:
        audio: AudioService

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
        self._text = TextService(
            RotorService(
                db_service=self._db,
                secrets=load_secrets(self._files.secrets_path()),
                timeout=self._timeout,
            )
        )
        self.scheduler = SchedulerService(
            controller=self._db, func=self._db.flush_usage
        )
        for name, service in self._plugins.attach(self).items():
            setattr(self, name, service)
        return self

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(PluginRegistry.missing(name))

    @property
    def version(self) -> str:
        return VersionRules.installed()

    @property
    def text(self) -> TextService:
        if self._text is None:
            raise self.__unbound()
        return self._text

    @property
    @main_exception_handler
    def db_path(self) -> Path:
        self.__require_db()
        return self._files.db_path(str(self.app_name))

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
        usages = self.__require_db().get_usage()

        return UsageRules.collate_usage(
            usages, lambda x: UsageRules.is_recent(x.created_at, per.value)
        )

    @main_exception_handler
    def get_tokens(self, per: UsagePeriod = UsagePeriod.ONE_DAY) -> ChudUsageResponse:
        usages = self.__require_db().get_usage()

        return UsageRules.collate_usage(
            usages,
            lambda x: UsageRules.is_recent(x.created_at, per.value),
            "total_tokens",
        )

    @main_exception_handler
    def get_usage_summary(self) -> ChudUsageSummary:
        db = self.__require_db()
        return ChudUsageSummary(usage=db.get_usage(), quotas=db.get_quota())
