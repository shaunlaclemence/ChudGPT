import json
from datetime import UTC, datetime
from importlib import resources

from sqlalchemy import create_engine, delete, inspect, select
from sqlalchemy.orm import Session, joinedload

from chudgpt._db.models import Base, Meta, ModelQuota, ModelUsage, Provider
from chudgpt._schemas import ChudProvider, ChudQuota, ChudUsageRecord, Usage, mask_key
from chudgpt._services.exceptions.db import (
    db_exception_handler,
    db_source_exception_handler,
)
from chudgpt._services.files import FilesService
from chudgpt._services.get_db import configure_session_factory
from chudgpt._utils.usage import UsageRules


class DBService:
    def __init__(self, files_service: FilesService, app_name: str) -> None:
        self.__files = files_service
        self.app_name = app_name
        self.__create_all()
        self.__init_providers()
        self.__init_quotas()

    @db_source_exception_handler
    def __create_all(self):
        self.__files.init_store(self.app_name)
        path = self.__files.db_path(self.app_name).resolve()
        self.engine = create_engine(f"sqlite:///{path}")
        self.__drop_outdated_usage()
        Base.metadata.create_all(self.engine)
        configure_session_factory(self.engine)

    def __drop_outdated_usage(self) -> None:
        # usage is disposable and flushed daily, so rebuilding beats backfilling
        inspector = inspect(self.engine)
        if not inspector.has_table(ModelUsage.__tablename__):
            return
        columns = {c["name"] for c in inspector.get_columns(ModelUsage.__tablename__)}
        if not columns >= {"quota_id", "provider_id"}:
            ModelUsage.__table__.drop(self.engine)

    def __replace_all(self, db: Session, model, rows: list) -> None:
        db.execute(delete(model))
        db.add_all(rows)

    @db_source_exception_handler
    def __gemini_providers(self) -> list[Provider]:
        providers = self.__files.json_to_dict(self.__files.secrets_path())
        return [
            Provider(
                email=p["account"],
                name=p["name"],
                project_name=p["project_name"],
                project_number=str(p["project_number"]),
                api_key=mask_key(p["api_key"]),
            )
            for p in providers["gemini"]
        ]

    @db_source_exception_handler
    def __gemini_quotas(self) -> list[ModelQuota]:
        configs = json.loads(
            resources.files("chudgpt")
            .joinpath("config.json")
            .read_text(encoding="utf-8")
        )
        return [
            ModelQuota(
                model=q["slug"],
                rpd=q["rpd"],
                rpm=q["rpm"],
                tpm=q["tpm"],
                inputs=",".join(q["inputs"]),
            )
            for q in configs["gemini"].values()
        ]

    @db_exception_handler
    def __init_providers(self, db: Session):
        self.__replace_all(db, Provider, self.__gemini_providers())

    @db_exception_handler
    def __init_quotas(self, db: Session):
        self.__replace_all(db, ModelQuota, self.__gemini_quotas())

    @db_exception_handler
    def flush_usage(self, db: Session):
        # TODO: offload data to save it for analytics
        db.execute(delete(ModelUsage))

    @db_exception_handler
    def create_usage_record(
        self, db: Session, usage: Usage, model: str, project_number: str
    ) -> None:
        quota = db.execute(
            select(ModelQuota).where(ModelQuota.model == model)
        ).scalar_one()
        provider = db.execute(
            select(Provider).where(Provider.project_number == project_number)
        ).scalar_one()
        db.add(
            ModelUsage(
                quota_id=quota.id,
                provider_id=provider.id,
                created_at=datetime.now(UTC),
                prompt_tokens=usage.prompt,
                completion_tokens=usage.completion,
                total_tokens=usage.total,
            )
        )

    @db_exception_handler
    def get_meta(self, db: Session, key: str) -> str | None:
        row = db.get(Meta, key)
        return row.value if row else None

    @db_exception_handler
    def set_meta(self, db: Session, key: str, value: str) -> None:
        row = db.get(Meta, key)
        if row:
            row.value = value
        else:
            db.add(Meta(key=key, value=value))

    @db_exception_handler
    def get_usage(self, db: Session) -> list[ChudUsageRecord]:
        rows = db.scalars(
            select(ModelUsage).options(
                joinedload(ModelUsage.quota), joinedload(ModelUsage.provider)
            )
        )
        return [
            ChudUsageRecord(
                model=row.quota.model,
                provider=ChudProvider.model_validate(row.provider),
                created_at=UsageRules.as_utc(row.created_at),
                prompt_tokens=row.prompt_tokens,
                completion_tokens=row.completion_tokens,
                reasoning_tokens=row.reasoning_tokens,
                total_tokens=row.total_tokens,
            )
            for row in rows
        ]

    @db_exception_handler
    def get_quota(self, db: Session) -> list[ChudQuota]:
        return [
            ChudQuota(
                slug=row.model,
                rpd=row.rpd,
                rpm=row.rpm,
                tpm=row.tpm,
                inputs=row.allowed_inputs,
            )
            for row in db.scalars(select(ModelQuota))
        ]
