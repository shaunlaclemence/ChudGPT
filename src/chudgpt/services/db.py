import json
from datetime import UTC, datetime
from importlib import resources

from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from chudgpt.db.models import Base, Meta, ModelQuota, ModelUsage, Provider
from chudgpt.schemas.chat import Usage, mask_key
from chudgpt.services.exceptions.db import (
    db_exception_handler,
    db_source_exception_handler,
)
from chudgpt.services.files import FilesService
from chudgpt.services.get_db import configure_session_factory


class DBService:
    def __init__(self, files_service: FilesService) -> None:
        self.__files = files_service
        self.__create_all()
        self.__init_providers()
        self.__init_quotas()

    @db_source_exception_handler
    def __create_all(self):
        self.__files.init_store()
        self.engine = create_engine(f"sqlite:///{self.__files.db_path().resolve()}")
        Base.metadata.create_all(self.engine)
        configure_session_factory(self.engine)

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
        db.execute(delete(ModelQuota))

    @db_exception_handler
    def create_usage_record(self, db: Session, usage: Usage, model: str) -> None:
        quota = db.execute(
            select(ModelQuota).where(ModelQuota.model == model)
        ).scalar_one()
        db.add(
            ModelUsage(
                quota_id=quota.id,
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
