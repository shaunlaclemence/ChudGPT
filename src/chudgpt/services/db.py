import functools
import json
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import resources
from typing import Any, Concatenate, ParamSpec, TypeVar

from sqlalchemy import Engine, create_engine, delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from chudgpt.db.models import Base, Meta, ModelQuota, ModelUsage, Provider
from chudgpt.schemas.chat import Usage, mask_key
from chudgpt.services.files import FilesService

_session_factory: sessionmaker | None = None


def configure_session_factory(engine: Engine) -> None:
    global _session_factory
    _session_factory = sessionmaker(engine)


def get_db() -> Session:
    """A ready-to-use session. Caller owns commit/rollback/close."""
    if _session_factory is None:
        raise ValueError("db not initialised")
    return _session_factory()


P = ParamSpec("P")
R = TypeVar("R")


def db_exception_handler(
    func: Callable[Concatenate[Any, Session, P], R],
) -> Callable[Concatenate[Any, P], R]:
    @functools.wraps(func)
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
        db = get_db()
        try:
            result = func(self, db, *args, **kwargs)
            db.commit()
        except SQLAlchemyError as err:
            db.rollback()
            raise ValueError(f"DB Error: {err}") from err
        finally:
            db.close()
        return result

    return wrapper


class DBService:
    def __init__(self, files_service: FilesService) -> None:
        self.__files = files_service
        self.__create_all()
        self.__init_providers()
        self.__init_quotas()

    def __create_all(self):
        self.__files.init_store()
        self.engine = create_engine(f"sqlite:///{self.__files.db_path().resolve()}")
        Base.metadata.create_all(self.engine)
        configure_session_factory(self.engine)

    def __replace_all(self, db: Session, model, rows: list) -> None:
        db.execute(delete(model))
        db.add_all(rows)

    @db_exception_handler
    def __init_providers(self, db: Session):
        secrets_path = self.__files.secrets_path()
        providers = self.__files.json_to_dict(secrets_path)
        gemini_providers = [
            Provider(
                email=p["account"],
                name=p["name"],
                project_name=p["project_name"],
                project_number=str(p["project_number"]),
                api_key=mask_key(p["api_key"]),
            )
            for p in providers["gemini"]
        ]
        self.__replace_all(db, Provider, gemini_providers)

    @db_exception_handler
    def __init_quotas(self, db: Session):
        configs = json.loads(
            resources.files("chudgpt")
            .joinpath("config.json")
            .read_text(encoding="utf-8")
        )
        gemini_configs = [
            ModelQuota(
                model=q["slug"],
                rpd=q["rpd"],
                rpm=q["rpm"],
                tpm=q["tpm"],
                inputs=",".join(q["inputs"]),
            )
            for q in configs["gemini"].values()
        ]
        self.__replace_all(db, ModelQuota, gemini_configs)

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
