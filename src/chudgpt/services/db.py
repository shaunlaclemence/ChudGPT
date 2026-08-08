import functools
import json
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import resources
from typing import Any, Concatenate, ParamSpec, TypeVar

from sqlalchemy import Engine, create_engine, delete, exc, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm import exc as orm_exc

from chudgpt.db.models import Base, Meta, ModelQuota, ModelUsage, Provider
from chudgpt.exceptions import (
    ChudGPTConflictException,
    ChudGPTDBConfigException,
    ChudGPTInternalServerException,
    ChudGPTNotFoundException,
    ChudGPTServiceUnavailableException,
    DBServiceException,
    ServiceCode,
)
from chudgpt.schemas.chat import Usage, mask_key
from chudgpt.services.files import FilesService

UNAVAILABLE = (
    exc.OperationalError,
    exc.InterfaceError,
    exc.DisconnectionError,
    exc.TimeoutError,
)
CONFLICT = (
    exc.IntegrityError,
    exc.MultipleResultsFound,
    orm_exc.StaleDataError,
    orm_exc.ObjectDeletedError,
)
CONFIG = (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError)

_session_factory: sessionmaker | None = None


def configure_session_factory(engine: Engine) -> None:
    global _session_factory
    _session_factory = sessionmaker(engine)


def get_db() -> Session:
    """A ready-to-use session. Caller owns commit/rollback/close."""
    if _session_factory is None:
        raise ChudGPTInternalServerException(
            "db used before initialisation", ServiceCode.DB_SERVICE
        )
    return _session_factory()


def to_db_exception(err: Exception) -> DBServiceException:
    """Narrow any failure to one of the five DB service exceptions.

    The originating error is kept on ``.error`` for diagnosis, so collapsing
    the long tail into ChudGPTDBInternalException loses nothing.
    """
    if isinstance(err, DBServiceException):
        return err
    if isinstance(err, exc.NoResultFound):
        return ChudGPTNotFoundException(
            "record does not exist", ServiceCode.DB_SERVICE, err
        )
    if isinstance(err, CONFLICT):
        return ChudGPTConflictException(
            "write rejected: constraint violated, duplicate row, or row changed "
            "by another transaction",
            ServiceCode.DB_SERVICE,
            err,
        )
    if isinstance(err, UNAVAILABLE):
        return ChudGPTServiceUnavailableException(
            "database unreachable: locked, unopenable, or the connection dropped",
            ServiceCode.DB_SERVICE,
            err,
        )
    if isinstance(err, CONFIG):
        return ChudGPTDBConfigException(
            "config or secrets JSON is invalid, or missing a required field", err
        )
    return ChudGPTInternalServerException(
        "unhandled database failure", ServiceCode.DB_SERVICE, err
    )


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
            raise to_db_exception(err) from err
        finally:
            db.close()
        return result

    return wrapper


def db_source_exception_handler(
    func: Callable[Concatenate[Any, P], R],
) -> Callable[Concatenate[Any, P], R]:
    """For methods that own no session: engine setup and config loading."""

    @functools.wraps(func)
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(self, *args, **kwargs)
        except (SQLAlchemyError, *CONFIG) as err:
            raise to_db_exception(err) from err

    return wrapper


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
        db.execute(delete(ModelUsage))

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
