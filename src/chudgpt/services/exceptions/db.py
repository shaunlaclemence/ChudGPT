import functools
import json
from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, TypeVar

from sqlalchemy import exc
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.orm import exc as orm_exc

from chudgpt.exceptions import (
    BaseException,
    ChudGPTConflictException,
    ChudGPTDBConfigException,
    ChudGPTInternalServerException,
    ChudGPTNotFoundException,
    ChudGPTServiceUnavailableException,
    ServiceCode,
)
from chudgpt.services.get_db import get_db

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


def to_db_exception(err: Exception) -> BaseException:
    """Narrow any failure to one of the five DB service exceptions.

    The originating error is kept on ``.error`` for diagnosis, so collapsing
    the long tail into ChudGPTDBInternalException loses nothing.
    """
    if isinstance(err, BaseException):
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
