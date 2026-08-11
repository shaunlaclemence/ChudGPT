import functools
import json
import sqlite3
from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, TypeVar

from chudgpt.exceptions import (
    ChudGPTBadDataException,
    ChudGPTConflictException,
    ChudGPTForbiddenException,
    ChudGPTInternalServerException,
    ChudGPTInvalidPathException,
    ChudGPTNotFoundException,
    ChudGPTServiceUnavailableException,
    ServiceCode,
)

P = ParamSpec("P")
R = TypeVar("R")


def file_exception_handler(
    func: Callable[Concatenate[Any, P], R],
) -> Callable[Concatenate[Any, P], R]:
    @functools.wraps(func)
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(self, *args, **kwargs)
        except FileNotFoundError as err:
            raise ChudGPTNotFoundException(
                f"file not found: {err.filename or 'unknown path'}",
                ServiceCode.FILE_SERVICE,
                err,
            )
        except PermissionError as err:
            raise ChudGPTForbiddenException(
                "OS denied read access; Restrictive file permissions or a windows "
                "process holding an external lock",
                ServiceCode.FILE_SERVICE,
                err,
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            raise ChudGPTBadDataException(
                "Corrupted, malformed or invalid JSON", ServiceCode.FILE_SERVICE, err
            )
        except (IsADirectoryError, NotADirectoryError) as err:
            raise ChudGPTInvalidPathException(
                "Resolved path points to a directory or an invalid path segment, "
                "not a readable file",
                err,
            )
        except FileExistsError as err:
            raise ChudGPTConflictException(
                "A file already occupies the target path, blocking directory creation",
                ServiceCode.FILE_SERVICE,
                err,
            )
        except sqlite3.OperationalError as err:
            raise ChudGPTServiceUnavailableException(
                "Unable to open the sqlite database file; check the path, disk "
                "space, and permissions",
                ServiceCode.FILE_SERVICE,
                err,
            )
        except OSError as err:
            raise ChudGPTInternalServerException(
                service_code=ServiceCode.FILE_SERVICE, error=err
            )

    return wrapper
