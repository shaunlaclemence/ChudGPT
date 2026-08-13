import functools
import subprocess
from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, TypeVar

from chudgpt.exceptions import (
    ChudGPTExecutionTimeoutException,
    ChudGPTInternalServerException,
    ServiceCode,
)

P = ParamSpec("P")
R = TypeVar("R")


def executor_exception_handler(
    func: Callable[Concatenate[Any, P], R],
) -> Callable[Concatenate[Any, P], R]:
    @functools.wraps(func)
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(self, *args, **kwargs)
        except subprocess.TimeoutExpired as err:
            raise ChudGPTExecutionTimeoutException(
                f"code did not finish within {err.timeout:.0f}s", err
            )
        except OSError as err:
            raise ChudGPTInternalServerException(
                "could not start the interpreter subprocess",
                ServiceCode.EXECUTOR_SERVICE,
                err,
            )

    return wrapper
