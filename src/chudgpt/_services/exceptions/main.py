import functools
import inspect
from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, TypeVar

from chudgpt.exceptions import BaseException, ChudGPTInternalServerException

P = ParamSpec("P")
R = TypeVar("R")


def to_main_exception(err: Exception) -> BaseException:
    if isinstance(err, BaseException):
        return err
    return ChudGPTInternalServerException("unhandled client failure", error=err)


def main_exception_handler(
    func: Callable[Concatenate[Any, P], R],
) -> Callable[Concatenate[Any, P], R]:
    # the client exposes both sync and async methods, and awaiting inside a
    # sync wrapper would hand back a coroutine that fails outside the guard
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(self, *args: P.args, **kwargs: P.kwargs):
            try:
                return await func(self, *args, **kwargs)
            except Exception as err:
                raise to_main_exception(err) from err

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(self, *args, **kwargs)
        except Exception as err:
            raise to_main_exception(err) from err

    return wrapper
