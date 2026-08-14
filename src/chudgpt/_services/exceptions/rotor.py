import asyncio
import functools
import inspect
from collections.abc import Callable, Coroutine
from typing import Any, Concatenate, ParamSpec, TypeVar, cast

import openai

from chudgpt.exceptions import (
    ChudGPTRateLimitException,
    ChudGPTServiceUnavailableException,
    ChudGPTTimeoutException,
    ServiceCode,
)

P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])

MAX_ROTATIONS = 3
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0


def to_rotor_exception(err):
    if isinstance(err, openai.RateLimitError):
        return ChudGPTRateLimitException("Rate limit exceeded on AI", err)
    if isinstance(err, openai.APITimeoutError):
        return ChudGPTTimeoutException(
            "provider did not answer in time. long audio needs a bigger "
            "timeout, e.g. ChudGPT(timeout=300), or smaller chunks",
            err,
        )
    if isinstance(err, openai.InternalServerError | openai.APIConnectionError):
        return ChudGPTServiceUnavailableException(
            "provider is overloaded or unreachable",
            ServiceCode.ROTOR_SERVICE,
            err,
        )
    return err


def rotor_exception_handler(func: F) -> F:
    if inspect.isasyncgenfunction(func):

        @functools.wraps(func)
        async def stream_wrapper(self, *args, **kwargs):
            try:
                async for item in func(self, *args, **kwargs):
                    yield item
            except Exception as err:
                raise to_rotor_exception(err) from err

        return cast("F", stream_wrapper)

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except Exception as err:
            raise to_rotor_exception(err) from err

    return cast("F", wrapper)


def rotor_retry_handler(
    func: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]],
) -> Callable[Concatenate[Any, P], Coroutine[Any, Any, R]]:
    """Wait out an overloaded model, up to MAX_RETRIES times.

    A 503 is the model being busy, not this key being spent, so rotating would
    only ask the same busy model under another name. Backoff doubles each time.
    Timeouts are deliberately not retried: the wait was too short, and repeating
    the identical call only spends the same quota to reach the same deadline.
    """

    @functools.wraps(func)
    async def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
        for attempt in range(MAX_RETRIES + 1):
            try:
                return await func(self, *args, **kwargs)
            except ChudGPTServiceUnavailableException:
                if attempt == MAX_RETRIES:
                    raise
                await asyncio.sleep(BACKOFF_SECONDS * 2**attempt)
        raise AssertionError("unreachable")

    return wrapper


def rotor_rotation_handler(
    func: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]],
) -> Callable[Concatenate[Any, P], Coroutine[Any, Any, R]]:
    """Wraps a `rotor_exception_handler`-decorated call: on rate limit, rotate
    to the next provider and retry the same call, up to MAX_ROTATIONS times."""

    @functools.wraps(func)
    async def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
        rotations = 0
        while True:
            try:
                return await func(self, *args, **kwargs)
            except ChudGPTRateLimitException:
                if rotations >= MAX_ROTATIONS:
                    raise
                rotations += 1
                self._rotate_provider()

    return wrapper
