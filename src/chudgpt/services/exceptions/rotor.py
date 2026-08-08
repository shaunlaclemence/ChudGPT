from collections.abc import Callable, Coroutine
from typing import Any, Concatenate, ParamSpec, TypeVar

import openai

from chudgpt.exceptions import ChudGPTRateLimitException

P = ParamSpec("P")
R = TypeVar("R")

MAX_ROTATIONS = 3


def rotor_exception_handler(
    func: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]],
) -> Callable[Concatenate[Any, P], Coroutine[Any, Any, R]]:
    async def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await func(self, *args, **kwargs)
        except openai.RateLimitError as err:
            raise ChudGPTRateLimitException("Rate limit exceeded on AI", err) from err

    return wrapper


def rotor_rotation_handler(
    func: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]],
) -> Callable[Concatenate[Any, P], Coroutine[Any, Any, R]]:
    """Wraps a `rotor_exception_handler`-decorated call: on rate limit, rotate
    to the next provider and retry the same call, up to MAX_ROTATIONS times."""

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
