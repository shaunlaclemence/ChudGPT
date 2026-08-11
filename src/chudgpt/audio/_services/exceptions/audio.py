import contextlib
import functools
import inspect
from collections.abc import Callable, Iterator
from typing import Any, Concatenate, ParamSpec, TypeVar

from chudgpt.exceptions import (
    ChudGPTBadDataException,
    ChudGPTForbiddenException,
    ChudGPTInternalServerException,
    ChudGPTInvalidPathException,
    ChudGPTNotFoundException,
    ServiceCode,
)

P = ParamSpec("P")
R = TypeVar("R")


@contextlib.contextmanager
def audio_errors() -> Iterator[None]:
    try:
        yield
    except FileNotFoundError as err:
        raise ChudGPTNotFoundException(
            f"audio file not found: {err.filename or 'unknown path'}",
            ServiceCode.AUDIO_SERVICE,
            err,
        )
    except PermissionError as err:
        raise ChudGPTForbiddenException(
            "OS denied read access to the audio file",
            ServiceCode.AUDIO_SERVICE,
            err,
        )
    except (IsADirectoryError, NotADirectoryError) as err:
        raise ChudGPTInvalidPathException(
            "Resolved path points to a directory or an invalid path segment, "
            "not a readable audio file",
            err,
        )
    # libsndfile errors subclass RuntimeError, so decode failures land here
    # without this module importing soundfile
    except (RuntimeError, ValueError, TypeError) as err:
        raise ChudGPTBadDataException(
            "Unreadable, corrupt, or unsupported audio data",
            ServiceCode.AUDIO_SERVICE,
            err,
        )
    except MemoryError as err:
        raise ChudGPTInternalServerException(
            "Ran out of memory decoding audio; use a smaller chunk_seconds",
            ServiceCode.AUDIO_SERVICE,
            err,
        )
    except OSError as err:
        raise ChudGPTInternalServerException(
            service_code=ServiceCode.AUDIO_SERVICE, error=err
        )


def audio_exception_handler(
    func: Callable[Concatenate[Any, P], R],
) -> Callable[Concatenate[Any, P], R]:
    # a decorated generator only enters its body on first next(), so the
    # guard has to span iteration rather than the call that builds it
    if inspect.isgeneratorfunction(func):

        @functools.wraps(func)
        def generator_wrapper(self, *args: P.args, **kwargs: P.kwargs):
            with audio_errors():
                yield from func(self, *args, **kwargs)

        return generator_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
        with audio_errors():
            return func(self, *args, **kwargs)

    return wrapper
