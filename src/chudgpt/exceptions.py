"""Every exception ChudGPT can raise.

    from chudgpt.exceptions import ChudGPTNotFoundException

Codes read ``<service_code>-<error_code>``, e.g. "002-404":

    001 FILE_SERVICE   002 DB_SERVICE   003 ROTOR_SERVICE
    004 AUDIO_SERVICE  999 UNKOWN_SERVICE

Catch ``BaseException`` for anything from this library, a service base
(``FileServiceException``, ``DBServiceException``, ``RotorServiceException``)
for one subsystem, or a leaf class for one condition.
"""

from enum import Enum
from typing import Any


class ServiceCode(str, Enum):
    """Which service a failure came from; the first half of an error code."""

    FILE_SERVICE = "001"
    DB_SERVICE = "002"
    ROTOR_SERVICE = "003"
    AUDIO_SERVICE = "004"
    UNKOWN_SERVICE = "999"


class BaseException(Exception):
    """Root of every ChudGPT error.

    Full code reads ``<service_code>-<error_code>`` (e.g. "002-404"). The
    originating exception, when there is one, is kept on ``.error``.
    """

    def __init__(
        self,
        message: str | None,
        error_code: str = "999",
        service_code: ServiceCode = ServiceCode.UNKOWN_SERVICE,
        error: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.service_code = service_code
        self.error = error


### SHARED EXCEPTIONS ###


class ChudGPTInternalServerException(BaseException):
    """500 Internal Server Error -- unhandled failure; treat as a bug.

    error_code "500"; service_code set by the raising service.
    """

    def __init__(
        self,
        message: str | None = None,
        service_code: ServiceCode = ServiceCode.UNKOWN_SERVICE,
        error: Any | None = None,
    ) -> None:
        super().__init__(message, "500", service_code, error)


class ChudGPTServiceUnavailableException(BaseException):
    """503 Service Unavailable -- the resource is unreachable; retry later.

    error_code "503"; service_code set by the raising service.
    """

    def __init__(
        self, message: str, service_code: ServiceCode, error: Any | None = None
    ) -> None:
        super().__init__(message, "503", service_code, error)


class ChudGPTConflictException(BaseException):
    """409 Conflict -- the write was rejected by a constraint or a duplicate.

    error_code "409"; service_code set by the raising service.
    """

    def __init__(
        self, message: str, service_code: ServiceCode, error: Any | None = None
    ) -> None:
        super().__init__(message, "409", service_code, error)


class ChudGPTNotFoundException(BaseException):
    """404 Not Found -- the requested record, file, or resource does not exist.

    error_code "404"; service_code set by the raising service.
    """

    def __init__(
        self, message: str, service_code: ServiceCode, error: Any | None = None
    ) -> None:
        super().__init__(message, "404", service_code, error)


class ChudGPTForbiddenException(BaseException):
    """403 Forbidden -- access was denied, usually by file permissions or a lock.

    error_code "403"; service_code set by the raising service.
    """

    def __init__(
        self, message: str, service_code: ServiceCode, error: Any | None = None
    ) -> None:
        super().__init__(message, "403", service_code, error)


class ChudGPTUnauthorizedException(BaseException):
    """401 Unauthorized -- credentials are missing, invalid, or rejected.

    error_code "401"; service_code set by the raising service.
    """

    def __init__(
        self, message: str, service_code: ServiceCode, error: Any | None = None
    ) -> None:
        super().__init__(message, "401", service_code, error)


class ChudGPTBadDataException(BaseException):
    """400 Bad Data -- the payload is corrupt, malformed, or the wrong type.

    error_code "400"; service_code set by the raising service.
    """

    def __init__(
        self, message: str, service_code: ServiceCode, error: Any | None = None
    ) -> None:
        super().__init__(message, "400", service_code, error)


#### ####

### FILE SERVICE EXCEPTIONS ####


class FileServiceException(BaseException):
    """Base for every file service failure. service_code "001" (FILE_SERVICE)."""

    def __init__(
        self, message: str | None, error_code: str = "999", error: Any | None = None
    ) -> None:
        super().__init__(message, error_code, ServiceCode.FILE_SERVICE, error)


class ChudGPTInvalidPathException(FileServiceException):
    """422 Invalid Path -- the path is a directory or an unusable path segment.

    Full code "001-422".
    """

    def __init__(self, message: str | None = None, error: Any | None = None) -> None:
        super().__init__(message, "422", error)


#### ####

### DB SERVICE EXCEPTION ###


class DBServiceException(BaseException):
    """Base for every database failure. service_code "002" (DB_SERVICE)."""

    def __init__(
        self, message: str | None, error_code: str = "999", error: Any | None = None
    ) -> None:
        super().__init__(message, error_code, ServiceCode.DB_SERVICE, error)


class ChudGPTDBConfigException(DBServiceException):
    """422 Bad Config -- config or secrets JSON is invalid or missing a field.

    Full code "002-422".
    """

    def __init__(self, message: str | None = None, error: Any | None = None) -> None:
        super().__init__(message, "422", error)


#### ####

### ROTOR SERVICE EXCEPTION ###


class RotorServiceException(BaseException):
    """Base for every provider/rotation failure. service_code "003" (ROTOR_SERVICE)."""

    def __init__(
        self, message: str | None, error_code: str = "999", error: Any | None = None
    ) -> None:
        super().__init__(message, error_code, ServiceCode.ROTOR_SERVICE, error)


class ChudGPTRateLimitException(RotorServiceException):
    """429 Rate Limited -- the provider refused the request; quota or RPM exhausted.

    Full code "003-429".
    """

    def __init__(self, message: str | None, error: Any | None = None) -> None:
        super().__init__(message, "429", error)


class ChudGPTTimeoutException(RotorServiceException):
    """504 Timeout -- the provider did not answer inside the client timeout.

    Full code "003-504". Distinct from 503: the model was reachable, the wait
    was too short. Retrying the same call unchanged times out again.
    """

    def __init__(self, message: str | None, error: Any | None = None) -> None:
        super().__init__(message, "504", error)


#### ####

### AUDIO SERVICE EXCEPTION ###


class AudioServiceException(BaseException):
    """Base for every audio failure. service_code "004" (AUDIO_SERVICE)."""

    def __init__(
        self, message: str | None, error_code: str = "999", error: Any | None = None
    ) -> None:
        super().__init__(message, error_code, ServiceCode.AUDIO_SERVICE, error)


class ChudGPTAudioBackendMissingException(AudioServiceException, ImportError):
    """424 Failed Dependency -- the audio extra is not installed.

    Full code "004-424". Also an ``ImportError``, so ``except ImportError`` and
    ``pytest.importorskip`` keep working.
    """

    def __init__(self, message: str | None = None, error: Any | None = None) -> None:
        super().__init__(message, "424", error)


#### ####

__all__ = [
    "AudioServiceException",
    "BaseException",
    "ChudGPTAudioBackendMissingException",
    "ChudGPTBadDataException",
    "ChudGPTConflictException",
    "ChudGPTDBConfigException",
    "ChudGPTForbiddenException",
    "ChudGPTInternalServerException",
    "ChudGPTInvalidPathException",
    "ChudGPTNotFoundException",
    "ChudGPTRateLimitException",
    "ChudGPTServiceUnavailableException",
    "ChudGPTTimeoutException",
    "ChudGPTUnauthorizedException",
    "DBServiceException",
    "FileServiceException",
    "RotorServiceException",
    "ServiceCode",
]
