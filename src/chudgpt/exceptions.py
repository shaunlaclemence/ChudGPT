from typing import Any


class BaseException(Exception):
    def __init__(
        self,
        message: str | None,
        error_code: str = "999",
        service_code: str = "999",
        error: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.service_code = service_code
        self.error = error


class FileServiceException(BaseException):
    def __init__(
        self, message: str | None, error_code: str = "999", error: Any | None = None
    ) -> None:
        super().__init__(message, error_code, "001", error)


class ChudGPTServiceUnavailableException(FileServiceException):
    def __init__(self, message: str | None = None, error: Any | None = None) -> None:
        super().__init__(message, "503", error)


class ChudGPTInternalServerException(FileServiceException):
    def __init__(self, message: str | None = None, error: Any | None = None) -> None:
        super().__init__(message, "500", error)


class ChudGPTInvalidPathException(FileServiceException):
    def __init__(self, message: str | None = None, error: Any | None = None) -> None:
        super().__init__(message, "422", error)


class ChudGPTConflictException(FileServiceException):
    def __init__(self, message: str | None = None, error: Any | None = None) -> None:
        super().__init__(message, "409", error)


class ChudGPTFileNotFoundException(FileServiceException):
    def __init__(self, message: str | None = None, error: Any | None = None) -> None:
        super().__init__(message, "404", error)


class ChudGPTForbiddenException(FileServiceException):
    def __init__(self, message: str | None = None, error: Any | None = None) -> None:
        super().__init__(message, "403", error)


class ChudGPTUnauthorizedException(FileServiceException):
    def __init__(self, message: str | None = None, error: Any | None = None) -> None:
        super().__init__(message, "401", error)


class ChudGPTBadDataException(FileServiceException):
    def __init__(self, message: str | None = None, error: Any | None = None) -> None:
        super().__init__(message, "400", error)
