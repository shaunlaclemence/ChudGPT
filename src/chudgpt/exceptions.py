from enum import Enum
from typing import Any


class ServiceCode(str, Enum):
    FILE_SERVICE = "001"
    DB_SERVICE = "002"
    UNKOWN_SERVICE = "999"


class BaseException(Exception):
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
    def __init__(
        self, message: str, service_code: ServiceCode, error: Any | None = None
    ) -> None:
        super().__init__(message, "500", service_code, error)


class ChudGPTServiceUnavailableException(BaseException):
    def __init__(
        self, message: str, service_code: ServiceCode, error: Any | None = None
    ) -> None:
        super().__init__(message, "503", service_code, error)


class ChudGPTConflictException(BaseException):
    def __init__(
        self, message: str, service_code: ServiceCode, error: Any | None = None
    ) -> None:
        super().__init__(message, "409", service_code, error)


class ChudGPTNotFoundException(BaseException):
    def __init__(
        self, message: str, service_code: ServiceCode, error: Any | None = None
    ) -> None:
        super().__init__(message, "404", service_code, error)


class ChudGPTForbiddenException(BaseException):
    def __init__(
        self, message: str, service_code: ServiceCode, error: Any | None = None
    ) -> None:
        super().__init__(message, "403", service_code, error)


class ChudGPTUnauthorizedException(BaseException):
    def __init__(
        self, message: str, service_code: ServiceCode, error: Any | None = None
    ) -> None:
        super().__init__(message, "401", service_code, error)


class ChudGPTBadDataException(BaseException):
    def __init__(
        self, message: str, service_code: ServiceCode, error: Any | None = None
    ) -> None:
        super().__init__(message, "400", service_code, error)


#### ####

### FILE SERVICE EXCEPTIONS ####


class FileServiceException(BaseException):
    def __init__(
        self, message: str | None, error_code: str = "999", error: Any | None = None
    ) -> None:
        super().__init__(message, error_code, ServiceCode.FILE_SERVICE, error)


class ChudGPTInvalidPathException(FileServiceException):
    def __init__(self, message: str | None = None, error: Any | None = None) -> None:
        super().__init__(message, "422", error)


#### ####

### DB SERVICE EXCEPTION ###


class DBServiceException(BaseException):
    def __init__(
        self, message: str | None, error_code: str = "999", error: Any | None = None
    ) -> None:
        super().__init__(message, error_code, ServiceCode.DB_SERVICE, error)


class ChudGPTDBConfigException(DBServiceException):
    def __init__(self, message: str | None = None, error: Any | None = None) -> None:
        super().__init__(message, "422", error)


#### ####
