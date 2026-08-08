"""ChudGPT's public API: the client, the types its methods take and return, and
every exception it can raise. Everything else (Rotor, DBService, FilesService,
DailyScheduler, ORM models) is internal and may change without notice.
"""

from .client import ChudGPT, MessageBuilder
from .exceptions import (
    BaseException,
    ChudGPTBadDataException,
    ChudGPTConflictException,
    ChudGPTDBConfigException,
    ChudGPTForbiddenException,
    ChudGPTInternalServerException,
    ChudGPTInvalidPathException,
    ChudGPTNotFoundException,
    ChudGPTServiceUnavailableException,
    ChudGPTUnauthorizedException,
    DBServiceException,
    FileServiceException,
    ServiceCode,
)
from .providers.gemini import GeminiModel
from .schemas.chat import Message, MessageRole, Response, Usage

__all__ = [
    "BaseException",
    "ChudGPT",
    "ChudGPTBadDataException",
    "ChudGPTConflictException",
    "ChudGPTDBConfigException",
    "ChudGPTForbiddenException",
    "ChudGPTInternalServerException",
    "ChudGPTInvalidPathException",
    "ChudGPTNotFoundException",
    "ChudGPTServiceUnavailableException",
    "ChudGPTUnauthorizedException",
    "DBServiceException",
    "FileServiceException",
    "GeminiModel",
    "Message",
    "MessageBuilder",
    "MessageRole",
    "Response",
    "ServiceCode",
    "Usage",
]
