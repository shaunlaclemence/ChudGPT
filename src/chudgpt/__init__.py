"""ChudGPT's public API.

    from chudgpt import ChudGPT
    from chudgpt.exceptions import ChudGPTRateLimitException

This module exposes the client and the types its methods take and return.
Every exception lives in ``chudgpt.exceptions``. Everything else -- the services
(rotor, db, files, scheduler), the ORM models, the key store -- is internal
and may change without notice.
"""

from .client import ChudGPT, MessageBuilder
from .providers.gemini import GeminiModel
from .schemas.chat import ChudResponse, Message, MessageRole, Usage

__all__ = [
    "ChudGPT",
    "ChudResponse",
    "GeminiModel",
    "Message",
    "MessageBuilder",
    "MessageRole",
    "Usage",
]
