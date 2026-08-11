"""ChudGPT's public API.

    from chudgpt import ChudGPT
    from chudgpt.exceptions import ChudGPTRateLimitException

This module exposes the client and the types its methods take and return.
Every exception lives in ``chudgpt.exceptions``. Everything else -- the services
(rotor, db, files, scheduler), the ORM models, the key store -- is internal
and may change without notice.
"""

from .client import ChudGPT, ChudMessageBuilder
from .providers.gemini import GeminiModel
from .schemas.chat import ChudMessage, ChudMessageRole, ChudResponse, Usage
from .schemas.quota import (
    ChudProvider,
    ChudQuota,
    ChudUsageRecord,
    ChudUsageSummary,
)
from .schemas.structured import (
    ChudSchema,
    Classification,
    Confidence,
    GeneratedCode,
    Language,
    Sentiment,
    SentimentAnalysis,
)

__all__ = [
    "ChudGPT",
    "ChudMessage",
    "ChudMessageBuilder",
    "ChudMessageRole",
    "ChudProvider",
    "ChudQuota",
    "ChudResponse",
    "ChudSchema",
    "ChudUsageRecord",
    "ChudUsageSummary",
    "Classification",
    "Confidence",
    "GeminiModel",
    "GeneratedCode",
    "Language",
    "Sentiment",
    "SentimentAnalysis",
    "Usage",
]
