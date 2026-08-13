"""ChudGPT's public API.

    from chudgpt import ChudGPT
    from chudgpt.exceptions import ChudGPTRateLimitException
    from chudgpt.messages import ChudMessageBuilder

This module exposes ``ChudGPT`` and the types its methods take and return.
Every exception lives in ``chudgpt.exceptions``, every message builder in
``chudgpt.messages``. Everything else -- the services (rotor, db, files,
scheduler), the providers, the ORM models, the key store -- is internal
and may change without notice.
"""

from chudgpt._schemas import (
    ChudMessage,
    ChudProvider,
    ChudQuota,
    ChudResponse,
    ChudSchema,
    ChudUsageRecord,
    ChudUsageSummary,
    Classification,
    Confidence,
    GeneratedCode,
    Language,
    Sentiment,
    SentimentAnalysis,
    Usage,
    UsagePeriod,
)

from ._main import ChudGPT
from ._providers.gemini import GeminiModel
from ._utils.version import VersionRules

__version__ = VersionRules.installed()

__all__ = [
    "ChudGPT",
    "ChudMessage",
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
    "UsagePeriod",
    "__version__",
]
