from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from .chud_provider import ChudProvider
from .model_quota import ChudQuota


class UsagePeriod(Enum):
    ONE_DAY = timedelta(days=1)
    FIVE_HOUR = timedelta(hours=5)
    ONE_HOUR = timedelta(hours=1)
    FIVE_MIN = timedelta(minutes=5)
    ONE_MIN = timedelta(minutes=1)

    @staticmethod
    def values() -> list[UsagePeriod]:
        return [UsagePeriod.ONE_DAY, UsagePeriod.FIVE_HOUR, UsagePeriod.ONE_HOUR, UsagePeriod.FIVE_MIN, UsagePeriod.ONE_MIN]


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt: int = 0
    completion: int = 0
    total: int = 0
    requests: int = 1

    @property
    def reasoning(self) -> int:
        """Hidden thinking tokens: what total leaves over prompt + completion."""
        return max(0, self.total - self.prompt - self.completion)

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt=self.prompt + other.prompt,
            completion=self.completion + other.completion,
            total=self.total + other.total,
            requests=self.requests + other.requests,
        )

    @classmethod
    def from_completion(cls, result: Any) -> Usage:
        usage = getattr(result, "usage", None)
        if usage is None:
            return cls()
        return cls(
            prompt=usage.prompt_tokens or 0,
            completion=usage.completion_tokens or 0,
            total=usage.total_tokens or 0,
        )

    def __str__(self) -> str:
        return f"Usage(prompt={self.prompt}, completion={self.completion}, reasoning={self.reasoning}, total={self.total}, requests={self.requests})"


class ChudUsageRecord(BaseModel):
    model: str
    provider: ChudProvider
    created_at: datetime
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int


class ChudUsageSummary(BaseModel):
    usage: list[ChudUsageRecord]
    quotas: list[ChudQuota]
