from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from chudgpt._schemas import ChudProvider, ChudUsageRecord


class UsageRules:
    ChudUsageResponse = dict[ChudProvider, dict[str, int]]

    @classmethod
    def as_utc(cls, moment: datetime) -> datetime:
        # sqlite drops the offset, so a naive value is the UTC it was written as
        return moment if moment.tzinfo else moment.replace(tzinfo=UTC)

    @classmethod
    def is_recent(
        cls, moment: datetime, window: timedelta, now: datetime | None = None
    ) -> bool:
        return cls.as_utc(moment) > (now or datetime.now(UTC)) - window

    @classmethod
    def collate_usage(
        cls,
        usages: list[ChudUsageRecord],
        filter: Callable[[ChudUsageRecord], bool],
        metric: str = "",
    ) -> ChudUsageResponse:
        providers = {u.provider for u in usages if filter(u)}
        res = {}
        for p in providers:
            usages_for_p = [u for u in usages if u.provider.id == p.id and filter(u)]
            model_usage = defaultdict(int)
            for u in usages_for_p:
                try:
                    model_usage[u.model] += u.model_dump()[metric]
                except KeyError:
                    model_usage[u.model] += 1
            res[p] = dict(model_usage)
        return res
