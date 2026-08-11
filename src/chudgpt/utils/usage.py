from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from chudgpt.schemas.quota import ChudUsageRecord


class UsageRules:
    RECENT_WINDOW = timedelta(minutes=1)

    @classmethod
    def as_utc(cls, moment: datetime) -> datetime:
        # sqlite drops the offset, so a naive value is the UTC it was written as
        return moment if moment.tzinfo else moment.replace(tzinfo=UTC)

    @classmethod
    def is_recent(cls, moment: datetime, now: datetime | None = None) -> bool:
        return cls.as_utc(moment) > (now or datetime.now(UTC)) - cls.RECENT_WINDOW

    @classmethod
    def collate_usage(
        cls, usages: list[ChudUsageRecord], filter: Callable[[ChudUsageRecord], bool]
    ):
        providers = {u.provider for u in usages if filter(u)}
        res = {}
        for p in providers:
            usages_for_p = [u for u in usages if u.provider.id == p.id and filter(p)]
            model_rpd = defaultdict(int)
            for u in usages_for_p:
                model_rpd[u.model] += 1
            res[p] = model_rpd
        return res
