"""Per-key usage counters, cooldowns, and daily reset boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from .config import ProviderConfig

PACIFIC = ZoneInfo("America/Los_Angeles")


def day_key(cfg: ProviderConfig, now: datetime) -> str:
    """The calendar day a request counts against, in the provider's reset frame."""
    tz = PACIFIC if cfg.reset == "midnight_pt" else UTC
    return now.astimezone(tz).date().isoformat()


def next_reset(cfg: ProviderConfig, now: datetime) -> datetime:
    """When this provider's daily window next resets."""
    if cfg.reset == "midnight_pt":
        local = now.astimezone(PACIFIC)
        midnight = (local + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return midnight.astimezone(UTC)
    return now + timedelta(seconds=cfg.default_cooldown_s)


@dataclass
class KeyState:
    day: str = ""
    count: int = 0
    tokens: int = 0
    exhausted_until: float = 0.0  # unix timestamp; 0 = healthy


@dataclass
class QuotaTracker:
    keys: dict[str, KeyState] = field(default_factory=dict)

    def _state(self, kid: str, cfg: ProviderConfig, now: datetime) -> KeyState:
        st = self.keys.setdefault(kid, KeyState())
        today = day_key(cfg, now)
        if st.day != today:
            st.day = today
            st.count = 0
            st.tokens = 0
        return st

    def record(
        self, kid: str, cfg: ProviderConfig, now: datetime, tokens: int = 0
    ) -> None:
        st = self._state(kid, cfg, now)
        st.count += 1
        st.tokens += tokens

    def mark_exhausted(
        self,
        kid: str,
        cfg: ProviderConfig,
        now: datetime,
        until: datetime | None = None,
    ) -> None:
        st = self._state(kid, cfg, now)
        st.exhausted_until = (until or next_reset(cfg, now)).timestamp()

    def status(self, kid: str, cfg: ProviderConfig, now: datetime) -> str | None:
        """None if the key is usable now, else a human-readable reason."""
        st = self._state(kid, cfg, now)
        if st.exhausted_until > now.timestamp():
            until = datetime.fromtimestamp(st.exhausted_until, tz=UTC)
            return f"cooling down until {until.isoformat()}"
        if cfg.known_rpd is not None and st.count >= cfg.known_rpd:
            return f"hit known daily cap ({cfg.known_rpd} requests)"
        return None

    def available_at(self, kid: str, cfg: ProviderConfig, now: datetime) -> datetime:
        """Best estimate of when an unusable key becomes usable again."""
        st = self._state(kid, cfg, now)
        if st.exhausted_until > now.timestamp():
            return datetime.fromtimestamp(st.exhausted_until, tz=UTC)
        return next_reset(cfg, now)

    def to_dict(self) -> dict:
        return {
            kid: {
                "day": s.day,
                "count": s.count,
                "tokens": s.tokens,
                "exhausted_until": s.exhausted_until,
            }
            for kid, s in self.keys.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> QuotaTracker:
        tracker = cls()
        for kid, raw in data.items():
            tracker.keys[kid] = KeyState(
                day=raw.get("day", ""),
                count=int(raw.get("count", 0)),
                tokens=int(raw.get("tokens", 0)),
                exhausted_until=float(raw.get("exhausted_until", 0.0)),
            )
        return tracker
