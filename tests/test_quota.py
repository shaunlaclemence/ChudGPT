from datetime import UTC, datetime, timedelta

from chudgpt.quota import PACIFIC, QuotaTracker, day_key, next_reset


def test_record_increments_and_day_rollover(providers, now):
    alpha = providers[0]
    t = QuotaTracker()
    t.record("alpha:x", alpha, now, tokens=10)
    t.record("alpha:x", alpha, now, tokens=5)
    assert t.keys["alpha:x"].count == 2
    assert t.keys["alpha:x"].tokens == 15

    tomorrow = now + timedelta(days=1)
    t.record("alpha:x", alpha, tomorrow)
    assert t.keys["alpha:x"].count == 1  # counters reset on the new day


def test_status_flags_known_daily_cap(providers, now):
    alpha = providers[0]  # known_rpd=100
    t = QuotaTracker()
    for _ in range(100):
        t.record("alpha:x", alpha, now)
    assert "daily cap" in t.status("alpha:x", alpha, now)
    # next day it clears
    assert t.status("alpha:x", alpha, now + timedelta(days=1)) is None


def test_mark_exhausted_and_recovery(providers, now):
    alpha = providers[0]
    t = QuotaTracker()
    t.mark_exhausted("alpha:x", alpha, now, until=now + timedelta(minutes=10))
    assert "cooling down" in t.status("alpha:x", alpha, now)
    assert t.status("alpha:x", alpha, now + timedelta(minutes=11)) is None


def test_next_reset_midnight_pacific(providers, now):
    beta = providers[1]  # reset="midnight_pt"
    reset = next_reset(beta, now)
    local = reset.astimezone(PACIFIC)
    assert (local.hour, local.minute, local.second) == (0, 0, 0)
    assert reset > now
    assert reset - now <= timedelta(hours=24)


def test_day_key_uses_pacific_for_midnight_pt_providers(providers):
    beta = providers[1]
    # 06:00 UTC on July 28 is still July 27 in Los Angeles (UTC-7 in summer)
    edge = datetime(2026, 7, 28, 6, 0, tzinfo=UTC)
    assert day_key(beta, edge) == "2026-07-27"


def test_round_trip_serialization(providers, now):
    alpha = providers[0]
    t = QuotaTracker()
    t.record("alpha:x", alpha, now, tokens=42)
    t.mark_exhausted("beta:y", providers[1], now)
    restored = QuotaTracker.from_dict(t.to_dict())
    assert restored.keys["alpha:x"].tokens == 42
    assert restored.keys["beta:y"].exhausted_until > 0
