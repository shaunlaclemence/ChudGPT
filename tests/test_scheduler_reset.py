"""The daily reset, exercised against a real database.

A day is simulated by backdating the ``last_reset`` marker rather than by
sleeping or by moving the clock: the scheduler decides whether it is due by
comparing that marker against the most recent midnight LA boundary, so a marker
older than that boundary is indistinguishable from the app having been shut
during the real fire time. That is the case worth testing, since the app is a
library and will usually be closed at midnight.
"""

import json
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from chudgpt.db.models import ModelQuota, ModelUsage
from chudgpt.scheduler import META_KEY, DailyScheduler, previous_fire_time
from chudgpt.schemas.chat import Usage
from chudgpt.services.db import DBService
from chudgpt.services.files import FilesService

SECRETS = {
    "gemini": [
        {
            "account": "test@example.com",
            "name": "test",
            "project_name": "proj",
            "project_number": 1,
            "api_key": "AIzaTESTKEYTESTKEY",
        }
    ]
}


class TempFiles(FilesService):
    """FilesService pinned to a tmp_path, so tests never touch the real data dir."""

    def __init__(self, root):
        self._root = root

    def data_dir(self):
        path = self._root / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def db_path(self):
        return self.data_dir() / "chudgpt.db"

    def secrets_path(self):
        return self._root / "secrets.json"


@pytest.fixture
def service(tmp_path):
    (tmp_path / "secrets.json").write_text(json.dumps(SECRETS))
    return DBService(TempFiles(tmp_path))


def rows(service, model):
    with Session(service.engine) as db:
        return list(db.scalars(select(model)))


def record_usage(service, count=3):
    model = rows(service, ModelQuota)[0].model
    for _ in range(count):
        service.create_usage_record(Usage(prompt=1, completion=2, total=3), model)
    return model


class Run(NamedTuple):
    was_due: bool
    fired: bool
    next_run: datetime | None
    last_run: datetime | None


def wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(interval)
    return predicate()


def run_scheduler(service, timeout=5.0):
    """Start the scheduler on flush_usage and wait for the job to finish.

    The scheduler stamps ``last_reset`` in a finally *after* the job body
    returns, so waiting on the job alone races the marker write -- wait for the
    marker to change too. State is read before shutdown, since the job store is
    torn down with it.
    """
    done = threading.Event()
    before = service.get_meta(META_KEY)

    def job():
        try:
            service.flush_usage()
        finally:
            done.set()

    scheduler = DailyScheduler(func=job, controller=service)
    try:
        was_due = scheduler.is_due
        scheduler.start()
        fired = done.wait(timeout=timeout) if was_due else done.wait(timeout=0.75)
        if fired:
            wait_until(lambda: service.get_meta(META_KEY) != before, timeout=timeout)
        return Run(was_due, fired, scheduler.next_run, scheduler.last_run)
    finally:
        scheduler.shutdown()


def test_reset_flushes_usage_after_a_day_has_passed(service):
    record_usage(service, count=3)
    assert len(rows(service, ModelUsage)) == 3

    # the app was closed when midnight LA went by
    stale = previous_fire_time() - timedelta(hours=1)
    service.set_meta(META_KEY, stale.isoformat())

    run = run_scheduler(service)
    assert run.was_due, "a day elapsed since the last reset, so it should be due"
    assert run.fired, "catch-up job did not run"

    assert rows(service, ModelUsage) == [], "usage should be cleared by the reset"
    assert rows(service, ModelQuota), "quota limits must survive the reset"

    assert run.last_run >= previous_fire_time()


def test_reset_does_not_run_twice_in_the_same_day(service):
    record_usage(service, count=2)
    service.set_meta(META_KEY, datetime.now(UTC).isoformat())

    run = run_scheduler(service)
    assert not run.was_due, "already reset since the last midnight"
    assert not run.fired, "reset must not run again the same day"
    assert len(rows(service, ModelUsage)) == 2


def test_first_ever_launch_is_due(service):
    assert service.get_meta(META_KEY) is None
    run = run_scheduler(service)
    assert run.was_due and run.fired


def test_usage_can_be_recorded_again_after_a_reset(service):
    model = record_usage(service, count=1)
    service.set_meta(META_KEY, (previous_fire_time() - timedelta(hours=1)).isoformat())
    run_scheduler(service)

    service.create_usage_record(Usage(prompt=1, completion=1, total=2), model)
    assert len(rows(service, ModelUsage)) == 1


def test_next_run_is_midnight_los_angeles(service):
    next_run = run_scheduler(service).next_run
    assert (next_run.hour, next_run.minute) == (0, 0)
    assert next_run > datetime.now(UTC)


def test_previous_fire_time_is_the_most_recent_midnight():
    now = datetime(2026, 8, 7, 9, 30, tzinfo=UTC)  # 02:30 LA on the 7th
    assert previous_fire_time(now).isoformat() == "2026-08-07T00:00:00-07:00"

    just_before = datetime(2026, 8, 7, 6, 59, tzinfo=UTC)  # 23:59 LA on the 6th
    assert previous_fire_time(just_before).isoformat() == "2026-08-06T00:00:00-07:00"
