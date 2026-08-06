from datetime import UTC, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from chudgpt.db.db_controller import DBController

RESET_TZ = ZoneInfo("America/Los_Angeles")
JOB_ID = "daily_reset"
CATCHUP_JOB_ID = "daily_reset_catchup"
META_KEY = "last_reset"


def previous_fire_time(now=None):
    local = (now or datetime.now(UTC)).astimezone(RESET_TZ)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if midnight > local:
        midnight -= timedelta(days=1)
    return midnight


class Controller(Protocol):
    def get_meta(self, key: str) -> str | None:
        pass

    def set_meta(self, key: str, value: str) -> None:
        pass


class DailyScheduler:
    def __init__(self, controller: Controller | None = None):
        self._controller = controller or DBController()
        self._scheduler = BackgroundScheduler(timezone=RESET_TZ)

    @property
    def running(self):
        return self._scheduler.running

    @property
    def last_run(self):
        value = self._controller.get_meta(META_KEY)
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @property
    def next_run(self):
        job = self._scheduler.get_job(JOB_ID)
        return job.next_run_time if job else None

    @property
    def is_due(self):
        last = self.last_run
        return last is None or last < previous_fire_time()

    def start(self, func):
        self._scheduler.add_job(
            self._run,
            CronTrigger(hour=0, minute=0, timezone=RESET_TZ),
            args=[func],
            id=JOB_ID,
            coalesce=True,
            misfire_grace_time=None,
            replace_existing=True,
        )
        if not self._scheduler.running:
            self._scheduler.start()
        if self.is_due:
            self._scheduler.add_job(
                self._run,
                args=[func],
                id=CATCHUP_JOB_ID,
                coalesce=True,
                misfire_grace_time=None,
                replace_existing=True,
            )
        return self

    def shutdown(self, wait=True):
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)

    def _run(self, func):
        try:
            func()
        finally:
            self._controller.set_meta(META_KEY, datetime.now(UTC).isoformat())
