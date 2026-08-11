import threading

import pytest
from conftest import APP_NAME

from chudgpt.db.db_initialiser import DBInitialiser
from chudgpt.services.scheduler import META_KEY, SchedulerService


class FakeController:
    def __init__(self):
        self._meta: dict[str, str] = {}

    def get_meta(self, key: str) -> str | None:
        return self._meta.get(key)

    def set_meta(self, key: str, value: str) -> None:
        self._meta[key] = value


# @pytest.mark.skip()
def test_launch_db():
    helper = DBInitialiser(APP_NAME)
    helper.launch_db_browser()


@pytest.mark.skip()
def test_read_json():
    helper = DBInitialiser(APP_NAME)
    helper.init_providers("secrets.json")
    helper.init_quotas()
    helper.launch_db_browser()


def test_scheduler_runs_catchup_job_and_records_last_run():
    controller = FakeController()
    ran = threading.Event()
    sched = SchedulerService(ran.set, controller)

    try:
        sched.start()
        assert sched.running

        assert ran.wait(timeout=2), "catchup job did not run"
        assert controller.get_meta(META_KEY) is not None
        assert sched.last_run is not None
        assert sched.next_run is not None
    finally:
        sched.shutdown()
