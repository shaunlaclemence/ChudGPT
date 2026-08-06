import pytest

from chudgpt.db.db_initialiser import DBInitialiser


@pytest.mark.skip()
def test_launch_db():
    helper = DBInitialiser()
    helper.launch_db_browser()


def test_read_json():
    helper = DBInitialiser()
    helper.init_providers("secrets.json")
    helper.init_quotas()
    helper.launch_db_browser()
