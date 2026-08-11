import pytest

from chudgpt import ChudGPT

APP_NAME = "chudgpt-pytests"


@pytest.fixture(scope="session")
def chud() -> ChudGPT:
    return ChudGPT().initialise(app_name=APP_NAME)
