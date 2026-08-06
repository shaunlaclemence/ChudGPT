import os
from pathlib import Path

from platformdirs import user_data_path

APP_NAME = "chudgpt"
HOME_ENV_VAR = "CHUDGPT_HOME"


def data_dir() -> Path:
    override = os.environ.get(HOME_ENV_VAR)
    path = Path(override) if override else user_data_path(APP_NAME, appauthor=False)
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    return data_dir() / "chudgpt.db"
