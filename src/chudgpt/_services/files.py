import json
import os
import sqlite3
import sys
from pathlib import Path

from platformdirs import user_data_path

from chudgpt._services.exceptions.files import file_exception_handler
from chudgpt._utils.naming import AppNameRules

APP_NAME = "chudgpt"
HOME_ENV_VAR = "CHUDGPT_HOME"
SECRETS_FILE = "secrets.json"


def _process_identity() -> Path:
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if main_file:
        return Path(main_file).resolve()
    if sys.argv and sys.argv[0]:
        return Path(sys.argv[0]).resolve()
    return Path.cwd()


class FilesService:
    def __init__(self) -> None:
        pass

    @file_exception_handler
    def json_to_dict(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            d = json.load(file)
        return d

    @file_exception_handler
    def init_store(self, app_name: str):
        connection = sqlite3.connect(self.db_path(app_name))
        connection.close()

    @file_exception_handler
    @file_exception_handler
    def data_dir(self) -> Path:
        override = os.environ.get(HOME_ENV_VAR)
        path = Path(override) if override else user_data_path(APP_NAME, appauthor=False)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @file_exception_handler
    def db_path(self, app_name: str) -> Path:
        return self.data_dir() / AppNameRules.db_filename(app_name)

    @file_exception_handler
    def db_exists(self, app_name: str) -> bool:
        return self.db_path(app_name).is_file()

    @file_exception_handler
    def root_dir(self) -> Path:
        """The directory holding the app's secrets.json.

        An existing secrets.json is the strongest signal, so it wins: a marker-only
        search stops at the first pyproject.toml, which in a monorepo is the
        sub-package (services/core) rather than the repo root where the file lives.
        Falls back to the nearest pyproject.toml/.git so a missing-secrets error
        still names a sensible directory.
        """
        start = _process_identity()
        current = start if start.is_dir() else start.parent
        candidates = (current, *current.parents)
        for directory in candidates:
            if (directory / SECRETS_FILE).exists():
                return directory
        for directory in candidates:
            if (directory / "pyproject.toml").exists() or (directory / ".git").exists():
                return directory
        return current

    @file_exception_handler
    def secrets_path(self) -> Path:
        return self.root_dir() / SECRETS_FILE
