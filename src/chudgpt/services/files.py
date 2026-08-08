import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

from platformdirs import user_data_path

from chudgpt.services.exceptions.files import file_exception_handler

APP_NAME = "chudgpt"
HOME_ENV_VAR = "CHUDGPT_HOME"
SECRETS_FILE = "secrets.json"

_app_name: str | None = None


def _process_identity() -> Path:
    """Best-effort path identifying the running app, so two unrelated apps
    that both depend on chudgpt don't silently share one database."""
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if main_file:
        return Path(main_file).resolve()
    if sys.argv and sys.argv[0]:
        return Path(sys.argv[0]).resolve()
    return Path.cwd()


def _slugify(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in label)


def _namespace() -> str:
    if _app_name:
        return _slugify(_app_name)
    identity = _process_identity()
    label = identity.parent.name or identity.name or "app"
    digest = hashlib.sha256(str(identity).encode()).hexdigest()[:8]
    return f"{_slugify(label)}-{digest}"


class FilesService:
    def __init__(self) -> None:
        pass

    @file_exception_handler
    def json_to_dict(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            d = json.load(file)
        return d

    @file_exception_handler
    def init_store(self):
        """Initalise the user data dir and chudgpt.db, Idempotent"""
        connection = sqlite3.connect(self.db_path())
        connection.close()

    @file_exception_handler
    def set_app_name(self, name: str | None) -> None:
        """Declare which app is using chudgpt, so its data directory doesn't
        collide with any other app's. Call this before touching the db."""
        global _app_name
        if name is not None and _app_name is not None and _app_name != name:
            raise ValueError(
                f"app_name already set to {_app_name!r}, cannot change to {name!r} "
                "within the same process"
            )
        if name is not None:
            _app_name = name

    @file_exception_handler
    def data_dir(self) -> Path:
        override = os.environ.get(HOME_ENV_VAR)
        path = (
            Path(override)
            if override
            else user_data_path(APP_NAME, appauthor=False) / _namespace()
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @file_exception_handler
    def db_path(self) -> Path:
        return self.data_dir() / "chudgpt.db"

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
