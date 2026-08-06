import hashlib
import os
import sys
from pathlib import Path

from platformdirs import user_data_path

APP_NAME = "chudgpt"
HOME_ENV_VAR = "CHUDGPT_HOME"

_app_name: str | None = None


def set_app_name(name: str | None) -> None:
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


def data_dir() -> Path:
    override = os.environ.get(HOME_ENV_VAR)
    path = (
        Path(override)
        if override
        else user_data_path(APP_NAME, appauthor=False) / _namespace()
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    return data_dir() / "chudgpt.db"
