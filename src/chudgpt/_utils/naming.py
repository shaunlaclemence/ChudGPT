import re

from chudgpt.exceptions import ChudGPTBadDataException, ServiceCode


class AppNameRules:
    _SPLIT = re.compile(r"[^0-9a-zA-Z]+|(?<=[a-z0-9])(?=[A-Z])")

    @classmethod
    def kebab(cls, app_name: str) -> str:
        if not isinstance(app_name, str) or not app_name.strip():
            raise ChudGPTBadDataException(
                "app_name must be a non-empty string",
                ServiceCode.FILE_SERVICE,
            )
        parts = [p for p in cls._SPLIT.split(app_name) if p]
        if not parts:
            raise ChudGPTBadDataException(
                f"app_name {app_name!r} has no alphanumeric characters",
                ServiceCode.FILE_SERVICE,
            )
        return "-".join(p.lower() for p in parts)

    @classmethod
    def db_filename(cls, app_name: str) -> str:
        return f"{cls.kebab(app_name)}.db"
