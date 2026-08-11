from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

_session_factory: sessionmaker | None = None


def configure_session_factory(engine: Engine) -> None:
    global _session_factory
    _session_factory = sessionmaker(engine)


def get_db() -> Session:
    """A ready-to-use session. Caller owns commit/rollback/close."""
    from chudgpt.exceptions import (
        ChudGPTInternalServerException,
        ServiceCode,
    )

    if _session_factory is None:
        raise ChudGPTInternalServerException(
            "db used before initialisation", ServiceCode.DB_SERVICE
        )
    return _session_factory()
