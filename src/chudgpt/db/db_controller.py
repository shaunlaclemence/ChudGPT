from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from chudgpt.db.db_initialiser import DBInitialiser
from chudgpt.db.exceptions import db_exception_handler
from chudgpt.db.models import Meta, ModelQuota, ModelUsage
from chudgpt.schemas.chat import Usage


class DBController:
    def __init__(self) -> None:
        DBInitialiser()

    @db_exception_handler
    def create_usage_record(self, db: Session, usage: Usage, model: str) -> None:
        quota = db.execute(
            select(ModelQuota).where(ModelQuota.model == model)
        ).scalar_one()
        db.add(
            ModelUsage(
                quota_id=quota.id,
                created_at=datetime.now(UTC),
                prompt_tokens=usage.prompt,
                completion_tokens=usage.completion,
                total_tokens=usage.total,
            )
        )

    @db_exception_handler
    def flush_usage(self, db: Session):
        db.execute(delete(ModelQuota))

    @db_exception_handler
    def get_meta(self, db: Session, key: str) -> str | None:
        row = db.get(Meta, key)
        return row.value if row else None

    @db_exception_handler
    def set_meta(self, db: Session, key: str, value: str) -> None:
        row = db.get(Meta, key)
        if row:
            row.value = value
        else:
            db.add(Meta(key=key, value=value))
