from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from chudgpt.db.db_initialiser import DBInitialiser
from chudgpt.db.exceptions import db_exception_handler
from chudgpt.db.models import Meta, ModelQuota, ModelUsage
from chudgpt.schemas.chat import Usage


class DBController:
    def __init__(self) -> None:
        db = DBInitialiser()
        self.__db = sessionmaker(db.engine)

    @db_exception_handler
    def create_usage_record(self, usage: Usage, model: str) -> None:
        with self.__db() as db:
            try:
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
                db.commit()
            except Exception:
                db.rollback()
                raise

    @db_exception_handler
    def get_meta(self, key: str) -> str | None:
        with self.__db() as db:
            row = db.get(Meta, key)
            return row.value if row else None

    @db_exception_handler
    def set_meta(self, key: str, value: str) -> None:
        with self.__db() as db:
            try:
                row = db.get(Meta, key)
                if row:
                    row.value = value
                else:
                    db.add(Meta(key=key, value=value))
                db.commit()
            except Exception:
                db.rollback()
                raise
