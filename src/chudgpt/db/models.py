from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Provider(Base):
    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    project_name: Mapped[str] = mapped_column(String(255))
    project_number: Mapped[str] = mapped_column(String(64), unique=True)
    api_key: Mapped[str] = mapped_column(String(16))

    def __repr__(self) -> str:
        return (
            f"ProviderRow(id={self.id}, email={self.email!r}, name={self.name!r}, "
            f"project_number={self.project_number!r}), api_key={self.api_key!r}"
        )


class ModelQuota(Base):
    """What this provider is *allowed* to spend on a model, mirrored from
    config.json. Holds limits only -- consumption lives in ModelUsage."""

    __tablename__ = "model_quota"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model: Mapped[str] = mapped_column(String(128))
    rpd: Mapped[int]
    rpm: Mapped[int]
    tpm: Mapped[int]
    inputs: Mapped[str] = mapped_column(String(128))

    usage: Mapped[list[ModelUsage]] = relationship(
        back_populates="quota", cascade="all, delete-orphan"
    )

    @property
    def allowed_inputs(self) -> list[str]:
        """``"text,video,audio"`` split back into a list."""
        return [i for i in self.inputs.split(",") if i]

    def __repr__(self) -> str:
        return (
            f"ModelQuota(model={self.model!r}, "
            f"rpd={self.rpd}, rpm={self.rpm}, tpm={self.tpm}, "
            f"inputs={self.inputs!r})"
        )


class ModelUsage(Base):
    """One row per completed request. Append-only: rows are never updated, only
    inserted, and the whole table is flushed on init and once per day."""

    __tablename__ = "model_usage"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    quota_id: Mapped[int] = mapped_column(ForeignKey("model_quota.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    prompt_tokens: Mapped[int]
    completion_tokens: Mapped[int]
    total_tokens: Mapped[int]

    quota: Mapped[ModelQuota] = relationship(back_populates="usage")

    @property
    def reasoning_tokens(self) -> int:
        return self.total_tokens - self.prompt_tokens - self.completion_tokens

    def __repr__(self) -> str:
        return (
            f"ModelUsage(quota_id={self.quota_id}, at={self.created_at!s}, "
            f"prompt={self.prompt_tokens}, completion={self.completion_tokens}, "
            f"reasoning={self.reasoning_tokens}, total={self.total_tokens})"
        )


class Meta(Base):
    """Single-row-per-key bookkeeping, currently just the last quota reset day."""

    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(64))
