from __future__ import annotations

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ProviderRow(Base):
    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    project_name: Mapped[str] = mapped_column(String(255))
    project_number: Mapped[str] = mapped_column(String(64), unique=True)

    quotas: Mapped[list[ModelQuota]] = relationship(
        back_populates="provider", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"ProviderRow(id={self.id}, email={self.email!r}, name={self.name!r}, "
            f"project_number={self.project_number!r})"
        )


class ModelQuota(Base):
    __tablename__ = "model_quota"
    __table_args__ = (UniqueConstraint("provider_id", "model"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    provider_id: Mapped[int] = mapped_column(ForeignKey("providers.id"))
    model: Mapped[str] = mapped_column(String(128))
    requests: Mapped[int] = mapped_column(default=0)
    prompt_tokens: Mapped[int] = mapped_column(default=0)
    completion_tokens: Mapped[int] = mapped_column(default=0)
    total_tokens: Mapped[int] = mapped_column(default=0)

    provider: Mapped[ProviderRow] = relationship(back_populates="quotas")

    def __repr__(self) -> str:
        return (
            f"ModelQuota(provider_id={self.provider_id}, model={self.model!r}, "
            f"requests={self.requests}, total_tokens={self.total_tokens})"
        )


class Meta(Base):
    """Single-row-per-key bookkeeping, currently just the last quota reset day."""

    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(64))
