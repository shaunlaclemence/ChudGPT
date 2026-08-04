from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..providers.gemini import CATALOG
from ..schemas.chat import Provider, Usage
from .models import Base, Meta, ModelQuota, ProviderRow

PACIFIC = ZoneInfo("America/Los_Angeles")
LAST_RESET = "last_reset"


def db_path() -> Path:
    """Project-local by default; never inside the installed package, which is
    wiped on upgrade and read-only in many deployments."""
    env = os.environ.get("CHUDGPT_DB")
    return Path(env) if env else Path.cwd() / ".chudgpt" / "quota.db"


def day_key(now: datetime) -> str:
    """The day quotas belong to. Gemini's RPD rolls over at midnight Pacific."""
    return now.astimezone(PACIFIC).date().isoformat()


class QuotaDB:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(f"sqlite+pysqlite:///{self.path}")
        self.session = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    def ensure_ready(
        self, providers: list[Provider], now: datetime | None = None
    ) -> None:
        """Resync providers from secrets.json, then reset counters once per day.

        Called before the first request rather than from a post-install hook,
        because wheels have no reliable post-install step. The resync runs on
        every call, so init and the daily reset both see current secrets.
        """
        now = now or datetime.now(UTC)
        with self.session.begin() as session:
            self._sync_providers(session, providers)
            self._reset_if_new_day(session, now)

    def _sync_providers(self, session: Session, providers: list[Provider]) -> None:
        """Mirror secrets.json into the providers table, api_key excluded.

        Rows are matched on project_number: new ones are inserted, existing ones
        have their metadata refreshed, and any row no longer in secrets.json is
        deleted along with its quota rows.
        """
        seen: set[str] = set()
        for provider in providers:
            seen.add(provider.project_number)
            row = session.scalar(
                select(ProviderRow).where(
                    ProviderRow.project_number == provider.project_number
                )
            )
            if row is None:
                row = ProviderRow(
                    email=provider.account,
                    name=provider.name,
                    project_name=provider.project_name,
                    project_number=provider.project_number,
                )
                session.add(row)
                session.flush()
            else:
                row.email = provider.account
                row.name = provider.name
                row.project_name = provider.project_name
            self._sync_models(session, row)

        for row in session.scalars(select(ProviderRow)):
            if row.project_number not in seen:
                session.delete(row)

    def _sync_models(self, session: Session, row: ProviderRow) -> None:
        existing = {
            q.model
            for q in session.scalars(
                select(ModelQuota).where(ModelQuota.provider_id == row.id)
            )
        }
        for entry in CATALOG["gemini"].values():
            if entry["slug"] not in existing:
                session.add(ModelQuota(provider_id=row.id, model=entry["slug"]))

    def _reset_if_new_day(self, session: Session, now: datetime) -> None:
        today = day_key(now)
        marker = session.get(Meta, LAST_RESET)
        if marker is None:
            session.add(Meta(key=LAST_RESET, value=today))
            return
        if marker.value == today:
            return
        session.execute(
            update(ModelQuota).values(
                requests=0, prompt_tokens=0, completion_tokens=0, total_tokens=0
            )
        )
        marker.value = today

    def last_reset(self) -> str | None:
        with self.session() as session:
            marker = session.get(Meta, LAST_RESET)
            return marker.value if marker else None

    def quotas(self, project_number: str) -> list[ModelQuota]:
        with self.session() as session:
            row = session.scalar(
                select(ProviderRow).where(ProviderRow.project_number == project_number)
            )
            if row is None:
                return []
            return list(
                session.scalars(
                    select(ModelQuota).where(ModelQuota.provider_id == row.id)
                )
            )

    def record(self, project_number: str, model: str, usage: Usage) -> None:
        """Add one request's spend onto the provider/model row.

        Done as a single UPDATE so two processes sharing the file can't lose an
        increment to a read-modify-write race.
        """
        with self.session.begin() as session:
            row = session.scalar(
                select(ProviderRow).where(ProviderRow.project_number == project_number)
            )
            if row is None:
                return
            result = session.execute(
                update(ModelQuota)
                .where(
                    ModelQuota.provider_id == row.id,
                    ModelQuota.model == model,
                )
                .values(
                    requests=ModelQuota.requests + usage.requests,
                    prompt_tokens=ModelQuota.prompt_tokens + usage.prompt,
                    completion_tokens=ModelQuota.completion_tokens + usage.completion,
                    total_tokens=ModelQuota.total_tokens + usage.total,
                )
            )
            if result.rowcount == 0:
                session.add(
                    ModelQuota(
                        provider_id=row.id,
                        model=model,
                        requests=usage.requests,
                        prompt_tokens=usage.prompt,
                        completion_tokens=usage.completion,
                        total_tokens=usage.total,
                    )
                )

    def quota(self, project_number: str, model: str) -> ModelQuota | None:
        with self.session() as session:
            row = session.scalar(
                select(ProviderRow).where(ProviderRow.project_number == project_number)
            )
            if row is None:
                return None
            return session.scalar(
                select(ModelQuota).where(
                    ModelQuota.provider_id == row.id, ModelQuota.model == model
                )
            )

    def providers(self) -> list[ProviderRow]:
        with self.session() as session:
            return list(session.scalars(select(ProviderRow)))
