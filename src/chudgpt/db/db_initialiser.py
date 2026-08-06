import json
import os
import shutil
import sqlite3
import subprocess
import sys
from importlib import resources
from pathlib import Path

from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker

from chudgpt.db.exceptions import db_exception_handler
from chudgpt.db.models import Base, ModelQuota, Provider
from chudgpt.paths import db_path
from chudgpt.schemas.chat import Provider as ProviderDTO
from chudgpt.schemas.quota import Quota

APP_NAME = "DB Browser for SQLite"

INSTALL_HINTS = {
    "win32": f"winget install DBBrowserForSQLite.{APP_NAME.replace(' ', '')}",
    "darwin": "brew install --cask db-browser-for-sqlite",
}
LINUX_HINT = "sudo apt install sqlitebrowser"


class DBInitialiser:
    def __init__(self) -> None:
        self.root_path = None
        self.db_path = None
        self.engine = None
        self.__init_store()
        self.__init_tables()

    def __json_to_dict(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            d = json.load(file)
        return d

    def __init_store(self):
        """Initalise the user data dir and chudgpt.db, Idempotent"""
        self.db_path = db_path()
        connection = sqlite3.connect(self.db_path)
        connection.close()

    def __init_tables(self):
        if not self.db_path:
            raise ValueError("db not initialised")
        self.engine = create_engine(f"sqlite:///{self.db_path.resolve()}")
        Base.metadata.create_all(self.engine)

    @db_exception_handler
    def init_providers(self, secrets_path: Path | str):
        providers = self.__json_to_dict(Path(secrets_path))
        gemini_providers = [
            ProviderDTO(
                account=p["account"],
                name=p["name"],
                project_name=p["project_name"],
                project_number=str(p["project_number"]),
                api_key=p["api_key"],
            )
            for p in providers["gemini"]
        ]

        session = sessionmaker(self.engine)
        with session() as db:
            # Truncate providers
            db.execute(delete(Provider))
            db.commit()

            db.add_all(
                [
                    Provider(
                        email=p.account,
                        name=p.name,
                        project_name=p.project_name,
                        project_number=p.project_number,
                        api_key=p.masked_key,
                    )
                    for p in gemini_providers
                ]
            )
            db.commit()

    @db_exception_handler
    def init_quotas(self):
        configs = json.loads(
            resources.files("chudgpt").joinpath("config.json").read_text(
                encoding="utf-8"
            )
        )
        gemini_configs = [
            Quota(slug=q["slug"], rpd=q["rpd"], rpm=q["rpm"], tpm=q["tpm"], inputs=q["inputs"])
            for q in configs["gemini"].values()
        ]

        session = sessionmaker(self.engine)
        with session() as db:
            db.execute(delete(ModelQuota))
            db.commit()

            db.add_all(
                [
                    ModelQuota(
                        model=q.slug,
                        rpd=q.rpd,
                        rpm=q.rpm,
                        tpm=q.tpm,
                        inputs=",".join(q.inputs),
                    )
                    for q in gemini_configs
                ]
            )
            db.commit()

    @db_exception_handler
    def flush_usage(self):
        session = sessionmaker(self.engine)
        with session() as db:
            # TODO: offload data to save it for analytics
            db.execute(delete(ModelQuota))
            db.commit()

    def db_browser_command(self) -> list[str] | None:
        target = str(self.db_path)

        if sys.platform == "darwin":
            for app in (
                Path(f"/Applications/{APP_NAME}.app"),
                Path.home() / "Applications" / f"{APP_NAME}.app",
            ):
                if app.exists():
                    return ["open", "-a", str(app), target]

        elif sys.platform == "win32":
            for name in (APP_NAME, "sqlitebrowser"):
                found = shutil.which(name)
                if found:
                    return [found, target]
            roots = (
                os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
                str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs"),
            )
            for root in roots:
                exe = Path(root) / APP_NAME / f"{APP_NAME}.exe"
                if exe.exists():
                    return [str(exe), target]

        found = shutil.which("sqlitebrowser")
        if found:
            return [found, target]
        if shutil.which("flatpak"):
            return ["flatpak", "run", "org.sqlitebrowser.sqlitebrowser", target]
        return None

    def launch_db_browser(self) -> bool:
        command = self.db_browser_command()
        if command is None:
            hint = INSTALL_HINTS.get(sys.platform, LINUX_HINT)
            print(f"\n[Error] {APP_NAME} could not be launched automatically.")
            print(f"Install it with:  {hint}")
            print(f"The database is at: {self.db_path}")
            return False
        try:
            subprocess.Popen(command)
        except OSError as error:
            print(f"\n[Error] could not start {APP_NAME}: {error}")
            return False
        return True
