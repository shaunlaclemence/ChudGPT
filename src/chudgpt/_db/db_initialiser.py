import os
import shutil
import subprocess
import sys
from pathlib import Path

from chudgpt._services.db import DBService
from chudgpt._services.files import FilesService

APP_NAME = "DB Browser for SQLite"

INSTALL_HINTS = {
    "win32": f"winget install DBBrowserForSQLite.{APP_NAME.replace(' ', '')}",
    "darwin": "brew install --cask db-browser-for-sqlite",
}
LINUX_HINT = "sudo apt install sqlitebrowser"


class DBInitialiser:
    def __init__(self, app_name: str) -> None:
        self.engine = None
        files_service = FilesService()
        self.app_name = app_name
        self.db_service = DBService(files_service, app_name)
        self.files_service = files_service

    def db_browser_command(self) -> list[str] | None:
        target = str(self.files_service.db_path(self.app_name))

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
            print(f"The database is at: {self.files_service.db_path(self.app_name)}")
            return False
        try:
            subprocess.Popen(command)
        except OSError as error:
            print(f"\n[Error] could not start {APP_NAME}: {error}")
            return False
        return True
