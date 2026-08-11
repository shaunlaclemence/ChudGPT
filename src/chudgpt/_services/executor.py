import subprocess
import sys

TIMEOUT = 10.0


class ExecutorService:
    def __init__(self) -> None:
        pass

    def run(self, code: str) -> subprocess.CompletedProcess[str]:
        res = subprocess.run(
            [sys.executable, "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
        return res
