import subprocess
import sys
from typing import Any

from chudgpt._schemas.execution import ExecutionResult
from chudgpt._services.exceptions.executor import executor_exception_handler
from chudgpt._utils.executor import ExecutorRules
from chudgpt.exceptions import ChudGPTExecutionFailedException

TIMEOUT = 10.0


class ExecutorService:
    def __init__(self, timeout: float = TIMEOUT) -> None:
        self._timeout = timeout

    @executor_exception_handler
    def run(self, code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=self._timeout,
            check=False,
        )

    def call(
        self, code: str, entrypoint: str, *args: Any, **kwargs: Any
    ) -> ExecutionResult:
        completed = self.run(ExecutorRules.harness(code, entrypoint, args, kwargs))
        if completed.returncode != 0:
            raise ChudGPTExecutionFailedException(
                completed.stderr.strip() or f"exited with {completed.returncode}"
            )
        printed, value = ExecutorRules.split(completed.stdout)
        return ExecutionResult(value=value, stdout=printed, stderr=completed.stderr)
