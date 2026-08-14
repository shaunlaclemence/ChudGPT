import pytest

from chudgpt._services.executor import ExecutorService
from chudgpt.exceptions import (
    ChudGPTBadDataException,
    ChudGPTExecutionFailedException,
    ChudGPTExecutionTimeoutException,
)

OBFUSCATE = """
import typing


def obfuscate(keys: list[str], data: typing.Any) -> typing.Any:
    print("HI")
    if isinstance(data, dict):
        return {
            k: "[OBFUSCATED]" if k in keys else obfuscate(keys, v)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [obfuscate(keys, item) for item in data]
    return data
"""


def test_call_passes_args_and_returns_the_value(save_output):
    payload = {
        "user": "shaun",
        "password": "hunter2",
        "nested": [{"api_key": "sk-123", "keep": 1}],
    }

    result = ExecutorService().call(
        OBFUSCATE, "obfuscate", ["password", "api_key"], payload
    )

    save_output(result)


def test_call_accepts_keyword_arguments():
    result = ExecutorService().call(
        OBFUSCATE, "obfuscate", keys=["password"], data={"password": "hunter2"}
    )

    assert result.value == {"password": "[OBFUSCATED]"}


def test_call_separates_printed_output_from_the_result():
    code = "def go():\n    print('working')\n    return 42"

    result = ExecutorService().call(code, "go")

    assert result.value == 42
    assert result.stdout.strip() == "working"


def test_call_raises_when_the_code_raises():
    code = "def go():\n    raise RuntimeError('boom')"

    with pytest.raises(ChudGPTExecutionFailedException) as raised:
        ExecutorService().call(code, "go")

    assert "boom" in str(raised.value)
    assert raised.value.service_code.value == "005"
    assert raised.value.error_code == "422"


def test_call_raises_on_a_missing_entrypoint():
    with pytest.raises(ChudGPTExecutionFailedException) as raised:
        ExecutorService().call(OBFUSCATE, "nope", [], {})

    assert "nope" in str(raised.value)


def test_call_rejects_a_bad_entrypoint_name_before_running():
    with pytest.raises(ChudGPTBadDataException, match="not a valid function name"):
        ExecutorService().call(OBFUSCATE, "obfuscate(); import os", [], {})


def test_call_rejects_arguments_that_cannot_be_serialised():
    with pytest.raises(ChudGPTBadDataException, match="not JSON serialisable"):
        ExecutorService().call(OBFUSCATE, "obfuscate", [], object())


def test_call_times_out_on_an_endless_function():
    code = "def go():\n    while True:\n        pass"

    with pytest.raises(ChudGPTExecutionTimeoutException):
        ExecutorService(timeout=1.0).call(code, "go")


def test_run_still_executes_a_bare_script():
    completed = ExecutorService().run("print('hello')")

    assert completed.returncode == 0
    assert completed.stdout.strip() == "hello"
