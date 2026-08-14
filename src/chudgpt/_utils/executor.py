from __future__ import annotations

import json
from typing import Any

from chudgpt.exceptions import ChudGPTBadDataException, ServiceCode


class ExecutorRules:
    # the called function's return value shares stdout with anything the code
    # prints, so the marker is what separates the two
    RESULT_MARKER = "\n__CHUD_RESULT__"

    @classmethod
    def harness(
        cls, code: str, entrypoint: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> str:
        if not entrypoint.isidentifier():
            raise cls.__bad(f"not a valid function name: {entrypoint!r}")
        payload = cls.__encode({"args": list(args), "kwargs": kwargs})
        return (
            f"{code}\n\n"
            "import json as __chud_json, sys as __chud_sys\n"
            f"__chud_call = __chud_json.loads({payload!r})\n"
            f"__chud_value = {entrypoint}("
            "*__chud_call['args'], **__chud_call['kwargs'])\n"
            f"__chud_sys.stdout.write({cls.RESULT_MARKER!r})\n"
            "__chud_sys.stdout.write(__chud_json.dumps(__chud_value))\n"
        )

    @classmethod
    def split(cls, stdout: str) -> tuple[str, Any]:
        printed, marker, encoded = stdout.rpartition(cls.RESULT_MARKER)
        if not marker:
            raise cls.__bad("the code produced no result; did the function return?")
        try:
            return printed, json.loads(encoded)
        except json.JSONDecodeError as err:
            raise cls.__bad(f"result is not valid JSON: {encoded[:200]!r}", err)

    @classmethod
    def __encode(cls, payload: dict[str, Any]) -> str:
        try:
            return json.dumps(payload)
        except TypeError as err:
            raise cls.__bad(f"arguments are not JSON serialisable: {err}", err)

    @staticmethod
    def __bad(message: str, error: Any | None = None) -> ChudGPTBadDataException:
        return ChudGPTBadDataException(message, ServiceCode.EXECUTOR_SERVICE, error)
