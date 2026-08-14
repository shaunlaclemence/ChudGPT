from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from chudgpt._schemas import ChudMessage
from chudgpt.exceptions import ChudGPTBadDataException, ServiceCode
from chudgpt.messages import ChudMessageBuilder


class ChatRules:
    @staticmethod
    def turn(
        prompt: str | None,
        messages: list[ChudMessage] | None,
        system: str | None,
        builder: ChudMessageBuilder | None,
    ) -> tuple[str | None, list[ChudMessage] | None, str | None]:
        if builder is None:
            return prompt, messages, system
        if prompt is not None or messages is not None or system is not None:
            raise ChudGPTBadDataException(
                "pass builder on its own, not alongside prompt/messages/system",
                ServiceCode.ROTOR_SERVICE,
            )
        return None, builder.messages_list, None

    @staticmethod
    def guard_models(
        builders: dict[str, ChudMessageBuilder], models: dict[str, Any]
    ) -> None:
        missing = sorted(set(builders) - set(models))
        if missing:
            raise ChudGPTBadDataException(
                f"no model given for sub-agent(s): {missing}", ServiceCode.ROTOR_SERVICE
            )

    @staticmethod
    def response_format(
        schema: type[BaseModel] | dict[str, Any], schema_name: str | None
    ) -> dict[str, Any]:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_name = schema_name or schema.__name__
            schema = schema.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name or "response", "schema": schema},
        }
