from __future__ import annotations

import json as _json
from functools import cached_property
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from chudgpt.exceptions import ChudGPTBadDataException, ServiceCode

from .chud_message import ChudMessage, ChudMessageRole
from .chud_provider import ChudProvider
from .model_usage import Usage

ModelT = TypeVar("ModelT", bound=BaseModel)


class ChudResponse(BaseModel):
    data: str
    service: str
    model: str
    usage: Usage
    provider: ChudProvider
    duration: float
    # wall-clock time the provider call took, in seconds

    @property
    def message(self) -> ChudMessage:
        return ChudMessage(role=ChudMessageRole.ASSISTANT, content=self.data)

    @cached_property
    def parsed_json(self) -> dict[str, Any]:
        try:
            return self.model_dump(mode="json") | {"data": _json.loads(self.data)}
        except _json.JSONDecodeError as err:
            raise ChudGPTBadDataException(
                "response data is not valid JSON",
                ServiceCode.ROTOR_SERVICE,
                err,
            ) from err

    def parse(self, model: type[ModelT]) -> ModelT:
        try:
            return model.model_validate(self.parsed_json["data"])
        except ValidationError as err:
            raise ChudGPTBadDataException(
                f"response does not match {model.__name__}",
                ServiceCode.ROTOR_SERVICE,
                err,
            ) from err

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"data:     {self.data!r}",
                f"service:  {self.service}",
                f"model:    {self.model}",
                f"usage:    {self.usage}",
                f"provider: {self.provider}",
                f"duration: {self.duration:.3f}s",
            ]
        )
