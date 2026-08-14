from __future__ import annotations

import json as _json
from functools import cached_property
from typing import Any, Generic, cast

from pydantic import BaseModel, ValidationError
from typing_extensions import TypeVar

from chudgpt.exceptions import ChudGPTBadDataException, ServiceCode

from .chud_message import ChudMessage, ChudMessageRole
from .chud_provider import ChudProvider
from .model_usage import Usage

ModelT = TypeVar("ModelT", bound=BaseModel)
# defaults to str, so a plain chat() reply is still ChudResponse with data: str
DataT = TypeVar("DataT", default=str)


class ChudResponse(BaseModel, Generic[DataT]):
    data: DataT
    service: str
    model: str
    usage: Usage
    provider: ChudProvider
    duration: float
    # wall-clock time the provider call took, in seconds

    @property
    def message(self) -> ChudMessage:
        return ChudMessage(role=ChudMessageRole.ASSISTANT, content=self.encoded)

    @property
    def encoded(self) -> str:
        """``data`` as the string the provider sent, or would accept back."""
        if isinstance(self.data, str):
            return self.data
        if isinstance(self.data, BaseModel):
            return self.data.model_dump_json()
        return _json.dumps(self.data)

    @cached_property
    def parsed_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json") | {"data": self.__decoded()}

    def parse(self, schema: type[ModelT]) -> ModelT:
        if isinstance(self.data, schema):
            return self.data
        try:
            return schema.model_validate(self.__decoded())
        except ValidationError as err:
            raise ChudGPTBadDataException(
                f"response does not match {schema.__name__}",
                ServiceCode.ROTOR_SERVICE,
                err,
            ) from err

    def typed(self, schema: type[ModelT]) -> ChudResponse[ModelT]:
        """Re-type a ``ChudResponse[str]`` by validating its JSON into ``schema``."""
        parameterised = cast("type[ChudResponse[ModelT]]", ChudResponse[schema])
        return parameterised(
            data=self.parse(schema),
            service=self.service,
            model=self.model,
            usage=self.usage,
            provider=self.provider,
            duration=self.duration,
        )

    def __decoded(self) -> Any:
        if isinstance(self.data, BaseModel):
            return self.data.model_dump(mode="json")
        if not isinstance(self.data, str):
            return self.data
        try:
            return _json.loads(self.data)
        except _json.JSONDecodeError as err:
            raise ChudGPTBadDataException(
                "response data is not valid JSON",
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
