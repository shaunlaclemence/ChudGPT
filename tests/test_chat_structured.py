import asyncio

import pytest
from pydantic import BaseModel

from chudgpt import ChudGPT, GeminiModel, GeneratedCode, Language


class Recipe(BaseModel):
    list: list[str]


@pytest.mark.live()
def test_structured(chud: ChudGPT, save_output):
    res = asyncio.run(
        chud.text.chat_json(
            "Write a function that recursively obfusactes any field in a JSON. def obfuscate(keys: list[str], data: json)",
            schema=GeneratedCode.pin(language=Language.PYTHON),
            schema_name="GeneratedCode",
            model=GeminiModel.FLASH_LITE_3_5,
        )
    )
    print(res)
    save_output(res)


@pytest.mark.live()
def test_structured_returns_a_typed_response(chud: ChudGPT, save_output):
    res = asyncio.run(
        chud.text.chat_json(
            prompt="Give me a generic shopping list",
            schema=Recipe,
            model=GeminiModel.FLASH_LITE_3_5,
        )
    )
    save_output(res)

    assert isinstance(res.data, Recipe)
    assert res.usage.requests == 1
    assert res.model
