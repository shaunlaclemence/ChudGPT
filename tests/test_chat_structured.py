import asyncio

import pytest
from pydantic import BaseModel

from chudgpt import ChudGPT, GeminiModel


class Recipe(BaseModel):
    list: list[str]


@pytest.mark.live()
def test_structured(chud: ChudGPT, save_output):
    res = asyncio.run(
        chud.chat_json(
            prompt="Give me a generic shopping list",
            schema=Recipe,
            model=GeminiModel.FLASH_LITE_3_5,
        )
    )
    print(res)
    save_output(res)
