import asyncio
from pathlib import Path

from chudgpt.client import ChudGPT
from chudgpt.providers.gemini import GeminiModel
import pytest

SECRETS_PATH = Path(__file__).resolve().parent.parent / "secrets.json"


def get_chud():
    return ChudGPT(secrets_path=SECRETS_PATH)

@pytest.mark.skip()
def test_chat_simple():
    chud = get_chud()
    response = asyncio.run(
        chud.chat(
            "generate a random image",
            system="",
            model=GeminiModel.FLASH_LITE_3_5,
        )
    )

    print(response)

    assert response.text
    assert response.service == "gemini"
    assert response.provider.api_key
    assert response.usage.requests == 1
    assert response.usage.prompt > 0
    assert response.usage.total >= response.usage.prompt + response.usage.completion

def test_chat_usage():
    chud = get_chud()
    model = GeminiModel.FLASH_LITE_3_1
    response = asyncio.run(
        chud.chat(
            "explain the odyssey",
            system="use 1 sentence only",
            model=model,
        )
    )

    print(response)

    usage = asyncio.run(
        chud.usage(model=model)
    )
    print(usage)
