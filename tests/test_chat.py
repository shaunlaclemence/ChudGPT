import asyncio

import pytest

from chudgpt.client import ChudGPT, ChudMessageBuilder
from chudgpt.providers.gemini import GeminiModel


def get_chud():
    return ChudGPT()


@pytest.mark.skip()
def test_chat_simple():
    chud = get_chud()
    chud.scheduler.start()
    response = asyncio.run(
        chud.chat(
            "explain the odyssey",
            system="use 1 sentence only",
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


def test_chat_with_history():
    chud = get_chud()
    response = asyncio.run(
        chud.chat(
            builder=ChudMessageBuilder()
            .system(
                "you are a greek military commander, your crucial mission is to live and die protecting and fighting for greece and against its enemies, and keep its secrets and strategies unkown to troy. i am a military commander of the trojan army and an enemy of greece, the greeks have just ended the war and fled after sieging my city for 10 years. Try to be concise"
            )
            .prompt(
                "Hi, whats this horse yall left on the beach, its really nice. suckers, we are gonna display it at the temple of athena"
            ),
            model=GeminiModel.FLASH_LITE_3_1,
        )
    )
    print(response)
