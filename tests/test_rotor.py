import asyncio

import openai
import pytest
from httpx import Request as HttpRequest
from httpx import Response as HttpResponse

from chudgpt.exceptions import ChudGPTRateLimitException
from chudgpt.providers.gemini import GeminiModel
from chudgpt.schemas.chat import ChudMessage, ChudResponse
from chudgpt.services.db import DBService
from chudgpt.services.exceptions.rotor import (
    rotor_exception_handler,
    rotor_rotation_handler,
)
from chudgpt.services.files import FilesService
from chudgpt.services.rotor import RotorService
from chudgpt.utils.keys import load_secrets

APP_NAME = "Test App"


class TestRotor(RotorService):
    def rotate_provider(self):
        self._rotate_provider()

    def provider(self):
        return self._provider()

    @rotor_exception_handler
    async def test_chat(
        self,
        prompt: str | None = None,
        *,
        messages: list[ChudMessage] | None = None,
        system: str | None = None,
        model: GeminiModel | None = None,
        raise_error: BaseException | None = None,
    ) -> ChudResponse:
        if raise_error is not None:
            error = raise_error
            raise error
        return await self.chat(prompt, messages=messages, system=system, model=model)

    @rotor_rotation_handler
    @rotor_exception_handler
    async def test_chat_rotate(
        self,
        prompt: str | None = None,
        *,
        messages: list[ChudMessage] | None = None,
        system: str | None = None,
        model: GeminiModel | None = None,
        raise_error: BaseException | None = None,
    ) -> ChudResponse:
        if raise_error is not None:
            error = raise_error
            raise error
        return await self.chat(prompt, messages=messages, system=system, model=model)


def get_rotor() -> TestRotor:
    files = FilesService()
    files.set_app_name(APP_NAME)

    db = DBService(files)
    rotor = TestRotor(
        db_service=db,
        secrets=load_secrets(files.secrets_path()),
        timeout=30.0,
    )
    return rotor


@pytest.mark.skip("Tested")
def test_rotote():
    rotor = get_rotor()

    def __print():
        print("\n\nprovider: ", rotor.provider())
        print("provider_config: ", rotor.provider_config)

    __print()
    rotor.rotate_provider()
    __print()
    rotor.rotate_provider()
    __print()


@pytest.mark.skip("Tested")
def test_rotate_and_chat():
    rotor = get_rotor()

    res1 = asyncio.run(
        rotor.chat(
            "Explain the epic of Gilgamesh in 1 sentence",
            model=GeminiModel.FLASH_LITE_3_5,
        )
    )

    print(res1)

    rotor.rotate_provider()

    res2 = asyncio.run(
        rotor.chat(
            "Explain the Odyssey in 1 sentence",
            model=GeminiModel.FLASH_LITE_3_5,
        )
    )

    print(res2)


@pytest.mark.skip()
def test_raise_429():
    rotor = get_rotor()

    with pytest.raises(ChudGPTRateLimitException) as exc_info:
        asyncio.run(
            rotor.test_chat(
                "Explain the epic of Gilgamesh in 1 sentence",
                model=GeminiModel.FLASH_LITE_3_5,
                raise_error=openai.RateLimitError(
                    "Simulated AI Rate Limit Error",
                    response=HttpResponse(
                        429, request=HttpRequest("POST", "https://example.com")
                    ),
                    body=None,
                ),
            )
        )

    err = exc_info.value.error
    assert err is not None
    print(repr(exc_info.value))
    print(repr(err))
    print(vars(err))
    print("status_code:", err.status_code)
    print("body:", err.body)
    print("response:", err.response)


def test_rotate_on_rate_limit_exhausts_rotations_then_raises():
    rotor = get_rotor()

    rotate_calls = 0
    original_rotate = rotor._rotate_provider

    def counting_rotate():
        nonlocal rotate_calls
        rotate_calls += 1
        original_rotate()

    rotor._rotate_provider = counting_rotate

    with pytest.raises(ChudGPTRateLimitException):
        asyncio.run(
            rotor.test_chat_rotate(
                "Explain the epic of Gilgamesh in 1 sentence",
                model=GeminiModel.FLASH_LITE_3_5,
                raise_error=openai.RateLimitError(
                    "Simulated AI Rate Limit Error",
                    response=HttpResponse(
                        429, request=HttpRequest("POST", "https://example.com")
                    ),
                    body=None,
                ),
            )
        )

    assert rotate_calls == 3
