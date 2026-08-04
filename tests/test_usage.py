import asyncio
from pathlib import Path

from chudgpt.client import ChudGPT
from chudgpt.providers.gemini import GeminiModel

SECRETS_PATH = Path(__file__).resolve().parent.parent / "secrets.json"
MODEL = GeminiModel.FLASH_LITE_3_5


def get_chud(tmp_path, monkeypatch):
    """A client whose quota db is a throwaway, so tests never touch the real one."""
    monkeypatch.setenv("CHUDGPT_DB", str(tmp_path / "quota.db"))
    return ChudGPT(secrets_path=SECRETS_PATH)


def test_usage_starts_empty_then_banks_a_request(tmp_path, monkeypatch):
    chud = get_chud(tmp_path, monkeypatch)

    async def run():
        before = await chud.usage(MODEL)
        response = await chud.chat("Say hi.", model=MODEL)
        return before, response, await chud.usage(MODEL)

    before, response, after = asyncio.run(run())
    print(f"\nbefore : {before}\nafter  : {after}\nchat   : {response.usage}")

    assert before is not None, "ensure_ready should have created the row"
    assert after is not None
    assert before.model == MODEL.slug
    assert before.requests == 0

    assert after.requests == 1
    assert after.prompt_tokens == response.usage.prompt
    assert after.completion_tokens == response.usage.completion
    assert after.total_tokens == response.usage.total


def test_usage_defaults_to_the_model_chat_would_pick(tmp_path, monkeypatch):
    chud = get_chud(tmp_path, monkeypatch)

    row = asyncio.run(chud.usage())

    assert row is not None
    assert row.model == GeminiModel.cheapest().slug


def test_usage_accumulates_across_calls(tmp_path, monkeypatch):
    chud = get_chud(tmp_path, monkeypatch)

    async def run():
        first = await chud.chat("Say hi.", model=MODEL)
        second = await chud.chat("Say bye.", model=MODEL)
        return first, second, await chud.usage(MODEL)

    first, second, row = asyncio.run(run())

    assert row is not None
    assert row.requests == 2
    assert row.total_tokens == first.usage.total + second.usage.total


def test_usage_is_tracked_per_model(tmp_path, monkeypatch):
    chud = get_chud(tmp_path, monkeypatch)

    async def run():
        await chud.chat("Say hi.", model=MODEL)
        return await chud.usage(MODEL), await chud.usage(GeminiModel.FLASH_2_5)

    used, untouched = asyncio.run(run())

    assert used is not None and untouched is not None
    assert used.requests == 1
    assert untouched.requests == 0, "spend on one model must not affect another"
