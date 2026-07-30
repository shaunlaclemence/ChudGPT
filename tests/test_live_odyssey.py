"""Live smoke test: asks a real provider to summarize the Odyssey.

Unlike the rest of the suite (mocked, no keys needed), this hits the network
using whatever real keys are in secrets.json. It's skipped automatically when
that file isn't present, so it never blocks `uv run pytest` for anyone else.
"""

from pathlib import Path

import pytest

from chudgpt import ChudClient, Model

SECRETS_PATH = Path("secrets.json")

pytestmark = pytest.mark.skipif(
    not SECRETS_PATH.exists(), reason="no secrets.json with real API keys present"
)


def test_gemini_2_5_flash():
    client = ChudClient(secrets_path=SECRETS_PATH)
    reply = client.ask(
        "Summarize Homer's The Odyssey in three sentences.",
        model=Model.GEMINI_3_1_FLASH_LITE,
    )
    print(f"\n[{reply.provider}/{reply.model}]\n{reply.text}\n")
    print(f"usage: {reply.usage}")
    assert reply.text.strip()
