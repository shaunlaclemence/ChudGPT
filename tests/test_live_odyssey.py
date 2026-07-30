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
        model=Model.GEMINI_3_5_FLASH_LITE,
    )
    print(f"\n[{reply.provider}/{reply.model}]\n{reply.text}\n")
    print(f"usage: {reply.usage}")
    assert reply.text.strip()


def test_usage_reports_limits_progress_and_provider():
    client = ChudClient(secrets_path=SECRETS_PATH)

    reply = client.ask("Say hi in five words.")
    print(f"\nserved by: {reply.provider}/{reply.model}")
    print(f"tokens this call: {reply.usage}")

    usage = client.usage()
    served_kid = next(kid for kid, u in usage.items() if u.provider == reply.provider)
    served = usage[served_kid]

    print("\ncurrent limits / progress per key:")
    for kid, u in usage.items():
        cap = f"{u.requests_today}/{u.known_rpd}" if u.known_rpd else str(u.requests_today)
        pct = f"{u.percent_used}%" if u.percent_used is not None else "unknown cap"
        print(
            f"  {kid} [{u.provider}] status={u.status} requests={cap} "
            f"({pct}) tokens_today={u.tokens_today} resets_at={u.resets_at.isoformat()}"
        )

    assert served.requests_today >= 1
    assert served.status == "ok"
    if served.percent_used is not None:
        assert 0 <= served.percent_used <= 100
