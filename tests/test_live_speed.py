"""Live speed benchmark: every catalog model, same prompt, ranked by latency.

Opt-in — it costs one real API call per model, so it's skipped unless you ask:

    CHUDGPT_SPEED=1 uv run pytest tests/test_live_speed.py -s

(``-s`` matters: the table is printed, and pytest swallows stdout without it.)

Each model gets exactly one streaming call, which yields both latency numbers
from a single request: time-to-first-token (what a user perceives as "did it
hang?") and total wall time. Models the account can't reach are reported in the
table as failures rather than failing the run — a 404 for a retired model is a
finding, not a broken test.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from chudgpt import ChudClient, ChudGPTError
from chudgpt.keystore import load_keys_from_secrets_json

SECRETS_PATH = Path("secrets.json")
CATALOG_PATH = Path(__file__).resolve().parent.parent / "src" / "chudgpt" / "config.json"

PROMPT = "In exactly one sentence, explain what an API is."
TIER = "fast"

pytestmark = [
    pytest.mark.skipif(
        not SECRETS_PATH.exists(), reason="no secrets.json with real API keys present"
    ),
    pytest.mark.skipif(
        os.environ.get("CHUDGPT_SPEED") != "1",
        reason="speed benchmark is opt-in: set CHUDGPT_SPEED=1 (one API call per model)",
    ),
]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _targets() -> list[tuple[str, str]]:
    """(provider, model) for every catalog model we actually hold a key for."""
    have_keys = set(load_keys_from_secrets_json(SECRETS_PATH))
    catalog = json.loads(CATALOG_PATH.read_text())
    return [
        (provider, model)
        for provider, models in catalog.items()
        if provider in have_keys
        for model in models
    ]


async def _measure(provider: str, model: str, state_file: Path) -> dict:
    """One streaming call: time-to-first-token, total time, and token usage."""
    client = ChudClient(
        secrets_path=SECRETS_PATH, providers=[provider], state_file=state_file
    )
    start = time.perf_counter()
    ttft: float | None = None
    text: list[str] = []
    usage: dict | None = None

    async for chunk in client.stream(PROMPT, tier=TIER, model=model):
        if chunk.done:
            usage = chunk.usage
            continue
        if ttft is None:
            ttft = time.perf_counter() - start
        text.append(chunk.delta)

    total = time.perf_counter() - start
    completion = (usage or {}).get("completion_tokens", 0)
    return {
        "provider": provider,
        "model": model,
        "ok": True,
        "ttft": ttft,
        "total": total,
        "tok_per_s": (completion / total) if completion and total else None,
        "usage": usage,
        "chars": len("".join(text)),
    }


@pytest.mark.anyio
async def test_model_speed_benchmark(tmp_path):
    targets = _targets()
    assert targets, "no catalog models match the providers in secrets.json"

    results = []
    for provider, model in targets:
        # isolated state per model so one model's cooldown never silently skips
        # the next one's request
        state_file = tmp_path / f"{provider}_{model.replace('/', '_')}.json"
        try:
            results.append(await _measure(provider, model, state_file))
        except ChudGPTError as e:
            reason = str(e)
            if "404" in reason:
                reason = "unavailable to this account (404)"
            elif "429" in reason:
                reason = "rate limited / quota exhausted (429)"
            results.append(
                {
                    "provider": provider,
                    "model": model,
                    "ok": False,
                    "reason": reason[:70],
                }
            )

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    ok.sort(key=lambda r: r["total"])

    print(f"\n\nprompt: {PROMPT!r}   tier={TIER}   models={len(results)}\n")
    header = f"{'model':<38} {'ttft':>7} {'total':>8} {'tok/s':>7} {'in':>5} {'out':>5} {'chars':>6}"
    print(header)
    print("-" * len(header))
    for r in ok:
        u = r["usage"] or {}
        ttft = f"{r['ttft']:.2f}s" if r["ttft"] is not None else "-"
        tps = f"{r['tok_per_s']:.1f}" if r["tok_per_s"] else "-"
        print(
            f"{r['model']:<38} {ttft:>7} {r['total']:>7.2f}s {tps:>7} "
            f"{u.get('prompt_tokens', 0):>5} {u.get('completion_tokens', 0):>5} {r['chars']:>6}"
        )

    if failed:
        print(f"\nunavailable ({len(failed)}):")
        for r in failed:
            print(f"  {r['model']:<38} {r['reason']}")

    if ok:
        print(f"\nfastest: {ok[0]['model']} ({ok[0]['total']:.2f}s total)")
        timed = [r for r in ok if r["ttft"] is not None]
        if timed:
            best = min(timed, key=lambda r: r["ttft"])
            print(f"snappiest first token: {best['model']} ({best['ttft']:.2f}s)")
        silent = [r for r in ok if r["ttft"] is None]
        if silent:
            # completed the request but streamed no content — worth knowing, since
            # it looks like success to the caller but produces an empty reply
            print(f"returned no content: {', '.join(r['model'] for r in silent)}")
    print()

    assert ok, f"every model failed: {[(r['model'], r['reason']) for r in failed]}"
