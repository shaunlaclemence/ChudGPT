from datetime import UTC, datetime

import pytest

from chudgpt.config import ProviderConfig

NOW = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def now():
    return NOW


@pytest.fixture
def providers():
    """Two fake OpenAI-compatible providers on distinct test hosts."""
    return (
        ProviderConfig(
            name="alpha",
            base_url="https://alpha.test/v1",
            env_var="ALPHA_API_KEY",
            models={"best": "alpha-big", "fast": "alpha-small"},
            priority=1,
            known_rpd=100,
        ),
        ProviderConfig(
            name="beta",
            base_url="https://beta.test/v1",
            env_var="BETA_API_KEY",
            models={"best": "beta-big", "fast": "beta-small"},
            priority=2,
            reset="midnight_pt",
        ),
    )


def completion_json(model: str, text: str = "hello") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
