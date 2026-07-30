import pytest
import respx
from httpx import Response as HttpResponse

from chudgpt.errors import AllProvidersExhausted
from chudgpt.rotor import Rotor

KEYS = {"alpha": ["sk-alpha"], "beta": ["sk-beta"]}

ALPHA_URL = "https://alpha.test/v1/chat/completions"
BETA_URL = "https://beta.test/v1/chat/completions"


def make_rotor(providers, tmp_path, now, keys=None):
    return Rotor(
        keys or dict(KEYS),
        providers=providers,
        state_file=tmp_path / "state.json",
        now=lambda: now,
    )


def sse(*lines: str) -> bytes:
    return ("".join(f"data: {line}\n\n" for line in lines) + "data: [DONE]\n\n").encode()


def chunk(model: str, content: str | None = None, usage: dict | None = None) -> str:
    import json

    return json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": (
                [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
                if content is not None
                else []
            ),
            "usage": usage,
        }
    )


def sse_response(model: str, text: str, usage: dict) -> HttpResponse:
    body = sse(
        *[chunk(model, content=c) for c in text],
        chunk(model, usage=usage),
    )
    return HttpResponse(
        200, content=body, headers={"content-type": "text/event-stream"}
    )


@pytest.mark.anyio
async def test_stream_yields_deltas_and_final_usage(providers, tmp_path, now):
    with respx.mock:
        respx.post(ALPHA_URL).mock(
            return_value=sse_response(
                "alpha-small",
                "hi",
                {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            )
        )
        rotor = make_rotor(providers, tmp_path, now)
        deltas = []
        final = None
        async for c in rotor.chat_stream("hi"):
            if c.done:
                final = c
            else:
                deltas.append(c.delta)
        assert "".join(deltas) == "hi"
        assert final is not None
        assert final.usage["total_tokens"] == 7
        assert final.provider == "alpha"


@pytest.mark.anyio
async def test_stream_rotates_to_next_provider_on_429(providers, tmp_path, now):
    with respx.mock:
        respx.post(ALPHA_URL).mock(
            return_value=HttpResponse(429, json={"error": {"message": "quota"}})
        )
        respx.post(BETA_URL).mock(
            return_value=sse_response(
                "beta-small",
                "yo",
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )
        )
        rotor = make_rotor(providers, tmp_path, now)
        deltas = []
        async for c in rotor.chat_stream("hi"):
            if not c.done:
                deltas.append(c.delta)
        assert "".join(deltas) == "yo"
        status = rotor.status()
        alpha_kid = next(k for k in status if k.startswith("alpha:"))
        assert "cooling down" in status[alpha_kid]


@pytest.mark.anyio
async def test_stream_all_exhausted_raises(providers, tmp_path, now):
    with respx.mock:
        respx.post(ALPHA_URL).mock(
            return_value=HttpResponse(429, json={"error": {"message": "quota"}})
        )
        respx.post(BETA_URL).mock(
            return_value=HttpResponse(429, json={"error": {"message": "quota"}})
        )
        rotor = make_rotor(providers, tmp_path, now)
        with pytest.raises(AllProvidersExhausted):
            async for _ in rotor.chat_stream("hi"):
                pass


@pytest.fixture
def anyio_backend():
    return "asyncio"
