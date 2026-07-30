import json

import pytest
import respx
from httpx import Response as HttpResponse

from chudgpt.client import ChudClient
from chudgpt.keystore import load_keys_from_secrets_json

ALPHA_URL = "https://alpha.test/v1/chat/completions"


def sse(*lines: str) -> bytes:
    return ("".join(f"data: {line}\n\n" for line in lines) + "data: [DONE]\n\n").encode()


def chunk(model: str, content: str | None = None, usage: dict | None = None) -> str:
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
    body = sse(*[chunk(model, content=c) for c in text], chunk(model, usage=usage))
    return HttpResponse(200, content=body, headers={"content-type": "text/event-stream"})


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_conversation_send_grows_history_across_turns(providers, tmp_path, now):
    with respx.mock:
        respx.post(ALPHA_URL).mock(
            return_value=sse_response(
                "alpha-small", "hi", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            )
        )
        client = ChudClient(
            keys={"alpha": ["sk-alpha"]},
            providers=providers,
            state_file=tmp_path / "state.json",
            now=lambda: now,
        )
        convo = client.start_conversation()

        first = [c async for c in convo.send("hello")]
        assert "".join(first) == "hi"
        assert convo.messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert convo.last_meta["provider"] == "alpha"
        assert convo.last_meta["usage"]["total_tokens"] == 2

        second = [c async for c in convo.send("again")]
        assert "".join(second) == "hi"
        assert convo.messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "hi"},
        ]


def test_load_keys_from_secrets_json(tmp_path):
    path = tmp_path / "secrets.json"
    path.write_text(
        json.dumps(
            {
                "gemini": [
                    {
                        "account": "work@example.com",
                        "name": "work",
                        "project_name": "proj-a",
                        "project_number": 111,
                        "api_key": "key-a",
                    },
                    {
                        "account": "personal@example.com",
                        "name": "personal",
                        "project_name": "proj-b",
                        "project_number": 222,
                        "api_key": "key-b",
                    },
                ]
            }
        )
    )
    keys = load_keys_from_secrets_json(path)
    assert keys == {"gemini": ["key-a", "key-b"]}
