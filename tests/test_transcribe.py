import base64

import pytest

from chudgpt.client import ChudClient, _audio_message
from chudgpt.config import AUDIO_PROVIDERS
from chudgpt.errors import InvalidRequestError
from chudgpt.rotor import Response


def test_audio_message_from_bytes_builds_input_audio_part():
    msg = _audio_message(b"\x00\x01\x02", "MP3", "transcribe this")

    assert msg["role"] == "user"
    text_part, audio_part = msg["content"]
    assert text_part == {"type": "text", "text": "transcribe this"}
    assert audio_part["type"] == "input_audio"
    assert audio_part["input_audio"]["format"] == "mp3"  # lower-cased, dot stripped
    assert base64.b64decode(audio_part["input_audio"]["data"]) == b"\x00\x01\x02"


def test_audio_message_infers_format_from_path(tmp_path):
    clip = tmp_path / "call.FLAC"
    clip.write_bytes(b"xyz")

    msg = _audio_message(clip, None, "hi")

    assert msg["content"][1]["input_audio"]["format"] == "flac"


def test_audio_message_requires_format_for_raw_bytes():
    with pytest.raises(InvalidRequestError):
        _audio_message(b"xyz", None, "hi")


def test_transcribe_restricts_to_audio_providers(monkeypatch, tmp_path):
    client = ChudClient(
        keys={"gemini": ["k"]},
        providers=["gemini"],
        state_file=tmp_path / "state.json",
    )
    captured: dict = {}

    def fake_chat(*, messages, tier, model, providers, **kwargs):
        captured.update(messages=messages, providers=providers, model=model)
        return Response(text="ok", provider="gemini", model="m")

    monkeypatch.setattr(client._rotor, "chat", fake_chat)

    reply = client.transcribe(b"audio-bytes", prompt="P", audio_format="mp3")

    assert reply.text == "ok"
    # Defaults to the audio-capable providers, never a text-only key.
    assert captured["providers"] == AUDIO_PROVIDERS
    # The audio rides as an input_audio content part alongside the prompt.
    assert captured["messages"][0]["content"][1]["type"] == "input_audio"
