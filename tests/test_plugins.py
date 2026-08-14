import importlib
import sys

import pytest

from chudgpt import ChudGPT
from chudgpt._utils.plugins import PluginRegistry


class BlockedFinder:
    def __init__(self, blocked: str) -> None:
        self.blocked = blocked

    def find_module(self, name, path=None):
        return self if name == self.blocked else None

    def load_module(self, name):
        raise ImportError(f"{name} is blocked")


@pytest.fixture()
def without_soundfile():
    modules = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "soundfile" or name.startswith("chudgpt.audio")
    }
    finder = BlockedFinder("soundfile")
    sys.meta_path.insert(0, finder)
    yield
    sys.meta_path.remove(finder)
    sys.modules.update(modules)


def test_audio_attaches_when_the_extra_is_installed(chud: ChudGPT):
    assert chud.audio is not None
    assert hasattr(chud.audio, "diarize")
    assert hasattr(chud.audio, "diarize_stream")
    assert hasattr(chud.audio, "transcribe")


def test_text_is_the_only_namespace_needing_no_extra(chud: ChudGPT):
    assert hasattr(chud.text, "chat")
    assert hasattr(chud.text, "chat_json")
    assert hasattr(chud.text, "parallel_chat")
    assert hasattr(chud.text, "stream")
    assert hasattr(chud.text, "parallel_stream")


def test_services_are_not_importable_from_the_public_api():
    import chudgpt
    import chudgpt.audio

    for name in ("RotorService", "TextService", "DBService", "FilesService"):
        assert not hasattr(chudgpt, name)
    for name in (
        "AudioService",
        "AudioChunker",
        "AudioDiarizer",
        "AudioTranscriber",
        "VoiceActivity",
    ):
        assert not hasattr(chudgpt.audio, name)


def test_every_audio_capability_is_reachable_from_the_namespace(chud: ChudGPT):
    for name in (
        "diarize",
        "diarize_stream",
        "transcribe",
        "voice_activity",
        "chunks",
        "builders",
    ):
        assert hasattr(chud.audio, name)


def test_audio_is_absent_without_the_extra(without_soundfile):
    assert PluginRegistry().attach(object()) == {}
    with pytest.raises(ImportError):
        importlib.import_module("chudgpt.audio")


def test_missing_plugin_names_the_extra_to_install():
    assert "chudgpt[audio]" in PluginRegistry.missing("audio")
    assert "no attribute" in PluginRegistry.missing("video")


def test_unbound_client_refuses_to_hand_out_text():
    from chudgpt.exceptions import ChudGPTNotFoundException

    with pytest.raises(ChudGPTNotFoundException):
        assert ChudGPT().text
