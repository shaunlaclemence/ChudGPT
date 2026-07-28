import json

import pytest

from chudgpt.errors import ConfigError
from chudgpt.keystore import key_id, load_keys


def test_loads_from_env(providers, tmp_path):
    env = {"ALPHA_API_KEY": "sk-one", "BETA_API_KEY": "sk-two"}
    keys = load_keys(providers, env=env, keys_file=tmp_path / "keys.json")
    assert keys == {"alpha": ["sk-one"], "beta": ["sk-two"]}


def test_comma_separated_multiple_keys(providers, tmp_path):
    env = {"ALPHA_API_KEY": "sk-one, sk-two ,sk-three"}
    keys = load_keys(providers, env=env, keys_file=tmp_path / "keys.json")
    assert keys["alpha"] == ["sk-one", "sk-two", "sk-three"]


def test_file_fallback_and_env_precedence(providers, tmp_path):
    f = tmp_path / "keys.json"
    f.write_text(json.dumps({"alpha": ["file-key"], "beta": "file-single"}))
    keys = load_keys(providers, env={}, keys_file=f)
    assert keys == {"alpha": ["file-key"], "beta": ["file-single"]}
    # env wins over the file for the same provider
    keys = load_keys(providers, env={"ALPHA_API_KEY": "env-key"}, keys_file=f)
    assert keys["alpha"] == ["env-key"]
    assert keys["beta"] == ["file-single"]


def test_no_keys_raises_config_error(providers, tmp_path):
    with pytest.raises(ConfigError, match="no API keys found"):
        load_keys(providers, env={}, keys_file=tmp_path / "keys.json")


def test_key_id_hides_raw_key():
    kid = key_id("alpha", "sk-supersecret")
    assert "sk-supersecret" not in kid
    assert kid.startswith("alpha:")
    assert kid == key_id("alpha", "sk-supersecret")  # stable
    assert kid != key_id("alpha", "sk-other")
