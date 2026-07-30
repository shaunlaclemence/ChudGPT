"""Live error-path suite: every ChudClient failure mode, against the real API.

Hits the network using the real keys in secrets.json, so it's skipped when that
file is absent. Complements the mocked suite: these assert that what the real
providers actually return maps onto the chudgpt exception hierarchy, which is
where the mocks can lie (e.g. Gemini answers a bad API key with 400, not 401).

Quota-exhaustion paths are deliberately NOT provoked here — the only honest way
to trigger a real 429 across every key is to burn the day's free quota, which
would leave the account unusable. AllProvidersExhausted is covered by the mocked
suite in test_rotor.py instead.
"""

from pathlib import Path

import pytest

from chudgpt import (
    AllProvidersExhausted,
    ChudClient,
    ChudGPTError,
    ConfigError,
    InvalidRequestError,
    InvalidTierError,
    SecretsFileError,
    UnknownProviderError,
)

SECRETS_PATH = Path("secrets.json")

pytestmark = pytest.mark.skipif(
    not SECRETS_PATH.exists(), reason="no secrets.json with real API keys present"
)


@pytest.fixture
def state_file(tmp_path):
    """Isolated quota state per test.

    Without this, a cooldown persisted by one test (or an earlier run) into the
    shared ~/.chudgpt/state.json makes later tests skip the key before any request
    goes out — so a network-path test would pass without touching the network.
    """
    return tmp_path / "state.json"


def client(state_file=None, **kwargs):
    return ChudClient(secrets_path=SECRETS_PATH, state_file=state_file, **kwargs)


# --- construction-time errors (no network) ---------------------------------


def test_unknown_tier_at_construction():
    with pytest.raises(InvalidTierError) as exc:
        client(tier="turbo")
    assert exc.value.tier == "turbo"
    assert "best" in exc.value.known and "fast" in exc.value.known


def test_unknown_provider_name():
    with pytest.raises(UnknownProviderError) as exc:
        client(providers=["gemeni"])  # typo
    assert exc.value.name == "gemeni"
    assert "gemini" in exc.value.known


def test_missing_secrets_file():
    with pytest.raises(SecretsFileError) as exc:
        ChudClient(secrets_path="definitely-not-here.json")
    assert "definitely-not-here.json" in exc.value.path


def test_malformed_secrets_file(tmp_path):
    bad = tmp_path / "secrets.json"
    bad.write_text('{"gemini": [{"nope": 1}]}')
    with pytest.raises(SecretsFileError):
        ChudClient(secrets_path=bad)


def test_empty_secrets_file(tmp_path):
    empty = tmp_path / "secrets.json"
    empty.write_text("{}")
    with pytest.raises(SecretsFileError):
        ChudClient(secrets_path=empty)


def test_no_keys_at_all_is_config_error():
    with pytest.raises(ConfigError):
        ChudClient(keys={})


def test_keys_for_unknown_provider_only():
    with pytest.raises(ConfigError):
        ChudClient(keys={"not-a-provider": ["x"]})


# --- request-shape errors (no network) -------------------------------------


def test_neither_prompt_nor_messages():
    with pytest.raises(InvalidRequestError):
        client().ask()


def test_both_prompt_and_messages():
    with pytest.raises(InvalidRequestError):
        client().ask("hi", messages=[{"role": "user", "content": "hi"}])


def test_unknown_tier_per_call():
    with pytest.raises(InvalidTierError):
        client().ask("hi", tier="cheapest")


def test_unknown_tier_on_conversation():
    with pytest.raises(InvalidTierError):
        client().start_conversation(tier="nope")


# --- real provider responses ----------------------------------------------


def test_pinned_model_that_does_not_exist(state_file):
    """A 404 for an unknown model id is a caller error, not a quota problem."""
    with pytest.raises(InvalidRequestError) as exc:
        client(state_file, providers=["gemini"]).ask(
            "hi", model="gemini-99-does-not-exist"
        )
    assert "gemini" in str(exc.value)


def test_invalid_api_key_is_contained_in_hierarchy(state_file):
    """Gemini returns 400 (not 401) for a junk key; it must still rotate/cool down.

    With only one bad key configured there's nothing to rotate to, so the pool
    exhausts — but as AllProvidersExhausted, never a raw openai exception.
    """
    bad = ChudClient(
        keys={"gemini": ["sk-not-a-real-key"]},
        providers=["gemini"],
        state_file=state_file,
    )
    with pytest.raises(AllProvidersExhausted) as exc:
        bad.ask("hi")
    assert any("authentication" in reason for reason in exc.value.statuses.values())


def test_bad_request_kwarg_surfaces_as_invalid_request(state_file):
    with pytest.raises(InvalidRequestError):
        client(state_file, providers=["gemini"]).ask("hi", temperature=99999)


def test_every_error_derives_from_chudgpt_error():
    """The whole point of the hierarchy: one except clause can catch everything."""
    for call in (
        lambda: client(tier="turbo"),
        lambda: client(providers=["gemeni"]),
        lambda: ChudClient(secrets_path="definitely-not-here.json"),
        lambda: ChudClient(keys={}),
        lambda: client().ask(),
        lambda: client().ask("hi", tier="cheapest"),
    ):
        with pytest.raises(ChudGPTError):
            call()
