from datetime import timedelta

import pytest
import respx
from conftest import completion_json
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


@respx.mock
def test_uses_highest_priority_provider(providers, tmp_path, now):
    route = respx.post(ALPHA_URL).mock(
        return_value=HttpResponse(200, json=completion_json("alpha-small"))
    )
    rotor = make_rotor(providers, tmp_path, now)
    reply = rotor.chat("hi")
    assert reply.text == "hello"
    assert reply.provider == "alpha"
    assert reply.usage["total_tokens"] == 12
    assert route.called


@respx.mock
def test_rotates_to_next_provider_on_429(providers, tmp_path, now):
    respx.post(ALPHA_URL).mock(
        return_value=HttpResponse(429, json={"error": {"message": "quota"}})
    )
    respx.post(BETA_URL).mock(
        return_value=HttpResponse(200, json=completion_json("beta-small"))
    )
    rotor = make_rotor(providers, tmp_path, now)
    reply = rotor.chat("hi")
    assert reply.provider == "beta"
    status = rotor.status()
    alpha_kid = next(k for k in status if k.startswith("alpha:"))
    assert "cooling down" in status[alpha_kid]


@respx.mock
def test_respects_retry_after_header(providers, tmp_path, now):
    respx.post(ALPHA_URL).mock(
        return_value=HttpResponse(
            429,
            json={"error": {"message": "slow down"}},
            headers={"retry-after": "120"},
        )
    )
    respx.post(BETA_URL).mock(
        return_value=HttpResponse(200, json=completion_json("beta-small"))
    )
    rotor = make_rotor(providers, tmp_path, now)
    rotor.chat("hi")
    alpha_kid = next(k for k in rotor.tracker.keys if k.startswith("alpha:"))
    exhausted_until = rotor.tracker.keys[alpha_kid].exhausted_until
    assert exhausted_until == pytest.approx((now + timedelta(seconds=120)).timestamp())


@respx.mock
def test_all_exhausted_raises_with_reset_time(providers, tmp_path, now):
    respx.post(ALPHA_URL).mock(
        return_value=HttpResponse(429, json={"error": {"message": "quota"}})
    )
    respx.post(BETA_URL).mock(
        return_value=HttpResponse(429, json={"error": {"message": "quota"}})
    )
    rotor = make_rotor(providers, tmp_path, now)
    with pytest.raises(AllProvidersExhausted) as exc:
        rotor.chat("hi")
    assert exc.value.earliest_reset is not None
    assert exc.value.earliest_reset > now
    assert len(exc.value.statuses) == 2


@respx.mock
def test_state_persists_across_instances(providers, tmp_path, now):
    respx.post(ALPHA_URL).mock(
        return_value=HttpResponse(429, json={"error": {"message": "quota"}})
    )
    respx.post(BETA_URL).mock(
        return_value=HttpResponse(200, json=completion_json("beta-small"))
    )
    rotor = make_rotor(providers, tmp_path, now)
    rotor.chat("hi")

    # A brand-new rotor (same state file) already knows alpha is exhausted and
    # goes straight to beta without touching alpha.
    respx.post(ALPHA_URL).mock(side_effect=AssertionError("should not be called"))
    rotor2 = make_rotor(providers, tmp_path, now + timedelta(minutes=1))
    reply = rotor2.chat("hi again")
    assert reply.provider == "beta"


@respx.mock
def test_proactive_skip_at_known_daily_cap(providers, tmp_path, now):
    capped = (
        providers[0].__class__(**{**providers[0].__dict__, "known_rpd": 1}),
        providers[1],
    )
    respx.post(ALPHA_URL).mock(
        return_value=HttpResponse(200, json=completion_json("alpha-small"))
    )
    respx.post(BETA_URL).mock(
        return_value=HttpResponse(200, json=completion_json("beta-small"))
    )
    rotor = make_rotor(capped, tmp_path, now)
    assert rotor.chat("one").provider == "alpha"
    # cap of 1 reached: second call must skip alpha without a request
    assert rotor.chat("two").provider == "beta"


@respx.mock
def test_transient_5xx_moves_on_with_short_cooldown(providers, tmp_path, now):
    respx.post(ALPHA_URL).mock(
        return_value=HttpResponse(500, json={"error": {"message": "boom"}})
    )
    respx.post(BETA_URL).mock(
        return_value=HttpResponse(200, json=completion_json("beta-small"))
    )
    rotor = make_rotor(providers, tmp_path, now)
    assert rotor.chat("hi").provider == "beta"
    # after the 60s transient cooldown, alpha is eligible again
    alpha_kid = next(k for k in rotor.tracker.keys if k.startswith("alpha:"))
    assert (
        rotor.tracker.status(alpha_kid, providers[0], now + timedelta(seconds=61))
        is None
    )


def test_explicit_model_and_bad_args(providers, tmp_path, now):
    rotor = make_rotor(providers, tmp_path, now)
    with pytest.raises(ValueError):
        rotor.chat()  # neither prompt nor messages
    with pytest.raises(ValueError):
        rotor.chat("hi", messages=[{"role": "user", "content": "hi"}])  # both
