import pytest

from chudgpt import UsagePeriod

periods = UsagePeriod.values()

@pytest.mark.parametrize("period", periods)
def test_get_requests(chud, period):
    req = chud.get_requests(per=period)
    print("\n")
    print("REQUESTS ", period)
    print({ p.masked_key: v for p, v in req.items()})


@pytest.mark.parametrize("period", periods)
def test_get_tokens(chud, period):
    req = chud.get_tokens(per=period)
    print("\n")
    print("TOKENS ", period)
    print({ p.masked_key: v for p, v in req.items()})
