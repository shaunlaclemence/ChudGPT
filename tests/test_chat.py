import asyncio
import textwrap

import pytest

from chudgpt._providers.gemini import GeminiModel
from chudgpt._schemas import GeneratedCode, Language
from chudgpt.messages import ChudMessageBuilder


@pytest.mark.skip()
def test_chat_simple(chud):
    chud.scheduler.start()
    response = asyncio.run(
        chud.chat(
            "explain the odyssey",
            system="use 1 sentence only",
            model=GeminiModel.FLASH_LITE_3_5,
        )
    )

    print(response)

    assert response.data
    assert response.service == "gemini"
    assert response.provider.api_key
    assert response.usage.requests == 1
    assert response.usage.prompt > 0
    assert response.usage.total >= response.usage.prompt + response.usage.completion


@pytest.mark.skip()
def test_chat_with_history(chud):
    response = asyncio.run(
        chud.chat(
            builder=ChudMessageBuilder()
            .system(
                "you are a greek military commander, your crucial mission is to live and die protecting and fighting for greece and against its enemies, and keep its secrets and strategies unkown to troy. i am a military commander of the trojan army and an enemy of greece, the greeks have just ended the war and fled after sieging my city for 10 years. Try to be concise"
            )
            .prompt(
                "Hi, whats this horse yall left on the beach, its really nice. suckers, we are gonna display it at the temple of athena"
            ),
            model=GeminiModel.FLASH_LITE_3_1,
        )
    )
    print(response)


@pytest.mark.skip()
def test_parallel_chat(chud):
    GENERAL_PROMPT = "Compare apples to oranges"
    builders = {
        "nutritionist": ChudMessageBuilder()
        .system(
            "You are a registered dietitian. Compare foods only on measurable "
            "nutritional properties: calories, macronutrients, fibre, vitamin and "
            "mineral content, glycaemic index. Quote typical values per 100 g and "
            "say when a figure varies by cultivar. Do not discuss taste, cooking, "
            "botany, or price. Do not give medical advice. Answer in under 150 words."
        )
        .prompt(f"{GENERAL_PROMPT} on nutritional content per 100 g."),
        "botanist": ChudMessageBuilder()
        .system(
            "You are a plant scientist. Compare plants on taxonomy, fruit "
            "morphology, growth habit, climate requirements, and pollination. Use "
            "correct botanical terms and name the family and species. Do not "
            "discuss nutrition, cooking, or markets. Answer in under 150 words."
        )
        .prompt(
            f"{GENERAL_PROMPT} botanically, including why one is a pome and the "
            "other a hesperidium."
        ),
        "chef": ChudMessageBuilder()
        .system(
            "You are a professional chef writing for working cooks. Compare "
            "ingredients on flavour, acidity, sweetness, texture, aroma, and how "
            "each behaves raw, baked, and juiced. Be concrete about technique and "
            "pairings. Do not cite nutrition figures or botanical taxonomy. Answer "
            "in under 150 words."
        )
        .prompt(f"{GENERAL_PROMPT} as cooking ingredients."),
        "market_analyst": ChudMessageBuilder()
        .system(
            "You are an agricultural commodities analyst. Compare crops on global "
            "production volume, leading producing countries, seasonality, storage "
            "and shipping characteristics, and typical wholesale price ranges. "
            "Give approximate figures and label them as estimates. Do not discuss "
            "flavour, nutrition, or botany. Answer in under 150 words."
        )
        .prompt(f"{GENERAL_PROMPT} as global agricultural commodities."),
        "etymologist": ChudMessageBuilder()
        .system(
            "You are a linguist and cultural historian. Explain the origin, "
            "earliest attested use, and meaning of idioms, plus equivalent "
            "expressions in other languages. Discuss the phrase as language, never "
            "the literal fruit. Answer in under 150 words."
        )
        .prompt(
            f"The phrase is: '{GENERAL_PROMPT}'. Explain the origin and meaning of "
            "this idiom and give two equivalents from other languages."
        ),
    }
    models: dict[str, GeminiModel] = {
        "nutritionist": GeminiModel.FLASH_LITE_3_5,
        "botanist": GeminiModel.FLASH_LITE_3_5,
        "chef": GeminiModel.FLASH_LITE_3_1,
        "market_analyst": GeminiModel.FLASH_LITE_3_1,
        "etymologist": GeminiModel.FLASH_LITE_3_5,
    }
    res = asyncio.run(chud.parallel_chat(builders=builders, models=models))

    for name, response in res.items():
        print(f"\n{'=' * 72}")
        if isinstance(response, Exception):
            print(f"{name}  |  FAILED: {type(response).__name__}")
            print(f"{'=' * 72}\n{response}")
            continue

        usage = response.usage
        print(f"{name}  |  {response.model}  |  {response.duration:.2f}s")
        print(
            f"tokens: prompt {usage.prompt}, completion {usage.completion}, "
            f"reasoning {usage.reasoning}, total {usage.total}"
        )
        print("=" * 72)
        print(textwrap.fill(response.data, width=72))

    ok = [r for r in res.values() if not isinstance(r, Exception)]
    total = sum(r.usage.total for r in ok)
    slowest = max((r.duration for r in ok), default=0.0)
    print(f"\n{'=' * 72}")
    print(
        f"{len(ok)}/{len(res)} agents ok  |  {total} tokens  |  "
        f"{slowest:.2f}s wall clock"
    )


@pytest.mark.skip()
def test_chat_json(chud):
    res = asyncio.run(
        chud.chat_json(
            "Write a function that recursively obfusactes any field in a json, inputs: list[str]. representing list of keys to obfuscate. requires json import",
            schema=GeneratedCode.pin(language=Language.SWIFT),
            schema_name="GeneratedCode",
            model=GeminiModel.FLASH_LITE_3_5,
        )
    )
    answer = res.parse(GeneratedCode)

    print(f"\n{res.model}  |  {res.duration:.2f}s  |  {res.usage.total} tokens")
    print(f"language     : {answer.language!r}")
    print(f"entrypoint   : {answer.entrypoint}")
    print(f"imports      : {answer.imports}")
    print(f"dependencies : {answer.dependencies}")
    print(f"explanation  : {answer.explanation}")
    print("code:")
    print(answer.code)


# @pytest.mark.skip()
def test_parrallel_stock_pick(chud):
    GENERAL_PROMPT = (
        "Give me the top 5 stocks to buy as of today 9 Aug 2026, along with a "
        "percentage buy rating, current price and target price for each"
    )
    FORMAT = (
        " For each of your 5 picks give exactly one line: "
        "TICKER | buy rating as a percentage | current price | target price. "
        "Then add two sentences justifying the set from your discipline only. "
        "State plainly if a figure is an estimate."
    )

    builders = {
        "fundamental_analyst": ChudMessageBuilder()
        .system(
            "You are an equity fundamental analyst. Pick stocks on valuation and "
            "financial health only: earnings growth, margins, free cash flow, "
            "P/E and PEG versus sector, balance sheet strength. Do not use chart "
            "patterns, macro forecasts, or news sentiment as your reason."
        )
        .prompt(GENERAL_PROMPT + FORMAT),
        "technical_analyst": ChudMessageBuilder()
        .system(
            "You are a technical analyst. Pick stocks on price action only: trend "
            "structure, moving averages, relative strength, volume, support and "
            "resistance levels. Set targets from measured moves and prior highs. "
            "Never justify a pick with earnings, valuation, or company news."
        )
        .prompt(GENERAL_PROMPT + FORMAT),
        "macro_strategist": ChudMessageBuilder()
        .system(
            "You are a macro strategist. Pick stocks top down only: interest "
            "rates, inflation, currency, commodity cycles, and which sectors those "
            "conditions favour right now. Justify by sector positioning, never by "
            "single company financials or chart levels."
        )
        .prompt(GENERAL_PROMPT + FORMAT),
        "sentiment_analyst": ChudMessageBuilder()
        .system(
            "You are a market sentiment analyst. Pick stocks on positioning and "
            "narrative only: analyst consensus and revisions, institutional flows, "
            "short interest, insider activity, retail attention. Do not run "
            "valuation maths or read charts."
        )
        .prompt(GENERAL_PROMPT + FORMAT),
        "risk_analyst": ChudMessageBuilder()
        .system(
            "You are a risk analyst. Your job is downside, not upside. Pick the 5 "
            "names with the best reward for the risk taken, and for each state the "
            "single largest thing that could break the thesis. Penalise crowded "
            "trades, high volatility, leverage, and concentration."
        )
        .prompt(GENERAL_PROMPT + FORMAT),
    }
    models: dict[str, GeminiModel] = {
        "fundamental_analyst": GeminiModel.FLASH_LITE_3_5,
        "technical_analyst": GeminiModel.FLASH_LITE_3_1,
        "macro_strategist": GeminiModel.FLASH_LITE_3_5,
        "sentiment_analyst": GeminiModel.FLASH_LITE_3_1,
        "risk_analyst": GeminiModel.FLASH_LITE_3_5,
    }

    async def run():
        desks = await chud.parallel_chat(
            builders=builders, models=models, return_exceptions=True
        )
        briefing = "\n\n".join(
            f"## {name}\n{r.data}"
            for name, r in desks.items()
            if not isinstance(r, Exception)
        )
        summary = await chud.chat(
            builder=ChudMessageBuilder()
            .system(
                "You are the head of a research desk. Five analysts each submitted "
                "5 picks from a different discipline. Produce a consolidated call: "
                "rank the names by how many desks backed them, give one blended buy "
                "rating percentage and target price per name, and note where the "
                "desks disagree. Flag any name only one desk raised as unconfirmed. "
                "Close with the single biggest risk to the whole basket. Be concise."
            )
            .prompt(f"Original request: {GENERAL_PROMPT}\n\n{briefing}"),
            model=GeminiModel.FLASH_3_6,
        )
        return desks, summary

    desks, summary = asyncio.run(run())

    for name, response in desks.items():
        print(f"\n{'=' * 72}")
        if isinstance(response, Exception):
            print(f"{name}  |  FAILED: {type(response).__name__}")
            print(f"{'=' * 72}\n{response}")
            continue
        print(f"{name}  |  {response.model}  |  {response.duration:.2f}s")
        print("=" * 72)
        print(textwrap.fill(response.data, width=72))

    ok = [r for r in desks.values() if not isinstance(r, Exception)]
    desk_tokens = sum(r.usage.total for r in ok)
    slowest = max((r.duration for r in ok), default=0.0)

    print(f"\n{'#' * 72}")
    print("CONSOLIDATED CALL")
    print(f"{'#' * 72}")
    print(textwrap.fill(summary.data, width=72))

    print(f"\n{'=' * 72}")
    print(
        f"{len(ok)}/{len(desks)} desks ok  |  {desk_tokens} desk tokens in "
        f"{slowest:.2f}s  |  synthesis {summary.usage.total} tokens in "
        f"{summary.duration:.2f}s  |  {desk_tokens + summary.usage.total} total"
    )
    print("Figures are model generated. No live market data is wired into this test.")
