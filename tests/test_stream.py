import asyncio

import pytest

from chudgpt import ChudChannel, ChudGPT, ChudResponse, ChudStreamEvent, GeminiModel
from chudgpt.messages import ChudMessageBuilder

PROMPT = "Think through your answer first. Why did odysseus blind the cyclops. And how does this parallel in the Spongebob Squarepants Movie?"


@pytest.mark.skip()
def test_stream_separates_thinking_from_the_answer(chud: ChudGPT, save_output):
    async def run() -> tuple[str, str, ChudResponse | None]:
        thinking: list[str] = []
        answer: list[str] = []
        done: ChudResponse | None = None
        channel: ChudChannel | None = None

        event: ChudStreamEvent
        async for event in chud.text.stream(
            PROMPT, model=GeminiModel.FLASH_LITE_3_5, think=True
        ):
            if event.channel is ChudChannel.DONE:
                done = event.response
                continue
            if channel != event.channel:
                print(f"\n--- {event.channel.value} ---")
                channel = event.channel
            print(event.text, end="", flush=True)
            bucket = thinking if event.channel is ChudChannel.THINKING else answer
            bucket.append(event.text)

        return "".join(thinking), "".join(answer), done

    thinking, answer, done = asyncio.run(run())

    print(f"\n\nthinking {len(thinking)} chars, answer {len(answer)} chars")
    print(f"usage {done.usage}, {done.duration:.2f}s on {done.model}")
    save_output({"thinking": thinking, "answer": answer, "response": done})

    assert answer, "no answer channel came through"
    assert "<thought>" not in thinking + answer, "tags leaked into the text"
    assert "</thought>" not in thinking + answer, "tags leaked into the text"
    assert done is not None, "the stream never closed with a DONE event"
    assert done.data == answer, "DONE must carry the assembled answer"
    assert done.usage.total > 0, "usage only arrives on the final chunk"
    if done.usage.reasoning:
        assert thinking, "reasoning tokens were billed but no thinking reached us"


DESKS = {
    "historian": "In three paragraphs, why did Odysseus blind the cyclops?",
    "critic": "In three paragraphs, what makes the Spongebob Squarepants Movie funny?",
}


@pytest.mark.skip()
def test_parallel_stream_interleaves_two_desks(chud: ChudGPT, save_output):
    builders = {
        name: ChudMessageBuilder().prompt(prompt) for name, prompt in DESKS.items()
    }
    models = dict.fromkeys(builders, GeminiModel.FLASH_LITE_3_5)

    async def run() -> tuple[dict[str, str], dict[str, ChudResponse], list[str]]:
        text: dict[str, str] = {name: "" for name in builders}
        finished: dict[str, ChudResponse] = {}
        order: list[str] = []
        speaking: str | None = None

        name: str
        event: ChudStreamEvent
        async for name, event in chud.text.parallel_stream(builders, models):
            if event.channel is ChudChannel.ERROR:
                print(f"\n\n[{name}] FAILED {event.text}", flush=True)
                speaking = None
                continue
            if event.channel is ChudChannel.DONE:
                finished[name] = event.response
                print(
                    f"\n\n[{name}] done, {event.response.usage.total} tokens,"
                    f" {event.response.duration:.2f}s",
                    flush=True,
                )
                speaking = None
                continue
            if name != speaking:
                print(f"\n\n[{name}] ", end="", flush=True)
                speaking = name
            print(event.text, end="", flush=True)
            order.append(name)
            text[name] += event.text

        return text, finished, order

    text, finished, order = asyncio.run(run())
    save_output({"text": text, "responses": finished})

    assert set(finished) == set(builders), "every desk must close with a DONE event"
    for name, response in finished.items():
        assert response.data == text[name], "DONE carries that desk's assembled answer"
        assert response.usage.total > 0

    assert set(order) == set(builders), "both desks must produce text"
    assert len(set(order[:20])) == 2, "events should interleave, not run one then two"
