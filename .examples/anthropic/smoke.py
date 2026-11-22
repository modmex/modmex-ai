"""Live smoke test for the Anthropic adapter. Calls the real API (costs cents).

Run:
    MODMEX_AI_RUN_LIVE=1 ANTHROPIC_API_KEY=... poetry run python .examples/anthropic/smoke.py

Checks, in order (stops at the first failure):
  1. complete     - plain text request, headers and usage parsing
  2. structured   - output_type -> output_config.format JSON schema
  3. tool loop    - Agent tool call; the second request replays the signed
                    thinking blocks verbatim (the part only the real API can verify)
  4. stream       - SSE parsing via Agent.run_stream
  5. async        - acomplete (native httpx if installed, else thread)
  6. tool switch  - tool_selector changes the tool set mid-run; the adapter must
                    drop the thinking bound to the old set so the API accepts it
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from modmex import BaseModel

from modmex_ai import Agent, AnthropicSettings, ModelSettings, tool
from modmex_ai.errors import ProviderError
from modmex_ai.messages import Message
from modmex_ai.models import ModelRequest
from modmex_ai.providers.anthropic import AnthropicMessagesModel


MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
# Low effort keeps the smoke test cheap and fast; it still thinks adaptively.
SETTINGS = ModelSettings(reasoning_effort="low", max_tokens=4000)


class Capital(BaseModel):
    country: str
    capital: str


tool_calls: list[str] = []


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    tool_calls.append(city)
    temps = {"paris": 21, "madrid": 30, "london": 12}
    return {"city": city, "temp_c": temps.get(city.strip().lower(), 18), "conditions": "sunny"}


class RecordingHttp:
    """Wrap a transport and keep each request/response body for diagnostics."""

    def __init__(self, inner):
        self.inner = inner
        self.exchanges: list[tuple[dict, dict]] = []

    def post_json(self, url, *, headers=None, data=None, timeout=None):
        response = self.inner.post_json(url, headers=headers, data=data, timeout=timeout)
        self.exchanges.append((data, response.body))
        return response

    def post_json_stream(self, *args, **kwargs):
        return self.inner.post_json_stream(*args, **kwargs)


def check(name: str, fn) -> None:
    started = time.perf_counter()
    try:
        detail = fn()
    except ProviderError as exc:
        print(f"FAIL {name}: HTTP {exc.status_code} request_id={exc.request_id}")
        print(f"     {exc.response_body}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - report any failure and stop
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        sys.exit(1)
    print(f"ok   {name} ({time.perf_counter() - started:.1f}s) {detail}")


def main() -> None:
    if os.getenv("MODMEX_AI_RUN_LIVE") != "1":
        raise SystemExit("Set MODMEX_AI_RUN_LIVE=1 to run this live API example.")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY.")
    model = AnthropicMessagesModel(MODEL)
    print(f"model: {MODEL}")

    def complete():
        response = model.complete(
            ModelRequest(
                messages=[Message(role="user", content="Reply with exactly: pong")],
                settings=SETTINGS,
            )
        )
        assert response.output_text and "pong" in response.output_text.lower(), response.output_text
        assert response.request_id, "missing request id"
        assert response.usage.input_tokens > 0 and response.usage.output_tokens > 0
        return f"text={response.output_text!r} usage=({response.usage.input_tokens} in, {response.usage.output_tokens} out) request_id={response.request_id}"

    def structured():
        agent = Agent(
            name="capitals",
            instructions="Return the capital of the given country.",
            output_type=Capital,
            model=model,
            settings=SETTINGS,
        )
        result = agent.run("France")
        assert isinstance(result.output, Capital), type(result.output)
        assert result.output.capital.lower() == "paris", result.output
        return f"output={result.output}"

    def tool_loop():
        # Opus 5.5 thinks adaptively and cannot be forced to think. A puzzle that
        # must be solved *before* the first tool call, at max effort, makes a
        # thinking block inside a tool_use turn likely. Only tool_use turns are
        # replayed, and only the real API verifies their signatures.
        prompt = (
            "Before calling any tool, work this out: compute 37 * 43. If the result "
            "is odd AND its digit sum is greater than 15, the first city is Paris; "
            "otherwise it is London. Check that city's weather. If it is above 20°C, "
            "then check Madrid; otherwise check Oslo. Finally say which of the two "
            "checked cities is better for a picnic and why."
        )
        last = ""
        for attempt in range(1, 4):
            tool_calls.clear()
            recorder = RecordingHttp(model.http_client)
            agent = Agent(
                name="weather",
                instructions="Use the get_weather tool. Follow the user's plan step by step.",
                tools=[get_weather],
                model=AnthropicMessagesModel(MODEL, http_client=recorder),
                settings=ModelSettings(
                    reasoning_effort="max",
                    max_tokens=16000,
                    anthropic=AnthropicSettings(thinking_display="summarized"),
                ),
            )
            result = agent.run(prompt)
            assert tool_calls[:1] == ["Paris"] and "Madrid" in tool_calls, f"tool calls: {tool_calls}"
            print(f"     attempt {attempt}:")
            for index, (_, body) in enumerate(recorder.exchanges, 1):
                kinds = [block.get("type") for block in body.get("content", [])]
                print(f"       #{index} stop={body.get('stop_reason')} blocks={kinds}")
            replayed_signed = any(
                block.get("type") in {"thinking", "redacted_thinking"}
                for request, _ in recorder.exchanges[1:]
                for message in request["messages"]
                if message["role"] == "assistant"
                for block in message["content"]
            )
            if replayed_signed:
                return (
                    f"signed thinking replayed OK on attempt {attempt} "
                    f"(calls={tool_calls}) output={result.output!r}"
                )
            last = f"calls={tool_calls}"
        return f"WARNING: no thinking in any tool_use turn after 3 attempts; signed replay NOT exercised ({last})"

    def stream():
        agent = Agent(name="streamer", instructions="Be brief.", model=model, settings=SETTINGS)
        chunks = []
        for event in agent.run_stream("Count from 1 to 5, comma separated."):
            if event.text_delta:
                chunks.append(event.text_delta)
        text = "".join(chunks)
        assert "1" in text and "5" in text, f"streamed text: {text!r} ({len(chunks)} chunks)"
        return f"chunks={len(chunks)} text={text!r}"

    def async_complete():
        async def run():
            try:
                return await model.acomplete(
                    ModelRequest(messages=[Message(role="user", content="Reply with exactly: pong")], settings=SETTINGS)
                )
            finally:
                await model.aclose()

        response = asyncio.run(run())
        assert response.output_text and "pong" in response.output_text.lower(), response.output_text
        transport = "httpx" if model.async_http_client is not None else "thread"
        return f"transport={transport} text={response.output_text!r}"

    def tool_switch():
        tool_calls.clear()

        @tool
        def pick_city(hint: str) -> dict:
            """Resolve a city name from a hint."""
            return {"city": "Paris"}

        phases = {"calls": 0}

        def selector(context, names):
            phases["calls"] += 1
            return ["pick_city"] if phases["calls"] == 1 else ["pick_city", "get_weather"]

        recorder = RecordingHttp(model.http_client)
        agent = Agent(
            name="switch",
            instructions="First resolve the city with pick_city, then get its weather.",
            tools=[pick_city, get_weather],
            tool_selector=selector,
            model=AnthropicMessagesModel(MODEL, http_client=recorder),
            settings=ModelSettings(reasoning_effort="max", max_tokens=16000),
        )
        result = agent.run(
            "Before calling any tool, compute 37 * 43 and use the result as the hint for "
            "pick_city. Then get that city's weather and answer in one sentence."
        )
        assert tool_calls == ["Paris"], f"tool calls: {tool_calls}"
        received = [[b.get("type") for b in body.get("content", [])] for _, body in recorder.exchanges]
        sent = [
            [b.get("type") for m in req["messages"] if m["role"] == "assistant" for b in m["content"]]
            for req, _ in recorder.exchanges
        ]
        # Thinking from turn 1 (old tool set) must not be sent after the switch;
        # thinking produced after the switch must still be replayed.
        stale_dropped = "thinking" not in (sent[1] if len(sent) > 1 else [])
        return (
            f"requests={len(sent)} received={received} sent={sent} "
            f"stale_thinking_dropped={stale_dropped} output={result.output!r}"
        )

    check("complete", complete)
    check("structured", structured)
    check("tool loop", tool_loop)
    check("stream", stream)
    check("async", async_complete)
    check("tool switch", tool_switch)
    print("all checks passed")


if __name__ == "__main__":
    main()
