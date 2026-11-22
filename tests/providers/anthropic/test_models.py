import asyncio
import json

from modmex_ai import Agent, tool
from modmex_ai.http import HttpResponse
from modmex_ai.messages import Message
from modmex_ai.models import AnthropicSettings, ModelRequest, ModelSettings, ModelStreamEventType
from modmex_ai.providers.anthropic import AnthropicMessagesModel
from modmex_ai.providers.anthropic.messages import RETRY_STATUS_CODES


class FakeHttpClient:
    def __init__(self, *bodies, stream_events=None):
        self.bodies = list(bodies)
        self.stream_events = stream_events or []
        self.calls = []

    def post_json(self, url, *, headers=None, data=None, timeout=None):
        # Snapshot: the runner keeps mutating its history after the call.
        self.calls.append({"url": url, "headers": headers, "data": json.loads(json.dumps(data)), "timeout": timeout})
        return HttpResponse(status_code=200, headers={"request-id": "req_1"}, body=self.bodies.pop(0))

    def post_json_stream(self, url, *, headers=None, data=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "data": data, "timeout": timeout})
        return iter(
            [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in self.stream_events]
        )


class FakeAsyncHttpClient:
    def __init__(self, body):
        self.body = body
        self.calls = []

    async def post_json(self, url, *, headers=None, data=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "data": data})
        return HttpResponse(status_code=200, headers={}, body=self.body)

    async def close(self):
        self.closed = True


def _text(text):
    return {"id": "msg_t", "model": "claude-opus-5-5", "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 2}}


def test_complete_posts_to_messages_api_with_headers(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    http = FakeHttpClient(_text("ok"))
    model = AnthropicMessagesModel("claude-opus-5-5", api_key="sk-test", http_client=http)
    response = model.complete(
        ModelRequest(
            messages=[Message(role="user", content="hi")],
            settings=ModelSettings(timeout=12, anthropic=AnthropicSettings(fallbacks="default")),
        )
    )
    call = http.calls[0]
    assert response.output_text == "ok"
    assert response.request_id == "req_1"
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"] == {
        "anthropic-version": "2026-10-21",
        "x-api-key": "sk-test",
        "anthropic-beta": "server-side-fallback-2026-07-01",
    }
    assert call["timeout"] == 12


def test_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    model = AnthropicMessagesModel("claude-opus-5-5", http_client=FakeHttpClient())
    assert model._headers(ModelRequest(messages=[]))["x-api-key"] == "sk-env"


def test_default_transport_retries_overloaded():
    model = AnthropicMessagesModel("claude-opus-5-5", api_key="k")
    assert 529 in model.http_client.retry.status_codes
    assert RETRY_STATUS_CODES >= {429, 500, 529}


def test_acomplete_uses_async_client():
    http = FakeAsyncHttpClient(_text("async ok"))
    model = AnthropicMessagesModel("claude-opus-5-5", api_key="k", async_http_client=http)
    response = asyncio.run(model.acomplete(ModelRequest(messages=[Message(role="user", content="hi")])))
    assert response.output_text == "async ok"
    assert http.calls[0]["url"].endswith("/messages")


def test_stream_parses_sse():
    events = [
        {"type": "message_start", "message": {"id": "msg_s", "content": [], "usage": {"input_tokens": 4}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hola"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    http = FakeHttpClient(stream_events=events)
    model = AnthropicMessagesModel("claude-opus-5-5", api_key="k", http_client=http)
    out = list(model.stream(ModelRequest(messages=[Message(role="user", content="hi")])))
    assert [e.text_delta for e in out if e.type == ModelStreamEventType.TEXT_DELTA] == ["Hola"]
    assert out[-1].response.output_text == "Hola"
    assert http.calls[0]["data"]["stream"] is True


def test_agent_tool_loop_replays_thinking_blocks():
    calls = []

    @tool
    def get_weather(city: str) -> dict:
        """Get the weather for a city."""
        calls.append(city)
        return {"city": city, "temp_c": 21}

    assistant_turn = [
        {"type": "thinking", "thinking": "", "signature": "sig-1"},
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
    ]
    http = FakeHttpClient(
        {"id": "msg_1", "content": assistant_turn, "stop_reason": "tool_use", "usage": {"input_tokens": 50, "output_tokens": 20}},
        _text("21°C en París"),
    )
    agent = Agent(
        name="weather",
        instructions="Answer with the weather.",
        tools=[get_weather],
        model=AnthropicMessagesModel("claude-opus-5-5", api_key="k", http_client=http),
    )

    result = agent.run("¿Clima en París?")

    assert result.output == "21°C en París"
    assert calls == ["Paris"]
    first, second = (call["data"] for call in http.calls)
    assert first["system"] == second["system"]
    assert first["tools"] == second["tools"]
    assert second["messages"][: len(first["messages"])] == first["messages"]  # append-only
    assert second["messages"][-2] == {"role": "assistant", "content": assistant_turn}
    assert second["messages"][-1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"city":"Paris","temp_c":21}'}],
    }
    assert result.usage.input_tokens == 53


def _tool_turn(call_id, name, signature):
    return {
        "id": f"msg_{call_id}",
        "content": [
            {"type": "thinking", "thinking": "", "signature": signature},
            {"type": "tool_use", "id": call_id, "name": name, "input": {}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def test_tool_selector_change_drops_now_invalid_thinking():
    @tool
    def first_step() -> str:
        """Step one."""
        return "one"

    @tool
    def second_step() -> str:
        """Step two."""
        return "two"

    selections = iter([["first_step"], ["first_step", "second_step"], ["first_step", "second_step"]])
    http = FakeHttpClient(
        _tool_turn("toolu_1", "first_step", "sig-A"),
        _tool_turn("toolu_2", "second_step", "sig-B"),
        _text("done"),
    )
    agent = Agent(
        name="phases",
        instructions="Do the steps.",
        tools=[first_step, second_step],
        tool_selector=lambda ctx, names: next(selections),
        model=AnthropicMessagesModel("claude-opus-5-5", api_key="k", http_client=http),
    )

    assert agent.run("go").output == "done"
    second, third = http.calls[1]["data"], http.calls[2]["data"]

    def signatures(data):
        return [b["signature"] for m in data["messages"] if m["role"] == "assistant" for b in m["content"] if b["type"] == "thinking"]

    # Tools changed between call 1 and 2: sig-A was bound to the old tool set.
    assert signatures(second) == []
    # Tools stable between call 2 and 3: sig-B is still valid and kept; sig-A stays dropped.
    assert signatures(third) == ["sig-B"]
    # tool_use blocks are always kept, so the history stays consistent.
    uses = [b["id"] for m in third["messages"] if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
    assert uses == ["toolu_1", "toolu_2"]
