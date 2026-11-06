import asyncio
import json

import pytest

from modmex_ai import Agent, ModelRequest, ModelSettings, OpenAISettings, ProviderState
from modmex_ai.http import HttpResponse
from modmex_ai.providers.openai import OpenAIResponsesModel
from modmex_ai.providers.openai.mapper import (
    to_responses_payload,
    to_chat_payload,
    from_responses_payload,
)


def options(threshold=12000):
    return ModelSettings(openai=OpenAISettings(compact_threshold=threshold))


def test_native_compaction_wire_shape_disabled_default_and_chat_rejection():
    request = ModelRequest(
        messages=[],
        settings=options(),
        provider_state=ProviderState(mode="response_chain"),
    )
    assert to_responses_payload(request, "test")["context_management"] == [
        {"type": "compaction", "compact_threshold": 12000}
    ]
    assert "context_management" not in to_responses_payload(
        ModelRequest(messages=[]), "test"
    )
    with pytest.raises(ValueError, match="Responses API"):
        to_chat_payload(request, "test")


@pytest.mark.parametrize("threshold", [0, -1, True])
def test_invalid_threshold_fails_before_network(threshold):
    with pytest.raises(ValueError, match="positive integer"):
        to_responses_payload(
            ModelRequest(
                messages=[],
                settings=options(threshold),
                provider_state=ProviderState(mode="response_chain"),
            ),
            "test",
        )


@pytest.mark.parametrize(
    "state", [None, ProviderState(mode="stateless"), ProviderState(conversation_id="c")]
)
def test_unsupported_state_cannot_silently_lose_opaque_compaction_item(state):
    with pytest.raises(ValueError, match="response_chain"):
        to_responses_payload(
            ModelRequest(messages=[], settings=options(), provider_state=state), "test"
        )


class Transport:
    def __init__(self):
        self.payloads = []

    def post_json(self, url, *, data, **kwargs):
        self.payloads.append(json.loads(json.dumps(data)))
        n = len(self.payloads)
        output = (
            [
                {
                    "type": "compaction",
                    "id": "cmp1",
                    "encrypted_content": "opaque-private-state",
                },
                {
                    "type": "function_call",
                    "call_id": "call1",
                    "name": "lookup",
                    "arguments": "{}",
                },
            ]
            if n == 1
            else [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ]
        )
        return HttpResponse(
            status_code=200,
            headers={},
            body={
                "id": f"r{n}",
                "output": output,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "total_tokens": 110,
                },
            },
        )

    def post_json_stream(self, url, *, data, **kwargs):
        response = self.post_json(url, data=data)
        for item in response.body["output"]:
            yield (
                "data: "
                + json.dumps({"type": "response.output_item.done", "item": item})
                + "\n\n"
            ).encode()
        yield (
            "data: "
            + json.dumps({"type": "response.completed", "response": response.body})
            + "\n\n"
        ).encode()


@pytest.mark.parametrize("mode", ["sync", "async", "stream", "async_stream"])
def test_compacted_chain_continues_tool_and_new_turn_without_resending_opaque_items(
    mode,
):
    transport = Transport()

    def lookup():
        return {"found": True}

    agent = Agent(
        name="carrier",
        instructions="Help",
        tools=[lookup],
        settings=options(),
        model=OpenAIResponsesModel("test", http_client=transport),
    )

    def execute(text, state):
        if mode == "sync":
            return agent.run(text, provider_state=state)
        if mode == "stream":
            return list(agent.run_stream(text, provider_state=state))[-1].result

        async def run():
            if mode == "async":
                return await agent.run_async(text, provider_state=state)
            events = [
                event async for event in agent.arun_stream(text, provider_state=state)
            ]
            return events[-1].result

        return asyncio.run(run())

    result = execute("start", ProviderState(mode="response_chain"))
    assert result.output == "done"
    assert result.usage.total_tokens == 220
    events = [step for step in result.trace.steps if step.type == "provider_compaction"]
    assert len(events) == 1
    assert events[0].data["count"] == 1
    assert "opaque-private-state" not in str(events[0].data)
    assert transport.payloads[1]["previous_response_id"] == "r1"
    assert [item["call_id"] for item in transport.payloads[1]["input"]] == ["call1"]
    assert all(
        body["context_management"][0]["compact_threshold"] == 12000
        for body in transport.payloads
    )
    restored = ProviderState(**result.provider_state.model_dump())
    execute("next", restored)
    assert transport.payloads[-1]["previous_response_id"] == "r2"
    assert len(transport.payloads[-1]["input"]) == 1
    assert transport.payloads[-1]["input"][0]["role"] == "user"


def test_parser_exposes_compaction_occurrence_without_interpreting_encrypted_state():
    parsed = from_responses_payload(
        {
            "id": "r",
            "output": [
                {"type": "compaction", "encrypted_content": "opaque"},
                {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
            ],
        },
        headers={},
        status_code=200,
        model="test",
    )
    assert parsed.compaction_count == 1
    assert parsed.output_text == "ok"
    assert parsed.provider_state.previous_response_id == "r"
