from typing import Literal

from modmex import BaseModel

from modmex_ai.messages import Message
from modmex_ai.models import ModelRequest, ModelSettings
from modmex_ai.providers.openai.mapper import (
    _apply_provider_state,
    _input_items_for_provider_state,
    _normalize_schema,
    _provider_state_from_response,
    chat_stream_events,
    from_chat_payload,
    from_responses_payload,
    to_openai_strict_schema,
    tool_results_to_responses_input,
    to_chat_payload,
    to_responses_payload,
    responses_stream_events,
)
from modmex_ai.models import ModelStreamEventType, ProviderState


def test_chat_stream_mapper_emits_text_tool_deltas_and_a_completed_response():
    events = list(chat_stream_events([
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "lookup", "arguments": "{\"x\":"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}}]},
    ], headers={}, status_code=200, model="gpt-test"))

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.TOOL_CALL_DELTA,
        ModelStreamEventType.TOOL_CALL_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[-1].response.output_text == "Hello"
    assert events[-1].response.tool_calls[0].arguments == '{"x":1}'


def test_responses_stream_mapper_emits_typed_deltas_and_uses_completed_response():
    events = list(responses_stream_events([
        {"type": "response.output_text.delta", "delta": "Hello"},
        {"type": "response.function_call_arguments.delta", "call_id": "call-1", "name": "lookup", "delta": "{}"},
        {"type": "response.completed", "response": {"id": "resp-1", "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}]}},
    ], headers={}, status_code=200, model="gpt-test"))

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.TOOL_CALL_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[-1].response.output_text == "Hello"
    assert events[-1].response.provider_state.previous_response_id == "resp-1"
from modmex_ai.schemas import schema_for_model
from modmex_ai.sessions import SessionItem


class Output(BaseModel):
    value: str


class NullableOutput(BaseModel):
    intent: Literal["quote", "support"]
    follow_up_note: str | None = None


def test_openai_responses_mapper_includes_tools_and_schema():
    request = ModelRequest(
        messages=[
            Message(role="developer", content="Be precise."),
            Message(role="user", content="hello"),
        ],
        tools=[
            {
                "name": "lookup",
                "description": "",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        ],
        output_schema=schema_for_model(Output),
    )

    payload = to_responses_payload(request, "gpt-test")

    assert payload["model"] == "gpt-test"
    assert payload["instructions"] == "Be precise."
    assert payload["tools"][0]["type"] == "function"
    assert payload["text"]["format"]["type"] == "json_schema"


def test_openai_mapper_normalizes_strict_tool_schemas():
    request = ModelRequest(
        messages=[Message(role="user", content="hello")],
        tools=[
            {
                "name": "record",
                "description": "",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note": {"type": "string", "nullable": True},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        ],
    )

    responses_payload = to_responses_payload(request, "gpt-test")
    chat_payload = to_chat_payload(request, "gpt-test")

    assert responses_payload["tools"][0]["parameters"] == {
        "type": "object",
        "properties": {"note": {"type": ["string", "null"]}},
        "required": ["note"],
        "additionalProperties": False,
    }
    assert chat_payload["tools"][0]["function"]["parameters"] == (
        responses_payload["tools"][0]["parameters"]
    )


def test_openai_response_parsers_extract_tool_calls_and_text():
    response = from_responses_payload(
        {
            "id": "resp-1",
            "usage": {
                "input_tokens": 9,
                "output_tokens": 3,
                "total_tokens": 12,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens_details": {"reasoning_tokens": 2},
            },
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": '{"id":"1"}',
                }
            ],
        },
        headers={"x-request-id": "req-1"},
        status_code=200,
        model="gpt-test",
    )

    assert response.request_id == "req-1"
    assert response.tool_calls[0].name == "lookup"
    assert response.usage.input_tokens == 9
    assert response.usage.output_tokens == 3
    assert response.usage.cached_input_tokens == 4
    assert response.usage.reasoning_output_tokens == 2


def test_chat_mapper_roundtrip_shape():
    request = ModelRequest(
        messages=[Message(role="user", content="hello")],
        output_schema=schema_for_model(Output),
    )

    payload = to_chat_payload(request, "gpt-test")

    assert payload["response_format"]["type"] == "json_schema"

    response = from_chat_payload(
        {
            "id": "chat-1",
            "usage": {
                "prompt_tokens": 6,
                "completion_tokens": 2,
                "total_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 1},
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
            "choices": [{"message": {"content": '{"value":"ok"}'}}],
        },
        headers={},
        status_code=200,
        model="gpt-test",
    )

    assert response.output_text == '{"value":"ok"}'
    assert response.usage.total_tokens == 8
    assert response.usage.cached_input_tokens == 1
    assert response.usage.reasoning_output_tokens == 3


def test_openai_mapper_handles_tool_messages_settings_and_tool_calls():
    request = ModelRequest(
        messages=[
            Message(role="developer", content="Be precise."),
            Message(role="tool", content='{"ok":true}', tool_call_id="call-1", name="lookup"),
        ],
        tools=[
            {
                "name": "lookup",
                "description": "",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        ],
        settings=ModelSettings(
            temperature=0.1,
            top_p=0.9,
            max_tokens=12,
            extra={"parallel_tool_calls": False},
        ),
    )

    responses_payload = to_responses_payload(request, "gpt-test")
    chat_payload = to_chat_payload(request, "gpt-test")

    assert responses_payload["input"][0]["type"] == "function_call_output"
    assert responses_payload["temperature"] == 0.1
    assert responses_payload["top_p"] == 0.9
    assert responses_payload["max_output_tokens"] == 12
    assert responses_payload["parallel_tool_calls"] is False
    assert chat_payload["tools"][0]["function"]["name"] == "lookup"
    assert chat_payload["messages"][1]["tool_call_id"] == "call-1"


def test_openai_mapper_includes_function_call_before_tool_output():
    request = ModelRequest(
        messages=[
            Message(role="user", content="hello"),
            SessionItem(
                type="function_call",
                tool_call_id="call-1",
                name="lookup",
                arguments={"id": "1"},
            ),
            SessionItem(
                type="function_call_output",
                tool_call_id="call-1",
                name="lookup",
                output={"ok": True},
            ),
        ],
    )

    responses_payload = to_responses_payload(request, "gpt-test")
    chat_payload = to_chat_payload(request, "gpt-test")

    assert responses_payload["input"][1] == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "lookup",
        "arguments": '{"id":"1"}',
    }
    assert responses_payload["input"][2]["type"] == "function_call_output"
    assert responses_payload["input"][2]["output"] == '{"ok":true}'
    assert chat_payload["messages"][1]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": '{"id":"1"}',
    }
    assert chat_payload["messages"][2]["role"] == "tool"


def test_openai_mapper_handles_session_message_and_handoff_items():
    request = ModelRequest(
        messages=[
            SessionItem(role="user", content="hello"),
            SessionItem(
                type="handoff_call",
                tool_call_id="call-1",
                name="transfer_to_support",
                arguments='{"reason":"help"}',
            ),
        ],
    )

    responses_payload = to_responses_payload(request, "gpt-test")
    chat_payload = to_chat_payload(request, "gpt-test")

    assert responses_payload["input"] == [
        {"role": "user", "content": "hello"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "transfer_to_support",
            "arguments": '{"reason":"help"}',
        },
    ]
    assert chat_payload["messages"][0] == {"role": "user", "content": "hello"}
    assert chat_payload["messages"][1]["role"] == "assistant"
    assert chat_payload["messages"][1]["tool_calls"][0]["function"]["name"] == "transfer_to_support"


def test_openai_parsers_handle_text_and_chat_tool_calls():
    responses = from_responses_payload(
        {
            "id": "resp-1",
            "output": [
                {"type": "message", "content": [{"type": "text", "text": "hello"}]},
            ],
        },
        headers={},
        status_code=200,
        model="gpt-test",
    )
    chat = from_chat_payload(
        {
            "id": "chat-1",
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "lookup", "arguments": '{"id":"1"}'},
                            }
                        ]
                    }
                }
            ],
        },
        headers={"x-request-id": "req-1"},
        status_code=200,
        model="gpt-test",
    )

    assert responses.output_text == "hello"
    assert chat.request_id == "req-1"
    assert chat.tool_calls[0].arguments == '{"id":"1"}'


def test_tool_results_to_responses_input_serializes_outputs():
    result = tool_results_to_responses_input([
        {"tool_call_id": "call-1", "output": {"ok": True}},
    ])

    assert result == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"ok":true}',
        }
    ]


def test_openai_strict_schema_requires_all_properties_and_converts_nullable():
    schema = schema_for_model(NullableOutput)
    strict = to_openai_strict_schema(schema)

    assert strict["required"] == ["intent", "follow_up_note"]
    assert strict["additionalProperties"] is False
    assert strict["properties"]["follow_up_note"]["type"] == ["string", "null"]


def test_openai_payload_uses_strict_schema_lowering():
    request = ModelRequest(
        messages=[Message(role="user", content="hello")],
        output_schema=schema_for_model(NullableOutput),
    )

    payload = to_responses_payload(request, "gpt-test")
    schema = payload["text"]["format"]["schema"]

    assert schema["required"] == ["intent", "follow_up_note"]
    assert schema["properties"]["follow_up_note"]["type"] == ["string", "null"]


def test_openai_payload_preserves_optional_fields_when_output_is_not_strict():
    request = ModelRequest(
        messages=[Message(role="user", content="hello")],
        output_schema=schema_for_model(NullableOutput),
        output_strict=False,
    )

    payload = to_responses_payload(request, "gpt-test")
    schema = payload["text"]["format"]["schema"]

    assert payload["text"]["format"]["strict"] is False
    assert schema["required"] == ["intent"]
    assert schema["properties"]["follow_up_note"] == {
        "type": "string",
        "nullable": True,
    }


def test_openai_mapper_normalizes_nested_schema_provider_state_and_incomplete_stream():
    schema = _normalize_schema({
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "string", "nullable": True}}},
        "oneOf": [{"type": "string"}],
    })
    assert schema["required"] == ["items"]
    assert schema["properties"]["items"]["items"]["type"] == ["string", "null"]
    payload = {}
    _apply_provider_state(payload, ProviderState(provider="openai", previous_response_id="r", conversation_id="c", values={"x": 1}))
    assert payload == {"previous_response_id": "r", "conversation": "c", "x": 1}
    assert _input_items_for_provider_state([{"role": "user"}, {"type": "function_call_output"}], ProviderState(previous_response_id="r")) == [{"type": "function_call_output"}]
    assert _provider_state_from_response({"conversation": {"id": "c"}}).conversation_id == "c"
    events = list(responses_stream_events([
        {"type": "response.function_call_arguments.done", "call_id": "tool", "name": "lookup", "arguments": "{}"},
    ], headers={}, status_code=200, model="gpt"))
    assert events[-1].response.tool_calls[0].name == "lookup"
