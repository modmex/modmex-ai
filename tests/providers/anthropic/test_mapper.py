import pytest

from modmex_ai.errors import ModelRefusalError, ProviderError
from modmex_ai.messages import FileInput, ImageInput, Message, TextInput
from modmex_ai.models import (
    AnthropicSettings,
    ModelRequest,
    ModelSettings,
    ModelStreamEventType,
    OpenAISettings,
    ProviderState,
)
from modmex_ai.providers.anthropic.mapper import (
    beta_headers,
    from_messages_payload,
    messages_stream_events,
    to_anthropic_schema,
    to_messages_payload,
)
from modmex_ai.sessions import SessionItem


THINKING = {"type": "thinking", "thinking": "", "signature": "sig-abc"}


def _payload(**kwargs):
    return to_messages_payload(ModelRequest(**kwargs), "claude-opus-5-5")


def test_basic_payload_splits_system_and_sets_max_tokens():
    payload = _payload(
        messages=[
            Message(role="system", content="Be terse."),
            Message(role="developer", content="Answer in Spanish."),
            Message(role="user", content="hola"),
        ]
    )
    assert payload == {
        "model": "claude-opus-5-5",
        "max_tokens": 16000,
        "system": "Be terse.\n\nAnswer in Spanish.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hola"}]}],
    }


def test_stream_payload_uses_larger_default_max_tokens():
    payload = to_messages_payload(
        ModelRequest(messages=[Message(role="user", content="hi")]),
        "claude-opus-5-5",
        stream=True,
    )
    assert payload["stream"] is True
    assert payload["max_tokens"] == 64000


def test_tools_and_output_schema():
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "minLength": 1},
            "days": {"type": "integer", "minimum": 1, "nullable": True},
        },
        "required": ["city"],
    }
    payload = _payload(
        messages=[Message(role="user", content="weather?")],
        tools=[{"name": "get_weather", "description": "Weather", "parameters": schema}],
        output_schema={"title": "Out", "type": "object", "properties": {"x": {"type": "string"}}},
    )
    tool = payload["tools"][0]
    assert tool["name"] == "get_weather"
    assert tool["strict"] is True
    assert tool["input_schema"] == {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": ["integer", "null"]},
        },
        "required": ["city"],
        "additionalProperties": False,
    }
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert payload["output_config"]["format"]["schema"]["additionalProperties"] is False
    # Original schema untouched
    assert "minLength" in schema["properties"]["city"]


def test_non_strict_tools_send_schema_as_is():
    params = {"type": "object", "properties": {"a": {"type": "integer", "minimum": 0}}}
    payload = _payload(
        messages=[Message(role="user", content="x")],
        tools=[{"name": "t", "description": "", "parameters": params}],
        tool_strict=False,
    )
    assert payload["tools"][0] == {"name": "t", "input_schema": params}


def test_nested_schema_normalization():
    schema = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "object", "properties": {"a": {"type": "string"}}}, "maxItems": 3},
            "ref": {"$ref": "#/$defs/Thing"},
        },
        "$defs": {"Thing": {"type": "object", "properties": {"b": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}},
    }
    out = to_anthropic_schema(schema)
    assert out["properties"]["items"]["items"]["additionalProperties"] is False
    assert "maxItems" not in out["properties"]["items"]
    assert out["$defs"]["Thing"]["additionalProperties"] is False


def test_settings_mapping():
    payload = _payload(
        messages=[Message(role="user", content="x")],
        tools=[{"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}],
        settings=ModelSettings(
            reasoning_effort="high",
            max_tokens=2048,
            parallel_tool_calls=False,
            anthropic=AnthropicSettings(
                cache_ttl="1h", thinking_display="summarized", fallbacks="default"
            ),
        ),
    )
    assert payload["max_tokens"] == 2048
    assert payload["output_config"] == {"effort": "high"}
    assert payload["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert payload["fallbacks"] == "default"


def test_fallback_list_and_betas():
    request = ModelRequest(
        messages=[Message(role="user", content="x")],
        settings=ModelSettings(
            anthropic=AnthropicSettings(fallbacks=["claude-opus-5"], betas=["foo-2026-01-01"])
        ),
    )
    payload = to_messages_payload(request, "claude-opus-5-5")
    assert payload["fallbacks"] == [{"model": "claude-opus-5"}]
    assert beta_headers(request) == ["foo-2026-01-01", "server-side-fallback-2026-07-01"]


@pytest.mark.parametrize(
    "settings, message",
    [
        (ModelSettings(temperature=0.2), "temperature"),
        (ModelSettings(top_p=0.9), "temperature"),
        (ModelSettings(reasoning_effort="minimal"), "reasoning_effort"),
        (ModelSettings(openai=OpenAISettings(prompt_cache_key="k")), "OpenAI"),
    ],
)
def test_rejected_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        _payload(messages=[Message(role="user", content="x")], settings=settings)


def test_remote_provider_state_rejected():
    with pytest.raises(ValueError, match="stateless"):
        _payload(
            messages=[Message(role="user", content="x")],
            provider_state=ProviderState(previous_response_id="resp_1"),
        )
    with pytest.raises(ValueError, match="does not belong"):
        _payload(
            messages=[Message(role="user", content="x")],
            provider_state=ProviderState(provider="openai"),
        )


def test_multimodal_content():
    payload = _payload(
        messages=[
            Message(
                role="user",
                content=[
                    TextInput(text="look"),
                    ImageInput(url="https://example.com/a.png"),
                    ImageInput(data=b"\x89PNG", media_type="image/png"),
                    FileInput(data="data:application/pdf;base64,JVBERg==", filename="a.pdf"),
                    FileInput(file_id="file_123"),
                    FileInput(data=b"hello", media_type="text/plain"),
                ],
            )
        ]
    )
    blocks = payload["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "look"}
    assert blocks[1] == {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}
    assert blocks[2]["source"] == {"type": "base64", "media_type": "image/png", "data": "iVBORw=="}
    assert blocks[3] == {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERg=="},
        "title": "a.pdf",
    }
    assert blocks[4] == {"type": "document", "source": {"type": "file", "file_id": "file_123"}}
    assert blocks[5]["source"] == {"type": "text", "media_type": "text/plain", "data": "hello"}


def test_image_data_requires_media_type():
    with pytest.raises(ValueError, match="media_type"):
        _payload(messages=[Message(role="user", content=[ImageInput(data="abc")])])


def test_tool_history_replays_thinking_verbatim_and_merges_results():
    content = [
        THINKING,
        {"type": "text", "text": "Checking both."},
        {"type": "tool_use", "id": "toolu_1", "name": "a", "input": {"x": 1}},
        {"type": "tool_use", "id": "toolu_2", "name": "b", "input": {}},
    ]
    payload = _payload(
        messages=[
            Message(role="user", content="go"),
            SessionItem(
                type="function_call",
                tool_call_id="toolu_1",
                name="a",
                arguments={"x": 1},
                provider_data={"anthropic": {"content": content}},
            ),
            SessionItem(type="function_call", tool_call_id="toolu_2", name="b", arguments="{}"),
            SessionItem(type="function_call_output", tool_call_id="toolu_1", output={"ok": True}),
            SessionItem(type="function_call_output", tool_call_id="toolu_2", output="done"),
        ]
    )
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
        {"role": "assistant", "content": content},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"ok":true}'},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "done"},
            ],
        },
    ]


def test_tool_history_without_provider_data_builds_tool_use():
    payload = _payload(
        messages=[
            Message(role="user", content="go"),
            SessionItem(type="function_call", tool_call_id="t1", name="a", arguments='{"q":"x"}'),
            SessionItem(type="function_call", tool_call_id="t2", name="b", arguments=None),
            SessionItem(type="function_call_output", tool_call_id="t1", output="1"),
            SessionItem(type="function_call_output", tool_call_id="t2", output="2"),
            Message(role="assistant", content=""),
        ]
    )
    assert payload["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "t1", "name": "a", "input": {"q": "x"}},
            {"type": "tool_use", "id": "t2", "name": "b", "input": {}},
        ],
    }
    assert len(payload["messages"]) == 3  # empty assistant text is dropped


def test_mid_conversation_system_message_placement():
    valid = _payload(
        messages=[
            Message(role="system", content="base"),
            Message(role="user", content="q"),
            Message(role="developer", content="be brief"),
        ]
    )
    assert valid["system"] == "base"
    assert valid["messages"][-1] == {"role": "system", "content": "be brief"}

    # Guardrail retry: developer message right after an assistant answer.
    retry = _payload(
        messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="bad"),
            Message(role="developer", content="fix it"),
        ]
    )
    assert retry["messages"][-1] == {
        "role": "user",
        "content": [{"type": "text", "text": "<system>\nfix it\n</system>"}],
    }


def test_from_payload_tool_use_keeps_full_content():
    content = [
        THINKING,
        {"type": "tool_use", "id": "toolu_1", "name": "a", "input": {"x": 1}},
    ]
    response = from_messages_payload(
        {
            "id": "msg_1",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 5, "output_tokens": 7},
        },
        headers={"Request-Id": "req_1"},
        status_code=200,
        model="claude-opus-5-5",
    )
    assert response.request_id == "req_1"
    assert response.provider == "anthropic"
    assert response.stop_reason == "tool_use"
    assert response.output_text is None
    assert response.tool_calls[0].arguments == {"x": 1}
    assert response.tool_calls[0].provider_data == {"anthropic": {"content": content}}
    assert response.usage.input_tokens == 105
    assert response.usage.cached_input_tokens == 90
    assert response.usage.total_tokens == 112


def test_from_payload_text_and_refusal():
    response = from_messages_payload(
        {"content": [THINKING, {"type": "text", "text": "Hi"}], "stop_reason": "end_turn"},
        headers={},
        status_code=200,
        model="m",
    )
    assert response.output_text == "Hi"
    assert response.tool_calls == []

    with pytest.raises(ModelRefusalError) as excinfo:
        from_messages_payload(
            {"content": [], "stop_reason": "refusal", "stop_details": {"type": "refusal", "category": "cyber"}},
            headers={},
            status_code=200,
            model="m",
        )
    assert excinfo.value.stop_details["category"] == "cyber"


def test_truncated_tool_call_raises():
    with pytest.raises(ProviderError, match="max_tokens"):
        from_messages_payload(
            {"content": [{"type": "tool_use", "id": "t", "name": "a", "input": {}}], "stop_reason": "max_tokens"},
            headers={},
            status_code=200,
            model="m",
        )


STREAM = [
    {"type": "message_start", "message": {"id": "msg_s", "model": "claude-opus-5-5", "content": [], "usage": {"input_tokens": 12, "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-abc"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "ping"},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Let me "}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "check."}},
    {"type": "content_block_stop", "index": 1},
    {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "get_weather", "input": {}}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"city": '}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '"Paris"}'}},
    {"type": "content_block_stop", "index": 2},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 30}},
    {"type": "message_stop"},
]


def test_stream_events_rebuild_final_message():
    events = list(messages_stream_events(STREAM, headers={}, status_code=200, model="m"))
    text = [e.text_delta for e in events if e.type == ModelStreamEventType.TEXT_DELTA]
    tool = [e.tool_call for e in events if e.type == ModelStreamEventType.TOOL_CALL_DELTA]
    final = events[-1].response
    assert text == ["Let me ", "check."]
    assert [t.arguments for t in tool] == ['{"city": ', '"Paris"}']
    assert tool[0].tool_call_id == "toolu_9" and tool[0].name == "get_weather"
    assert final.output_text == "Let me check."
    assert final.stop_reason == "tool_use"
    assert final.tool_calls[0].arguments == {"city": "Paris"}
    assert final.usage.input_tokens == 12 and final.usage.output_tokens == 30
    replay = final.tool_calls[0].provider_data["anthropic"]["content"]
    assert replay[0] == THINKING
    assert replay[2] == {"type": "tool_use", "id": "toolu_9", "name": "get_weather", "input": {"city": "Paris"}}


def test_stream_error_event_raises():
    with pytest.raises(ProviderError) as excinfo:
        list(
            messages_stream_events(
                [STREAM[0], {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}],
                headers={},
                status_code=200,
                model="m",
            )
        )
    assert excinfo.value.status_code == 529


def _replayed_call(call_id, signature, binding):
    return SessionItem(
        type="function_call",
        tool_call_id=call_id,
        name="a",
        arguments={},
        provider_data={"anthropic": {"binding": binding, "content": [
            {"type": "thinking", "thinking": "", "signature": signature},
            {"type": "tool_use", "id": call_id, "name": "a", "input": {}},
        ]}},
    )


def _history_with(bindings):
    items = [Message(role="system", content="Agent B"), Message(role="user", content="go")]
    for index, binding in enumerate(bindings):
        items += [
            _replayed_call(f"t{index}", f"sig-{index}", binding),
            SessionItem(type="function_call_output", tool_call_id=f"t{index}", output="ok"),
        ]
    return items


def _signatures(payload):
    return [b["signature"] for m in payload["messages"] if m["role"] == "assistant" for b in m["content"] if b["type"] == "thinking"]


def test_stale_thinking_is_removed_as_a_leading_run():
    from modmex_ai.providers.anthropic.mapper import binding_digest

    current = binding_digest({"system": "Agent B"})
    payload = _payload(messages=_history_with(["old", current, "old", current, None]))
    # Last stale block is index 2: blocks 0..2 go, later current/legacy ones stay.
    assert _signatures(payload) == ["sig-3", "sig-4"]


def test_handoff_to_agent_with_other_instructions_drops_previous_thinking():
    from modmex_ai.providers.anthropic.mapper import binding_digest

    agent_a = binding_digest({"system": "Agent A"})
    payload = _payload(messages=_history_with([agent_a, agent_a]))
    assert _signatures(payload) == []
    # Same agent, same tools: nothing is removed.
    same = binding_digest({"system": "Agent B"})
    assert _signatures(_payload(messages=_history_with([same, same]))) == ["sig-0", "sig-1"]


def test_failed_tool_result_is_flagged_as_error():
    failure = {"ok": False, "error": {"code": "tool_failed", "retryable": False, "message": "x"}}
    payload = _payload(messages=[
        Message(role="user", content="go"),
        SessionItem(type="function_call", tool_call_id="t1", name="a", arguments={}),
        SessionItem(type="function_call_output", tool_call_id="t1", output=failure),
        SessionItem(type="function_call", tool_call_id="t2", name="a", arguments={}),
        SessionItem(type="function_call_output", tool_call_id="t2", output={"ok": True}),
    ])
    results = [b for m in payload["messages"] for b in m["content"] if b["type"] == "tool_result"]
    assert results[0]["is_error"] is True
    assert "is_error" not in results[1]
