import asyncio
import json
import sys
import types

import pytest

from modmex_ai import Agent, GuardrailResult, Handoff, tool
from modmex_ai.providers.openai import (
    OpenAIRealtimeSession,
    OpenAIRealtimeSessionConfig,
)
from modmex_ai.providers.openai.realtime import OpenAIRealtimeClient, _usage_from_response
from modmex_ai.errors import RealtimeConnectionError
from modmex_ai.errors import RealtimeProtocolError
from modmex_ai.voice import VoiceSessionEventType
from modmex_ai.voice import VoiceTerminationReason


class FakeRealtimeTransport:
    def __init__(self, events=None):
        self.events = list(events or [])
        self.sent = []
        self.closed = False

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        return json.dumps(self.events.pop(0))

    async def close(self):
        self.closed = True


def test_openai_realtime_configures_agent_tools_and_voice():
    def lookup_order(order_number: str):
        return {"order_number": order_number, "status": "shipped"}

    async def run():
        transport = FakeRealtimeTransport()
        agent = Agent(
            name="orders",
            instructions="Handle order questions.",
            tools=[lookup_order],
        )
        session = OpenAIRealtimeSession(
            agent=agent,
            transport=transport,
            config=OpenAIRealtimeSessionConfig(voice="marin"),
        )
        await session.configure()

        event = transport.sent[0]
        assert event["type"] == "session.update"
        assert event["session"]["model"] == "gpt-realtime-2.1"
        assert event["session"]["audio"]["output"]["voice"] == "marin"
        assert event["session"]["tools"] == [{
            "type": "function",
            "name": "lookup_order",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"order_number": {"type": "string"}},
                "required": ["order_number"],
                "additionalProperties": False,
            },
            "strict": True,
        }]

    asyncio.run(run())


def test_openai_realtime_executes_function_call_and_continues_response():
    def lookup_order(order_number: str):
        return {"order_number": order_number, "status": "shipped"}

    async def run():
        transport = FakeRealtimeTransport([{
            "type": "response.function_call_arguments.done",
            "call_id": "call_1",
            "name": "lookup_order",
            "arguments": '{"order_number":"A-100"}',
        }])
        session = OpenAIRealtimeSession(
            agent=Agent(
                name="orders",
                instructions="Handle order questions.",
                tools=[lookup_order],
            ),
            transport=transport,
        )

        await session.handle(await session.receive())

        assert transport.sent == [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"order_number":"A-100","status":"shipped"}',
                },
            },
            {"type": "response.create", "response": {}},
        ]
        assert [step.type for step in session.context.trace.steps] == [
            "realtime_server_event",
            "realtime_tool_call",
            "realtime_client_event",
            "realtime_client_event",
        ]

    asyncio.run(run())


def test_openai_realtime_switches_the_active_agent_after_a_handoff():
    async def run():
        transport = FakeRealtimeTransport([{
            "type": "response.function_call_arguments.done",
            "call_id": "call_1",
            "name": "transfer_to_billing",
            "arguments": "{}",
        }])
        billing = Agent(name="billing", instructions="Handle billing questions.")
        triage = Agent(
            name="triage",
            instructions="Route requests.",
            handoffs=[Handoff("billing")],
        )
        session = OpenAIRealtimeSession(
            agent=triage,
            agents=[billing],
            transport=transport,
            config=OpenAIRealtimeSessionConfig(model="gpt-test"),
        )

        await session.handle(await session.receive())

        assert session.current_agent is billing
        assert transport.sent[0]["type"] == "session.update"
        assert transport.sent[0]["session"]["instructions"] == "Handle billing questions."
        assert transport.sent[1]["item"]["output"] == '{"transferred":true,"handoff_input":{}}'
        assert transport.sent[2] == {"type": "response.create", "response": {}}

    asyncio.run(run())


def test_openai_realtime_collects_response_usage_and_sip_omits_model_update():
    async def run():
        transport = FakeRealtimeTransport([{
            "type": "response.done",
            "response": {
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 8,
                    "total_tokens": 20,
                    "input_token_details": {"cached_tokens": 3},
                    "output_token_details": {"reasoning_tokens": 2},
                }
            },
        }])
        session = OpenAIRealtimeSession(
            agent=Agent(name="voice", instructions="Help the caller."),
            transport=transport,
            include_model_on_update=False,
        )

        await session.configure()
        await session.handle(await session.receive())

        assert "model" not in transport.sent[0]["session"]
        assert session.usage.model_dump() == {
            "input_tokens": 12,
            "output_tokens": 8,
            "total_tokens": 20,
            "cached_input_tokens": 3,
            "reasoning_output_tokens": 2,
            "details": {"raw_items": [{
                "input_tokens": 12,
                "output_tokens": 8,
                "total_tokens": 20,
                "input_token_details": {"cached_tokens": 3},
                "output_token_details": {"reasoning_tokens": 2},
            }]},
        }

    asyncio.run(run())


def test_openai_realtime_uses_one_configuration_shape_for_accept_and_session_update():
    agent = Agent(name="voice", instructions="Help the caller.")
    config = OpenAIRealtimeSessionConfig(
        model="gpt-test",
        voice="marin",
        turn_detection={"type": "server_vad"},
    )

    accept = config.to_accept_payload(agent)
    update = config.to_session_update(agent, include_model=False)

    assert accept["model"] == "gpt-test"
    assert "model" not in update
    assert {key: value for key, value in accept.items() if key != "model"} == update


def test_openai_realtime_normalizes_a_provider_error_as_terminal_voice_events():
    async def run():
        transport = FakeRealtimeTransport([{
            "type": "error",
            "error": {"message": "invalid session"},
        }])
        session = OpenAIRealtimeSession(
            agent=Agent(name="voice", instructions="Help the caller."),
            transport=transport,
        )

        events = [event async for event in session.voice_events()]

        assert [event.type for event in events] == [
            VoiceSessionEventType.ERROR,
            VoiceSessionEventType.SESSION_ENDED,
        ]
        assert events[0].data["reason"] == VoiceTerminationReason.PROVIDER_ERROR
        assert events[1].data == {"reason": VoiceTerminationReason.PROVIDER_ERROR}

    asyncio.run(run())


def test_openai_realtime_terminates_cleanly_when_a_tool_guardrail_blocks_execution():
    class RejectToolOutput:
        name = "reject_tool_output"

        def check(self, value, context=None):
            return GuardrailResult(passed=False, reason="requires approval")

    @tool(output_guardrails=[RejectToolOutput()])
    def sensitive_action() -> dict:
        return {"performed": True}

    async def run():
        transport = FakeRealtimeTransport([{
            "type": "response.function_call_arguments.done",
            "call_id": "call-sensitive",
            "name": "sensitive_action",
            "arguments": "{}",
        }])
        session = OpenAIRealtimeSession(
            agent=Agent(
                name="voice",
                instructions="Help the caller.",
                tools=[sensitive_action],
            ),
            transport=transport,
        )

        events = [event async for event in session.voice_events()]

        assert [event.type for event in events] == [
            VoiceSessionEventType.ERROR,
            VoiceSessionEventType.SESSION_ENDED,
        ]
        assert events[0].data["reason"] == VoiceTerminationReason.TOOL_ERROR
        assert "requires approval" in events[0].data["message"]

    asyncio.run(run())


def test_openai_realtime_exposes_provider_neutral_voice_events():
    def lookup_order(order_number: str):
        return {"order_number": order_number, "status": "shipped"}

    async def run():
        transport = FakeRealtimeTransport([
            {"type": "session.created"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_user_1",
                "transcript": "Where is order A-100?",
            },
            {
                "type": "response.function_call_arguments.done",
                "call_id": "tool_1",
                "name": "lookup_order",
                "arguments": '{"order_number":"A-100"}',
            },
            {"type": "response.done", "response": {"usage": {}}},
        ])
        session = OpenAIRealtimeSession(
            agent=Agent(
                name="orders",
                instructions="Handle order questions.",
                tools=[lookup_order],
            ),
            transport=transport,
        )

        received = []
        async for event in session.voice_events():
            received.append(event)
            if event.type == VoiceSessionEventType.RESPONSE_COMPLETED:
                break

        assert [event.type for event in received] == [
            VoiceSessionEventType.SESSION_STARTED,
            VoiceSessionEventType.TRANSCRIPT_FINAL,
            VoiceSessionEventType.TOOL_FINISHED,
            VoiceSessionEventType.RESPONSE_COMPLETED,
        ]
        assert received[1].data == {
            "transcript": "Where is order A-100?",
            "item_id": "item_user_1",
        }
        assert received[2].data == {
            "tool_call_id": "tool_1",
            "tool_name": "lookup_order",
        }

    asyncio.run(run())


def test_openai_realtime_integrates_tool_handoff_and_remote_close_in_one_session():
    def lookup_lane(origin: str) -> dict:
        return {"origin": origin, "available": True}

    async def run():
        transport = FakeRealtimeTransport([
            {
                "type": "response.function_call_arguments.done",
                "call_id": "tool-1",
                "name": "lookup_lane",
                "arguments": '{"origin":"Dallas"}',
            },
            {
                "type": "response.function_call_arguments.done",
                "call_id": "handoff-1",
                "name": "transfer_to_dispatch",
                "arguments": "{}",
            },
            {"type": "session.ended"},
        ])
        dispatch = Agent(name="dispatch", instructions="Handle the load.")
        triage = Agent(
            name="triage",
            instructions="Qualify then route.",
            tools=[lookup_lane],
            handoffs=["dispatch"],
        )
        session = OpenAIRealtimeSession(
            agent=triage,
            agents=[dispatch],
            transport=transport,
        )

        events = []
        async for event in session.voice_events():
            events.append(event)
            if event.type == VoiceSessionEventType.SESSION_ENDED:
                break

        assert [event.type for event in events] == [
            VoiceSessionEventType.TOOL_FINISHED,
            VoiceSessionEventType.HANDOFF_COMPLETED,
            VoiceSessionEventType.SESSION_ENDED,
        ]
        assert events[-1].data == {"reason": VoiceTerminationReason.ENDED_BY_CALLER}
        assert session.current_agent is dispatch
        assert transport.sent == [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": "tool-1",
                    "output": '{"origin":"Dallas","available":true}',
                },
            },
            {"type": "response.create", "response": {}},
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": "gpt-realtime-2.1",
                    "instructions": "Handle the load.",
                    "output_modalities": ["audio"],
                    "tools": [],
                    "tool_choice": "auto",
                },
            },
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": "handoff-1",
                    "output": '{"transferred":true,"handoff_input":{}}',
                },
            },
            {"type": "response.create", "response": {}},
        ]

    asyncio.run(run())


def test_realtime_session_message_audio_close_and_event_helpers():
    async def run():
        transport = FakeRealtimeTransport()
        session = OpenAIRealtimeSession(agent=Agent(name="a", instructions="Help."), transport=transport)
        await session.send_text("hello")
        await session.append_audio("base64")
        await session.commit_audio()
        await session.close()
        assert [event["type"] for event in transport.sent] == [
            "conversation.item.create", "response.create", "input_audio_buffer.append",
            "input_audio_buffer.commit", "response.create",
        ]
        assert transport.closed
        assert session._function_call(type("E", (), {"type": "response.output_item.done", "data": {"item": {"type": "message"}}})()) is None

    asyncio.run(run())
    assert _usage_from_response(None).total_tokens == 0
    assert _usage_from_response({"usage": None}).total_tokens == 0
    assert OpenAIRealtimeSession(agent=Agent(name="a", instructions="x"), transport=FakeRealtimeTransport())._voice_event_from(
        type("E", (), {"type": "response.output_audio_transcript.done", "data": {"transcript": "hi", "item_id": "i"}, "raw": {}})()
    ).data == {"transcript": "hi", "item_id": "i"}


def test_realtime_client_requires_key_without_opening_a_socket():
    async def run():
        with pytest.raises(RealtimeConnectionError):
            await OpenAIRealtimeClient(api_key="").connect(agent=Agent(name="a", instructions="x"))

    asyncio.run(run())


def test_realtime_session_rejects_invalid_transport_messages_and_context_closes():
    class InvalidTransport(FakeRealtimeTransport):
        async def recv(self):
            return b"not-json"

    async def run():
        transport = InvalidTransport()
        session = OpenAIRealtimeSession(agent=Agent(name="a", instructions="x"), transport=transport)
        with pytest.raises(RealtimeProtocolError):
            await session.receive()
        async with OpenAIRealtimeSession(agent=Agent(name="b", instructions="x"), transport=transport):
            pass
        assert transport.closed

    asyncio.run(run())


def test_realtime_client_connects_for_websocket_and_sip_and_normalizes_connection_failure(monkeypatch):
    captured = []

    async def connect(url, *, additional_headers):
        captured.append((url, additional_headers))
        return FakeRealtimeTransport()

    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=connect))

    async def run():
        client = OpenAIRealtimeClient(api_key="key", websocket_url="wss://example.test/realtime")
        web = await client.connect(agent=Agent(name="a", instructions="x"), safety_identifier="user-1")
        sip = await client.connect(agent=Agent(name="a", instructions="x"), realtime_call_id="call-1")
        assert web.include_model_on_update
        assert not sip.include_model_on_update

    asyncio.run(run())
    assert "model=gpt-realtime-2.1" in captured[0][0]
    assert captured[0][1]["OpenAI-Safety-Identifier"] == "user-1"
    assert "call_id=call-1" in captured[1][0]

    async def fail(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=fail))
    async def run_failure():
        with pytest.raises(RealtimeConnectionError, match="Could not connect"):
            await OpenAIRealtimeClient(api_key="key").connect(agent=Agent(name="a", instructions="x"))

    asyncio.run(run_failure())


def test_realtime_normalizes_unknown_handoff_and_ignores_malformed_function_output_item():
    async def run():
        session = OpenAIRealtimeSession(
            agent=Agent(name="a", instructions="x", handoffs=[Handoff("missing")]),
            transport=FakeRealtimeTransport(),
        )
        malformed = type("Event", (), {"type": "response.output_item.done", "data": {"item": {"type": "function_call", "name": 1}}, "raw": {}})()
        assert session._function_call(malformed) is None
        event = type("Event", (), {"type": "response.function_call_arguments.done", "data": {"call_id": "h", "name": "transfer_to_missing", "arguments": "{}"}, "raw": {}})()
        with pytest.raises(Exception, match="unknown agent"):
            await session.handle(event)

    asyncio.run(run())
