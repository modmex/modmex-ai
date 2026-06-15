import asyncio
import json

import pytest

from modmex_ai.agents import RunContext
from modmex_ai.providers.openai import (
    OpenAIRealtimeTranscriptionConfig,
    OpenAIRealtimeTranscriptionProvider,
)
from modmex_ai.errors import RealtimeConnectionError, RealtimeProtocolError
from modmex_ai.providers.openai.transcription import _event


class FakeTransport:
    def __init__(self, events):
        self.events = list(events)
        self.sent = []
        self.closed = False

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        return json.dumps(self.events.pop(0))

    async def close(self):
        self.closed = True


def test_openai_realtime_transcription_streams_deltas_and_commits_audio():
    async def audio():
        yield b"first"
        yield b"second"

    async def run():
        transport = FakeTransport([
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "delta": "Hello,",
            },
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "Hello, dispatcher.",
            },
        ])

        async def connect(url, headers):
            assert "model=gpt-realtime-whisper" in url
            assert headers == {"Authorization": "Bearer key"}
            return transport

        provider = OpenAIRealtimeTranscriptionProvider(
            api_key="key",
            config=OpenAIRealtimeTranscriptionConfig(language="en", delay="low"),
            transport_factory=connect,
        )
        result = [item async for item in provider.transcribe_stream(audio(), context=RunContext(input=None))]

        assert [(item.text, item.is_final) for item in result] == [
            ("Hello,", False),
            ("Hello, dispatcher.", True),
        ]
        assert transport.sent[0] == {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {
                            "model": "gpt-realtime-whisper",
                            "language": "en",
                            "delay": "low",
                        },
                        "turn_detection": None,
                    },
                },
            },
        }
        assert [event["type"] for event in transport.sent[1:]] == [
            "input_audio_buffer.append",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
        ]
        assert transport.closed

    asyncio.run(run())


def test_openai_realtime_transcription_session_handles_multiple_committed_turns():
    async def run():
        transport = FakeTransport([
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": "turn-1", "transcript": "First."},
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": "turn-2", "transcript": "Second."},
        ])

        async def connect(_url, _headers):
            return transport

        provider = OpenAIRealtimeTranscriptionProvider(api_key="key", transport_factory=connect)
        session = await provider.connect()
        await session.append_audio(b"one")
        await session.commit_turn()
        events = session.events()
        first = await anext(events)
        await session.append_audio(b"two")
        await session.commit_turn()
        second = await anext(events)
        await session.close()

        assert [(first.text, first.item_id), (second.text, second.item_id)] == [
            ("First.", "turn-1"),
            ("Second.", "turn-2"),
        ]
        assert transport.closed

    asyncio.run(run())


def test_realtime_transcription_supports_single_turn_and_validates_connection_and_events():
    async def run():
        transport = FakeTransport([{
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "Single turn.",
        }])

        async def connect(_url, _headers):
            return transport

        provider = OpenAIRealtimeTranscriptionProvider(api_key="key", transport_factory=connect)
        assert await provider.transcribe(b"audio", context=RunContext(input=None)) == "Single turn."
        with pytest.raises(RealtimeConnectionError):
            await OpenAIRealtimeTranscriptionProvider(api_key="", transport_factory=connect).connect()

    asyncio.run(run())
    with pytest.raises(RealtimeProtocolError):
        _event("not-json")
    with pytest.raises(RealtimeProtocolError):
        _event(json.dumps({"missing": "type"}))


def test_realtime_transcription_propagates_audio_producer_failure():
    async def audio():
        raise RuntimeError("microphone failed")
        yield b"never"

    async def run():
        class SlowTransport(FakeTransport):
            async def recv(self):
                await asyncio.sleep(0.01)
                return await super().recv()

        transport = SlowTransport([{"type": "session.ended"}])
        async def connect(_url, _headers):
            return transport
        provider = OpenAIRealtimeTranscriptionProvider(api_key="key", transport_factory=connect)
        with pytest.raises(RuntimeError, match="microphone failed"):
            _ = [item async for item in provider.transcribe_stream(audio(), context=RunContext(input=None))]
        assert transport.closed

    asyncio.run(run())


def test_realtime_transcription_live_session_handles_bytes_and_provider_error():
    async def run():
        transport = FakeTransport([
            {"type": "conversation.item.input_audio_transcription.delta", "delta": "partial", "content_index": 1},
            {"type": "error", "error": {"message": "bad audio"}},
        ])
        session = await OpenAIRealtimeTranscriptionProvider(
            api_key="key",
            transport_factory=lambda _url, _headers: _return(transport),
        ).connect_live()
        first = await anext(session.transcriptions())
        assert first.text == "partial"
        with pytest.raises(RealtimeProtocolError, match="bad audio"):
            await anext(session.transcriptions())
        await session.close()

    asyncio.run(run())


async def _return(value):
    return value
