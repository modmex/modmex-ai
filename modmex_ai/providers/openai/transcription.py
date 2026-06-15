from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import field
from typing import Any
from urllib.parse import urlencode

from modmex import BaseModel

from modmex_ai.agents import RunContext
from modmex_ai.errors import RealtimeConnectionError, RealtimeProtocolError
from modmex_ai.realtime import RealtimeTransport
from modmex_ai.voice import (
    LiveSpeechToTextProvider,
    LiveSpeechToTextSession,
    Transcription,
)


class OpenAIRealtimeTranscriptionConfig(BaseModel):
    model: str = "gpt-realtime-whisper"
    language: str | None = None
    delay: str | None = None
    audio_format: dict[str, Any] = field(default_factory=lambda: {
        "type": "audio/pcm",
        "rate": 24000,
    })

    def session_update(self) -> dict[str, Any]:
        transcription: dict[str, Any] = {"model": self.model}
        if self.language is not None:
            transcription["language"] = self.language
        if self.delay is not None:
            transcription["delay"] = self.delay
        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": self.audio_format,
                        "transcription": transcription,
                        "turn_detection": None,
                    },
                },
            },
        }


TransportFactory = Callable[[str, dict[str, str]], Awaitable[RealtimeTransport]]


class OpenAIRealtimeTranscriptionProvider(LiveSpeechToTextProvider):
    """OpenAI WebSocket transcription adapter for one live, manually committed turn."""

    websocket_url = "wss://api.openai.com/v1/realtime"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        config: OpenAIRealtimeTranscriptionConfig | None = None,
        websocket_url: str | None = None,
        transport_factory: TransportFactory | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.config = config or OpenAIRealtimeTranscriptionConfig()
        self.websocket_url = websocket_url or self.websocket_url
        self.transport_factory = transport_factory or _connect_transport

    async def transcribe(self, audio: bytes, *, context: RunContext) -> str:
        async def chunks():
            yield audio

        final = ""
        async for transcription in self.transcribe_stream(chunks(), context=context):
            if transcription.is_final:
                final = transcription.text
        return final

    async def connect(self) -> "OpenAIRealtimeTranscriptionSession":
        if not self.api_key:
            raise RealtimeConnectionError("OPENAI_API_KEY is required for Realtime transcription")
        url = f"{self.websocket_url}?{urlencode({'model': self.config.model})}"
        transport = await self.transport_factory(url, {"Authorization": f"Bearer {self.api_key}"})
        session = OpenAIRealtimeTranscriptionSession(transport)
        await session.configure(self.config)
        return session

    async def connect_live(self) -> "OpenAIRealtimeTranscriptionSession":
        return await self.connect()

    async def transcribe_stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        context: RunContext,
    ) -> AsyncIterator[Transcription]:
        session = await self.connect()
        producer: asyncio.Task[None] | None = None
        try:
            producer = asyncio.create_task(self._append_audio(session, audio))
            async for transcription in session.events():
                yield transcription
                if transcription.is_final:
                    await producer
                    return
        finally:
            if producer is not None and not producer.done():
                producer.cancel()
            producer_results = (
                await asyncio.gather(producer, return_exceptions=True)
                if producer is not None
                else []
            )
            await session.close()
            if producer_results:
                producer_error = producer_results[0]
                if isinstance(producer_error, Exception) and not isinstance(
                    producer_error,
                    asyncio.CancelledError,
                ):
                    raise producer_error

    async def _append_audio(self, session: "OpenAIRealtimeTranscriptionSession", audio: AsyncIterator[bytes]) -> None:
        async for chunk in audio:
            await session.append_audio(chunk)
        await session.commit_turn()


class OpenAIRealtimeTranscriptionSession(LiveSpeechToTextSession):
    """A reusable transcription connection for many manually committed turns."""

    def __init__(self, transport: RealtimeTransport) -> None:
        self.transport = transport

    async def configure(self, config: OpenAIRealtimeTranscriptionConfig) -> None:
        await self.transport.send(json.dumps(config.session_update(), separators=(",", ":")))

    async def append_audio(self, chunk: bytes) -> None:
        await self.transport.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")}, separators=(",", ":")))

    async def commit_turn(self) -> None:
        await self.transport.send(json.dumps({"type": "input_audio_buffer.commit"}, separators=(",", ":")))

    def transcriptions(self) -> AsyncIterator[Transcription]:
        return self.events()

    async def events(self) -> AsyncIterator[Transcription]:
        while True:
            event = _event(await self.transport.recv())
            event_type = event["type"]
            if event_type == "conversation.item.input_audio_transcription.delta":
                yield Transcription(text=event.get("delta", ""), item_id=event.get("item_id"), content_index=event.get("content_index"))
            elif event_type == "conversation.item.input_audio_transcription.completed":
                yield Transcription(text=event.get("transcript", ""), is_final=True, item_id=event.get("item_id"), content_index=event.get("content_index"))
            elif event_type == "error":
                raise RealtimeProtocolError(str(event.get("error", event)))

    async def close(self) -> None:
        await self.transport.close()


async def _connect_transport(url: str, headers: dict[str, str]) -> RealtimeTransport:
    try:
        import websockets
    except ImportError as error:
        raise RealtimeConnectionError(
            "Install modmex-ai[realtime] to use OpenAI Realtime transcription"
        ) from error
    try:
        return await websockets.connect(url, additional_headers=headers)
    except Exception as error:
        raise RealtimeConnectionError("Could not connect to OpenAI Realtime transcription") from error


def _event(message: str | bytes) -> dict[str, Any]:
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    try:
        event = json.loads(message)
    except (TypeError, json.JSONDecodeError) as error:
        raise RealtimeProtocolError("Realtime transcription returned invalid JSON") from error
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise RealtimeProtocolError("Realtime transcription event must include a type")
    return event
