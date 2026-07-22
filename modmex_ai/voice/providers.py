from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from enum import Enum

from modmex import BaseModel

from modmex_ai.agents import RunContext


class SpeechToTextProvider(ABC):
    """Converts one completed user-audio turn into text."""

    @abstractmethod
    async def transcribe(self, audio: bytes, *, context: RunContext) -> str:
        ...


class TextToSpeechProvider(ABC):
    """Converts one completed assistant-text turn into audio."""

    @abstractmethod
    async def synthesize(self, text: str, *, context: RunContext) -> bytes:
        ...


class Transcription(BaseModel):
    text: str
    is_final: bool = False
    item_id: str | None = None
    content_index: int | None = None


class StreamingSpeechToTextProvider(SpeechToTextProvider):
    """Produces partial and final transcripts from a live audio stream."""

    @abstractmethod
    def transcribe_stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        context: RunContext,
    ) -> AsyncIterator[Transcription]:
        ...


class LiveSpeechToTextSession(ABC):
    """Receives live audio and emits transcripts across multiple user turns."""

    @abstractmethod
    async def append_audio(self, chunk: bytes) -> None:
        """Append an audio chunk for the current user turn."""

    @abstractmethod
    async def commit_turn(self) -> None:
        """Tell the provider that the current user turn is complete."""

    @abstractmethod
    def transcriptions(self) -> AsyncIterator[Transcription]:
        """Yield partial and final transcripts from the live connection."""

    @abstractmethod
    async def close(self) -> None:
        """Release the live transcription connection."""


class LiveSpeechToTextProvider(StreamingSpeechToTextProvider):
    """Creates a reusable transcription connection for a live conversation."""

    @abstractmethod
    async def connect_live(self) -> LiveSpeechToTextSession:
        """Open and configure the provider connection once per conversation."""


class VoiceInputEventType(str, Enum):
    AUDIO = "audio"
    COMMIT_TURN = "commit_turn"


class VoiceInputEvent(BaseModel):
    """One host-controlled action in a continuous voice conversation."""

    type: VoiceInputEventType
    audio: bytes | None = None

    @classmethod
    def audio_chunk(cls, audio: bytes) -> "VoiceInputEvent":
        return cls(type=VoiceInputEventType.AUDIO, audio=audio)

    @classmethod
    def commit_turn(cls) -> "VoiceInputEvent":
        return cls(type=VoiceInputEventType.COMMIT_TURN)


class StreamingTextToSpeechProvider(TextToSpeechProvider):
    """Produces playable audio chunks while synthesizing one assistant turn."""

    @abstractmethod
    def synthesize_stream(
        self,
        text: str,
        *,
        context: RunContext,
    ) -> AsyncIterator[bytes]:
        ...


class CallableSpeechToTextProvider(SpeechToTextProvider):
    """Adapts an async or sync callable without importing a provider SDK."""

    def __init__(self, transcribe: Callable[[bytes, RunContext], str | Awaitable[str]]) -> None:
        self._transcribe = transcribe

    async def transcribe(self, audio: bytes, *, context: RunContext) -> str:
        result = self._transcribe(audio, context)
        return await result if inspect.isawaitable(result) else result


class CallableTextToSpeechProvider(TextToSpeechProvider):
    """Adapts an async or sync callable without importing a provider SDK."""

    def __init__(self, synthesize: Callable[[str, RunContext], bytes | Awaitable[bytes]]) -> None:
        self._synthesize = synthesize

    async def synthesize(self, text: str, *, context: RunContext) -> bytes:
        result = self._synthesize(text, context)
        return await result if inspect.isawaitable(result) else result
