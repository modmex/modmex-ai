from __future__ import annotations

import asyncio
import os
from typing import Any

from modmex_ai.agents import RunContext
from modmex_ai.errors import ProviderError
from modmex_ai.http import AsyncHttpClient, HttpClient, HttpFile
from modmex_ai.voice import (
    SpeechToTextProvider,
    StreamingTextToSpeechProvider,
)


class OpenAITranscriptionProvider(SpeechToTextProvider):
    """Request-based OpenAI speech-to-text adapter for completed audio turns."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini-transcribe",
        base_url: str = "https://api.openai.com/v1",
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.filename = filename
        self.content_type = content_type
        self.language = language
        self.prompt = prompt
        self.http_client = http_client or HttpClient()

    async def transcribe(self, audio: bytes, *, context: RunContext) -> str:
        response = await asyncio.to_thread(
            self.http_client.post_multipart,
            f"{self.base_url}/audio/transcriptions",
            fields=self._fields(),
            files={
                "file": HttpFile(
                    data=audio,
                    filename=self.filename,
                    content_type=self.content_type,
                ),
            },
            headers=self._headers(),
        )
        body = response.body
        if isinstance(body, dict) and isinstance(body.get("text"), str):
            return body["text"]
        if isinstance(body, str):
            return body
        raise ProviderError("OpenAI transcription response did not include text")

    def _fields(self) -> dict[str, str]:
        fields = {"model": self.model}
        if self.language is not None:
            fields["language"] = self.language
        if self.prompt is not None:
            fields["prompt"] = self.prompt
        return fields

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}


class OpenAISpeechProvider(StreamingTextToSpeechProvider):
    """Request-based OpenAI text-to-speech adapter for completed assistant turns."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini-tts",
        voice: str = "marin",
        response_format: str = "wav",
        instructions: str | None = None,
        speed: float | None = None,
        base_url: str = "https://api.openai.com/v1",
        http_client: HttpClient | None = None,
        async_http_client: AsyncHttpClient | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.instructions = instructions
        self.speed = speed
        self.base_url = base_url.rstrip("/")
        self.http_client = http_client or HttpClient()
        self._owns_async_http_client = async_http_client is None
        self.async_http_client = async_http_client or (
            AsyncHttpClient()
            if http_client is None and AsyncHttpClient.is_available()
            else None
        )

    async def synthesize(self, text: str, *, context: RunContext) -> bytes:
        response = await asyncio.to_thread(
            self.http_client.post_json_bytes,
            f"{self.base_url}/audio/speech",
            headers=self._headers(),
            data=self._payload(text),
        )
        if not isinstance(response.body, bytes):
            raise ProviderError("OpenAI speech response did not contain audio bytes")
        return response.body

    async def synthesize_stream(self, text: str, *, context: RunContext):
        if self.async_http_client is not None:
            async for chunk in self.async_http_client.post_json_stream(
                f"{self.base_url}/audio/speech",
                headers=self._headers(),
                data=self._payload(text),
            ):
                yield chunk
            return
        chunks = self.http_client.post_json_stream(
            f"{self.base_url}/audio/speech",
            headers=self._headers(),
            data=self._payload(text),
        )
        while True:
            chunk = await asyncio.to_thread(_next_chunk, chunks)
            if chunk is None:
                return
            yield chunk

    async def aclose(self) -> None:
        """Release the internally created native async transport, if any."""
        if self._owns_async_http_client and self.async_http_client is not None:
            await self.async_http_client.close()

    def _payload(self, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": self.response_format,
        }
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        if self.speed is not None:
            payload["speed"] = self.speed
        return payload

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def _next_chunk(chunks) -> bytes | None:
    try:
        return next(chunks)
    except StopIteration:
        return None
