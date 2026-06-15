import asyncio
import pytest

from modmex_ai.agents import RunContext
from modmex_ai.http import HttpResponse
from modmex_ai.providers.openai import OpenAISpeechProvider, OpenAITranscriptionProvider
from modmex_ai.errors import ProviderError


class FakeAudioHttpClient:
    def __init__(self):
        self.calls = []

    def post_multipart(self, url, *, fields, files, headers):
        self.calls.append(("multipart", url, fields, files, headers))
        return HttpResponse(status_code=200, headers={}, body={"text": "hola"})

    def post_json_bytes(self, url, *, data, headers):
        self.calls.append(("json_bytes", url, data, headers))
        return HttpResponse(status_code=200, headers={}, body=b"speech")

    def post_json_stream(self, url, *, data, headers):
        self.calls.append(("json_stream", url, data, headers))
        yield b"first"
        yield b"second"


class FakeAsyncAudioHttpClient:
    def __init__(self):
        self.calls = []

    async def post_json_stream(self, url, *, data, headers):
        self.calls.append((url, data, headers))
        yield b"async-first"
        yield b"async-second"


def test_openai_audio_providers_use_lightweight_http_contracts():
    async def run():
        http = FakeAudioHttpClient()
        context = RunContext(input=None)
        transcription = OpenAITranscriptionProvider(
            api_key="key",
            model="gpt-4o-transcribe",
            language="es",
            prompt="Freight dispatch vocabulary.",
            http_client=http,
        )
        speech = OpenAISpeechProvider(
            api_key="key",
            voice="marin",
            instructions="Speak calmly.",
            http_client=http,
        )

        assert await transcription.transcribe(b"audio", context=context) == "hola"
        assert await speech.synthesize("Hola", context=context) == b"speech"
        assert b"".join([
            chunk async for chunk in speech.synthesize_stream("Hola", context=context)
        ]) == b"firstsecond"

        kind, url, fields, files, headers = http.calls[0]
        assert kind == "multipart"
        assert url.endswith("/audio/transcriptions")
        assert fields == {
            "model": "gpt-4o-transcribe",
            "language": "es",
            "prompt": "Freight dispatch vocabulary.",
        }
        assert files["file"].data == b"audio"
        assert headers == {"authorization": "Bearer key"}
        kind, url, data, headers = http.calls[2]
        assert kind == "json_stream"
        assert url.endswith("/audio/speech")
        assert data["input"] == "Hola"
        assert headers == {"authorization": "Bearer key"}
        kind, url, data, headers = http.calls[1]
        assert kind == "json_bytes"
        assert url.endswith("/audio/speech")
        assert data == {
            "model": "gpt-4o-mini-tts",
            "voice": "marin",
            "input": "Hola",
            "response_format": "wav",
            "instructions": "Speak calmly.",
        }
        assert headers == {"authorization": "Bearer key"}

    asyncio.run(run())


def test_openai_speech_provider_uses_async_stream_transport_when_provided():
    async def run():
        http = FakeAsyncAudioHttpClient()
        speech = OpenAISpeechProvider(api_key="key", async_http_client=http)

        assert b"".join([
            chunk async for chunk in speech.synthesize_stream("Hola", context=RunContext(input=None))
        ]) == b"async-firstasync-second"
        assert http.calls[0][0].endswith("/audio/speech")

    asyncio.run(run())


def test_openai_audio_providers_reject_invalid_provider_payloads_and_close_owned_async_client():
    class InvalidHttp(FakeAudioHttpClient):
        def post_multipart(self, *args, **kwargs):
            return HttpResponse(status_code=200, headers={}, body={})

        def post_json_bytes(self, *args, **kwargs):
            return HttpResponse(status_code=200, headers={}, body={})

    class AsyncClient(FakeAsyncAudioHttpClient):
        def __init__(self):
            super().__init__()
            self.closed = False

        async def close(self):
            self.closed = True

    async def run():
        context = RunContext(input=None)
        with pytest.raises(ProviderError):
            await OpenAITranscriptionProvider(http_client=InvalidHttp()).transcribe(b"a", context=context)
        with pytest.raises(ProviderError):
            await OpenAISpeechProvider(http_client=InvalidHttp()).synthesize("x", context=context)
        async_client = AsyncClient()
        speech = OpenAISpeechProvider(async_http_client=async_client)
        await speech.aclose()
        assert not async_client.closed

    asyncio.run(run())
