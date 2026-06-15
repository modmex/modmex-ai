import urllib.error
import asyncio

import pytest

from modmex_ai.errors import ModelTimeoutError, ProviderError, RateLimitError
from modmex_ai.http import AsyncHttpClient
from modmex_ai.http import async_client as async_http
from modmex_ai.http.client import HttpClient, HttpFile, HttpResponse
from modmex_ai.http.retries import RetryConfig


def test_http_client_retries_retryable_provider_errors(monkeypatch):
    client = HttpClient(retry=RetryConfig(attempts=1, backoff_seconds=0))
    calls = {"count": 0}

    def send(_request, *, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ProviderError("temporary", status_code=500)
        return HttpResponse(status_code=200, headers={}, body={"ok": True})

    monkeypatch.setattr(client, "_send", send)

    response = client.post_json("https://example.test")

    assert response.body == {"ok": True}
    assert calls["count"] == 2


def test_http_client_maps_429_to_rate_limit(monkeypatch):
    class FakeHeaders:
        def items(self):
            return [("x-request-id", "req-1")]

    class FakeHTTPError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("url", 429, "rate limited", FakeHeaders(), None)

        def read(self):
            return b'{"error":"limited"}'

    def urlopen(_request, timeout):
        raise FakeHTTPError()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    with pytest.raises(RateLimitError) as exc:
        HttpClient(retry=RetryConfig(attempts=0)).post_json("https://example.test")

    assert exc.value.request_id == "req-1"
    assert exc.value.response_body == {"error": "limited"}


def test_http_client_send_success(monkeypatch):
    class FakeHeaders:
        def items(self):
            return [("x-request-id", "req-1")]

    class FakeResponse:
        status = 200
        headers = FakeHeaders()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"ok":true}'

    monkeypatch.setattr("urllib.request.urlopen", lambda _request, timeout: FakeResponse())

    response = HttpClient().post_json("https://example.test")

    assert response.status_code == 200
    assert response.body == {"ok": True}


def test_http_client_send_success_with_empty_body(monkeypatch):
    class FakeHeaders:
        def items(self):
            return []

    class FakeResponse:
        status = 204
        headers = FakeHeaders()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b""

    monkeypatch.setattr("urllib.request.urlopen", lambda _request, timeout: FakeResponse())

    response = HttpClient().post_json("https://example.test")

    assert response.status_code == 204
    assert response.body == {}


def test_http_client_maps_timeout(monkeypatch):
    def urlopen(_request, timeout):
        raise TimeoutError()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    with pytest.raises(ModelTimeoutError):
        HttpClient().post_json("https://example.test")


def test_http_client_maps_http_error_with_text_body(monkeypatch):
    class FakeHeaders:
        def items(self):
            return [("request-id", "req-2")]

    class FakeHTTPError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("url", 500, "server error", FakeHeaders(), None)

        def read(self):
            return b"not-json"

    monkeypatch.setattr("urllib.request.urlopen", lambda _request, timeout: (_ for _ in ()).throw(FakeHTTPError()))

    with pytest.raises(ProviderError) as exc:
        HttpClient(retry=RetryConfig(attempts=0)).post_json("https://example.test")

    assert exc.value.request_id == "req-2"
    assert exc.value.response_body == "not-json"


def test_http_client_maps_url_error(monkeypatch):
    def urlopen(_request, timeout):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    with pytest.raises(ProviderError):
        HttpClient().post_json("https://example.test")


def test_http_client_builds_multipart_requests(monkeypatch):
    captured = {}

    def send(request, *, timeout, decode_json=True):
        captured["content_type"] = request.headers["Content-type"]
        captured["body"] = request.data
        return HttpResponse(status_code=200, headers={}, body={"text": "hello"})

    client = HttpClient()
    monkeypatch.setattr(client, "_send", send)

    client.post_multipart(
        "https://example.test/audio",
        fields={"model": "test"},
        files={"file": HttpFile(b"audio", "audio.wav", "audio/wav")},
    )

    assert captured["content_type"].startswith("multipart/form-data; boundary=")
    assert b'name="model"' in captured["body"]
    assert b'filename="audio.wav"' in captured["body"]
    assert b"audio" in captured["body"]


def test_http_client_returns_raw_bytes(monkeypatch):
    class FakeHeaders:
        def items(self):
            return []

    class FakeResponse:
        status = 200
        headers = FakeHeaders()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"audio-bytes"

    monkeypatch.setattr("urllib.request.urlopen", lambda _request, timeout: FakeResponse())

    assert HttpClient().post_json_bytes("https://example.test/speech").body == b"audio-bytes"


def test_http_client_yields_chunked_binary_responses(monkeypatch):
    class FakeHeaders:
        def items(self):
            return []

    class FakeResponse:
        status = 200
        headers = FakeHeaders()

        def __init__(self):
            self.chunks = [b"first", b"second", b""]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _chunk_size):
            return self.chunks.pop(0)

    monkeypatch.setattr("urllib.request.urlopen", lambda _request, timeout: FakeResponse())

    assert list(HttpClient().post_json_stream("https://example.test/speech")) == [
        b"first",
        b"second",
    ]


def test_async_http_client_reuses_owned_transport_and_closes_it():
    class Response:
        status_code = 200
        headers = {}
        content = b'{"ok":true}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class Transport:
        def __init__(self):
            self.posts = []
            self.closed = False

        async def post(self, url, **kwargs):
            self.posts.append((url, kwargs))
            return Response()

        async def aclose(self):
            self.closed = True

    async def run():
        transport = Transport()
        client = AsyncHttpClient(client=transport, own_client=True)

        assert (await client.post_json("https://example.test/one")).body == {"ok": True}
        assert (await client.post_json("https://example.test/two")).body == {"ok": True}
        await client.close()

        assert len(transport.posts) == 2
        assert transport.closed
        assert client.is_closed
        with pytest.raises(RuntimeError, match="closed"):
            await client.post_json("https://example.test/three")

    asyncio.run(run())


def test_async_http_client_does_not_close_injected_transport_by_default():
    class Transport:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    async def run():
        transport = Transport()
        async with AsyncHttpClient(client=transport):
            pass

        assert not transport.closed

    asyncio.run(run())


def test_async_http_client_streams_retries_and_maps_provider_failures(monkeypatch):
    monkeypatch.setattr(async_http, "_provider_error", lambda error: error)
    class Response:
        status_code = 200
        headers = {"x-request-id": "r"}
        content = b'{"ok":true}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

        async def aiter_bytes(self, *, chunk_size):
            yield b"one"
            yield b""
            yield b"two"

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, *_args):
            return None

    class Transport:
        def __init__(self):
            self.calls = 0

        async def post(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("temporary", status_code=500)
            return Response()

        def stream(self, *_args, **_kwargs):
            return Stream()

        async def aclose(self):
            return None

    async def run():
        transport = Transport()
        client = AsyncHttpClient(client=transport, retry=RetryConfig(attempts=1, backoff_seconds=0))
        assert (await client.post_json("https://example.test")).body == {"ok": True}
        assert [chunk async for chunk in client.post_json_stream("https://example.test")] == [b"one", b"two"]

    asyncio.run(run())


def test_async_http_error_mapping_and_cancellation(monkeypatch):
    class Timeout(Exception):
        pass

    class Status(Exception):
        def __init__(self, response):
            self.response = response

    class Response:
        status_code = 429
        headers = {"request-id": "r"}
        text = "limited"

        def json(self):
            raise ValueError()

    class Httpx:
        TimeoutException = Timeout
        HTTPStatusError = Status

    monkeypatch.setattr(async_http, "_httpx", lambda: Httpx)
    assert isinstance(async_http._provider_error(Timeout()), ModelTimeoutError)
    error = async_http._provider_error(Status(Response()))
    assert isinstance(error, RateLimitError)
    assert error.response_body == "limited"

    async def cancelled():
        class Transport:
            async def post(self, *_args, **_kwargs):
                raise asyncio.CancelledError()
        client = AsyncHttpClient(client=Transport())
        with pytest.raises(asyncio.CancelledError):
            await client.post_json("https://example.test")

    asyncio.run(cancelled())


def test_http_stream_retries_a_retryable_http_error_and_exposes_terminal_error(monkeypatch):
    class Headers:
        def items(self):
            return []

    class Error(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("url", 500, "retry", Headers(), None)

        def read(self):
            return b'{"error":"retry"}'

    class Response:
        status = 200
        headers = Headers()
        chunks = [b"ok", b""]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            return self.chunks.pop(0)

    calls = {"count": 0}
    def urlopen(_request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise Error()
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert list(HttpClient(retry=RetryConfig(attempts=1, backoff_seconds=0)).post_json_stream("https://example.test")) == [b"ok"]


def test_http_stream_maps_timeout_url_error_and_non_json_http_error(monkeypatch):
    client = HttpClient(retry=RetryConfig(attempts=0))
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(ModelTimeoutError):
        list(client.post_json_stream("https://example.test"))

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    with pytest.raises(ProviderError, match="offline"):
        list(client.post_json_stream("https://example.test"))
