import asyncio
import io
import urllib.error
from unittest.mock import Mock, patch
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone

import pytest

from modmex_ai.budget import execution_budget
from modmex_ai.errors import ProviderError
from modmex_ai.http.retries import RetryConfig
from modmex_ai.http.client import HttpClient
from modmex_ai.http.async_client import AsyncHttpClient


def test_retry_after_quota_jitter_and_budget():
    events = []
    retry = RetryConfig(on_retry=events.append, jitter_seconds=0.1)
    error = ProviderError("limited", status_code=429, headers={"Retry-After": "2"})
    assert 2 <= retry.delay(error, 0) <= 2.1
    assert events[0]["attempt"] == 1
    with execution_budget(0.5):
        assert retry.delay(error, 0) is None
    assert RetryConfig(status_codes=set()).delay(error, 0) is None
    quota = ProviderError(
        "quota",
        status_code=429,
        response_body={"error": {"code": "insufficient_quota"}},
    )
    assert retry.delay(quota, 0) is None
    date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))
    assert (
        retry.delay(
            ProviderError("", status_code=503, headers={"retry-after": date}), 0
        )
        > 28
    )


def test_sync_http_retry_and_timeout_use_remaining_budget():
    client = HttpClient(retry=RetryConfig(backoff_seconds=0))
    client._send = Mock(side_effect=[ProviderError("busy", status_code=503), "ok"])
    with execution_budget(1):
        assert client.post_json("https://example.test", timeout=30) == "ok"
    assert client._send.call_count == 2
    assert all(call.kwargs["timeout"] <= 1 for call in client._send.call_args_list)


def test_sync_stream_never_replays_after_partial_output():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            if not getattr(self, "read_once", False):
                self.read_once = True
                return b"already emitted"
            raise urllib.error.HTTPError(
                "https://example.test", 503, "busy", {}, io.BytesIO(b"{}")
            )

    with patch("urllib.request.urlopen", return_value=Response()) as send:
        stream = HttpClient().post_json_stream("https://example.test")
        assert next(stream) == b"already emitted"
        with pytest.raises(ProviderError):
            next(stream)
        assert send.call_count == 1


def test_async_http_honors_retry_after_and_remaining_budget():
    httpx = pytest.importorskip("httpx")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            503, headers={"Retry-After": "10"}, json={"error": "busy"}
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ) as transport:
            client = AsyncHttpClient(client=transport)
            with execution_budget(0.5), pytest.raises(ProviderError):
                await client.post_json("https://example.test")

    asyncio.run(run())
    assert len(requests) == 1
