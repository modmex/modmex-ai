"""Shared transport for JSON-over-HTTP model adapters.

A provider adapter only describes what is specific to it: the endpoint, headers,
how a request becomes a payload and how responses/SSE events are parsed. This
base owns the four execution modes (sync, async, stream, async stream), the
optional native async transport and its lifecycle, and the thread fallback when
``httpx`` is not installed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from modmex_ai.http import (
    AsyncHttpClient,
    HttpClient,
    RetryConfig,
    parse_sse_lines,
    parse_sse_lines_async,
)
from modmex_ai.models import AsyncModelClient, ModelRequest, ModelResponse


class HttpModel(AsyncModelClient):
    #: Path appended to ``base_url`` (e.g. "/messages").
    path: str = ""
    #: Defaults for transports this adapter creates itself.
    default_timeout: float = 30.0

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        http_client: HttpClient | None = None,
        async_http_client: AsyncHttpClient | None = None,
        auth: Any = None,
    ) -> None:
        self.name = model
        self.model = model
        self.base_url = base_url.rstrip("/")
        #: Optional request signer with ``sign(method, url, headers, body) ->
        #: headers`` (e.g. ``modmex_ai.aws.AwsSigV4``). When set, the body is
        #: encoded once and that exact byte string is signed and sent.
        self.auth = auth
        self.http_client = http_client or HttpClient(
            retry=self.default_retry(), timeout=self.default_timeout
        )
        self._owns_async_http_client = async_http_client is None
        self.async_http_client = async_http_client or (
            AsyncHttpClient(retry=self.default_retry(), timeout=self.default_timeout)
            if http_client is None and AsyncHttpClient.is_available()
            else None
        )

    # -- provider hooks -------------------------------------------------------

    def default_retry(self) -> RetryConfig | None:
        """Retry policy for transports created by this adapter (None = library default)."""
        return None

    def _headers(self, request: ModelRequest) -> dict[str, str]:
        raise NotImplementedError

    def _payload(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        raise NotImplementedError

    def _parse(
        self, body: Any, *, headers: dict[str, str], status_code: int, payload: dict[str, Any]
    ) -> ModelResponse:
        raise NotImplementedError

    def _events(self, payloads: Iterator[dict[str, Any]], *, payload: dict[str, Any]):
        raise NotImplementedError

    def _aevents(self, payloads: AsyncIterator[dict[str, Any]], *, payload: dict[str, Any]):
        raise NotImplementedError

    def _endpoint(self, request: ModelRequest, *, stream: bool) -> str:
        """URL for this request; most adapters post everything to one path."""
        return self._url

    def _decode_stream(self, chunks: Iterator[bytes]) -> Iterator[dict[str, Any]]:
        """Turn the raw streaming body into event dicts (SSE by default)."""
        return parse_sse_lines(chunks)

    def _adecode_stream(self, chunks: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
        return parse_sse_lines_async(chunks)

    def _prepare(
        self, request: ModelRequest, *, stream: bool
    ) -> tuple[str, dict[str, str], dict[str, Any], dict[str, Any] | bytes]:
        """Build (url, headers, payload, wire body), signing it when ``auth`` is set."""
        payload = self._payload(request, stream=stream)
        url = self._endpoint(request, stream=stream)
        headers = self._headers(request)
        if self.auth is None:
            return url, headers, payload, payload
        from modmex_ai.http.client import encode_json_body

        body = encode_json_body(payload)
        signed = self.auth.sign(
            "POST", url, {"content-type": "application/json", **headers}, body
        )
        return url, signed, payload, body

    # -- execution modes ------------------------------------------------------

    @property
    def _url(self) -> str:
        return f"{self.base_url}{self.path}"

    def complete(self, request: ModelRequest) -> ModelResponse:
        url, headers, payload, body = self._prepare(request, stream=False)
        response = self.http_client.post_json(
            url, headers=headers, data=body, timeout=_timeout(request)
        )
        return self._parse(
            response.body,
            headers=response.headers,
            status_code=response.status_code,
            payload=payload,
        )

    async def acomplete(self, request: ModelRequest) -> ModelResponse:
        if self.async_http_client is None:
            import asyncio

            return await asyncio.to_thread(self.complete, request)
        url, headers, payload, body = self._prepare(request, stream=False)
        response = await self.async_http_client.post_json(
            url, headers=headers, data=body, timeout=_timeout(request)
        )
        return self._parse(
            response.body,
            headers=response.headers,
            status_code=response.status_code,
            payload=payload,
        )

    def stream(self, request: ModelRequest):
        url, headers, payload, body = self._prepare(request, stream=True)
        chunks = self.http_client.post_json_stream(
            url, headers=headers, data=body, timeout=_timeout(request)
        )
        yield from self._events(self._decode_stream(chunks), payload=payload)

    async def astream(self, request: ModelRequest):
        if self.async_http_client is not None:
            url, headers, payload, body = self._prepare(request, stream=True)
            chunks = self.async_http_client.post_json_stream(
                url, headers=headers, data=body, timeout=_timeout(request)
            )
            async for event in self._aevents(self._adecode_stream(chunks), payload=payload):
                yield event
            return
        import asyncio

        iterator = self.stream(request)
        sentinel = object()
        while True:
            event = await asyncio.to_thread(next, iterator, sentinel)
            if event is sentinel:
                return
            yield event

    async def aclose(self) -> None:
        """Release the internally created native async transport, if any."""
        if self._owns_async_http_client and self.async_http_client is not None:
            await self.async_http_client.close()


def _timeout(request: ModelRequest) -> float | None:
    return request.settings.timeout if request.settings else None
