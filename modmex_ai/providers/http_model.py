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
    ) -> None:
        self.name = model
        self.model = model
        self.base_url = base_url.rstrip("/")
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

    # -- execution modes ------------------------------------------------------

    @property
    def _url(self) -> str:
        return f"{self.base_url}{self.path}"

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload = self._payload(request, stream=False)
        response = self.http_client.post_json(
            self._url,
            headers=self._headers(request),
            data=payload,
            timeout=_timeout(request),
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
        payload = self._payload(request, stream=False)
        response = await self.async_http_client.post_json(
            self._url,
            headers=self._headers(request),
            data=payload,
            timeout=_timeout(request),
        )
        return self._parse(
            response.body,
            headers=response.headers,
            status_code=response.status_code,
            payload=payload,
        )

    def stream(self, request: ModelRequest):
        payload = self._payload(request, stream=True)
        chunks = self.http_client.post_json_stream(
            self._url,
            headers=self._headers(request),
            data=payload,
            timeout=_timeout(request),
        )
        yield from self._events(parse_sse_lines(chunks), payload=payload)

    async def astream(self, request: ModelRequest):
        if self.async_http_client is not None:
            payload = self._payload(request, stream=True)
            chunks = self.async_http_client.post_json_stream(
                self._url,
                headers=self._headers(request),
                data=payload,
                timeout=_timeout(request),
            )
            async for event in self._aevents(parse_sse_lines_async(chunks), payload=payload):
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
