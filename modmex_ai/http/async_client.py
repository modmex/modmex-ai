from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import AsyncIterator
from typing import Any

from modmex_ai.errors import ModelTimeoutError, ProviderError, RateLimitError
from modmex_ai.http.client import HttpResponse
from modmex_ai.http.retries import RetryConfig


class AsyncHttpClient:
    """Reusable optional ``httpx`` transport with explicit resource ownership."""

    def __init__(
        self,
        *,
        retry: RetryConfig | None = None,
        timeout: float = 30.0,
        client: Any | None = None,
        own_client: bool | None = None,
    ) -> None:
        self.retry = retry or RetryConfig()
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None if own_client is None else own_client
        self._closed = False

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("httpx") is not None

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def open(self) -> "AsyncHttpClient":
        """Create the pooled transport on first use."""
        await self._get_client()
        return self

    async def close(self) -> None:
        """Close a transport owned by this instance and reject further requests."""
        if self._closed:
            return
        self._closed = True
        if self._client is not None and self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncHttpClient":
        return await self.open()

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        async for response in self._post(
            url,
            headers=headers,
            data=data,
            timeout=timeout,
            stream=False,
        ):
            return response
        raise AssertionError("Async HTTP request did not return a response")

    async def post_json_stream(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
        chunk_size: int = 8_192,
        timeout: float | None = None,
    ) -> AsyncIterator[bytes]:
        async for response in self._post(
            url,
            headers=headers,
            data=data,
            timeout=timeout,
            stream=True,
        ):
            async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                if chunk:
                    yield chunk

    async def _post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None,
        data: dict[str, Any] | None,
        timeout: float | None,
        stream: bool,
    ) -> AsyncIterator[Any]:
        client = await self._get_client()
        for attempt in range(self.retry.attempts + 1):
            try:
                if stream:
                    async with client.stream(
                        "POST",
                        url,
                        headers=headers,
                        json=data,
                        timeout=timeout or self.timeout,
                    ) as response:
                        response.raise_for_status()
                        yield response
                else:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=data,
                        timeout=timeout or self.timeout,
                    )
                    response.raise_for_status()
                    yield HttpResponse(
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        body=response.json() if response.content else {},
                    )
                return
            except asyncio.CancelledError:
                raise
            except Exception as error:
                provider_error = _provider_error(error)
                if (
                    provider_error.status_code not in self.retry.status_codes
                    or attempt >= self.retry.attempts
                ):
                    raise provider_error
                await asyncio.sleep(self.retry.backoff_seconds * (2**attempt))

    async def _get_client(self) -> Any:
        if self._closed:
            raise RuntimeError("Async HTTP client is closed")
        if self._client is None:
            self._client = _httpx().AsyncClient(timeout=self.timeout)
        return self._client


def _httpx():
    try:
        import httpx
    except ImportError as error:
        raise ProviderError(
            "Install modmex-ai[async] to use the native async HTTP transport"
        ) from error
    return httpx


def _provider_error(error: Exception) -> ProviderError:
    httpx = _httpx()
    if isinstance(error, httpx.TimeoutException):
        return ModelTimeoutError("Model request timed out")
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        try:
            body = response.json()
        except ValueError:
            body = response.text
        error_cls = RateLimitError if response.status_code == 429 else ProviderError
        return error_cls(
            f"Provider returned HTTP {response.status_code}",
            status_code=response.status_code,
            request_id=response.headers.get("x-request-id") or response.headers.get("request-id"),
            response_body=body,
            headers=dict(response.headers),
        )
    return ProviderError(str(error))
