from __future__ import annotations

import os

from modmex_ai.http import (
    AsyncHttpClient,
    HttpClient,
    RetryConfig,
    parse_sse_lines,
    parse_sse_lines_async,
)
from modmex_ai.models import AsyncModelClient, ModelRequest, ModelResponse
from modmex_ai.providers.anthropic.mapper import (
    beta_headers,
    binding_digest,
    from_messages_payload,
    messages_stream_events,
    messages_stream_events_async,
    to_messages_payload,
)
from modmex_ai.providers.anthropic.profile import ANTHROPIC_MESSAGES_PROFILE


API_VERSION = "2026-10-21"
# Anthropic signals temporary overload with 529 in addition to the usual codes.
RETRY_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}
# Thinking models can take minutes per request; bounded_timeout still caps every
# request at the remaining Lambda/execution budget.
DEFAULT_TIMEOUT_SECONDS = 300.0


def default_retry() -> RetryConfig:
    return RetryConfig(status_codes=set(RETRY_STATUS_CODES))


class AnthropicMessagesModel(AsyncModelClient):
    """Claude via ``POST /v1/messages`` using the built-in zero-dependency transport.

    ``api_key`` falls back to the ``ANTHROPIC_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        http_client: HttpClient | None = None,
        async_http_client: AsyncHttpClient | None = None,
        api_version: str = API_VERSION,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.name = model
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.default_headers = dict(default_headers or {})
        self.http_client = http_client or HttpClient(retry=default_retry(), timeout=DEFAULT_TIMEOUT_SECONDS)
        self._owns_async_http_client = async_http_client is None
        self.async_http_client = async_http_client or (
            AsyncHttpClient(retry=default_retry(), timeout=DEFAULT_TIMEOUT_SECONDS)
            if http_client is None and AsyncHttpClient.is_available()
            else None
        )
        self.profile = ANTHROPIC_MESSAGES_PROFILE

    @property
    def _url(self) -> str:
        return f"{self.base_url}/messages"

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload = to_messages_payload(request, self.model)
        response = self.http_client.post_json(
            self._url,
            headers=self._headers(request),
            data=payload,
            timeout=_timeout(request),
        )
        return from_messages_payload(
            response.body,
            headers=response.headers,
            status_code=response.status_code,
            model=self.model,
            binding=binding_digest(payload),
        )

    async def aclose(self) -> None:
        """Release the internally created native async transport, if any."""
        if self._owns_async_http_client and self.async_http_client is not None:
            await self.async_http_client.close()

    async def acomplete(self, request: ModelRequest) -> ModelResponse:
        import asyncio
        if self.async_http_client is None:
            return await asyncio.to_thread(self.complete, request)
        payload = to_messages_payload(request, self.model)
        response = await self.async_http_client.post_json(
            self._url,
            headers=self._headers(request),
            data=payload,
            timeout=_timeout(request),
        )
        return from_messages_payload(
            response.body,
            headers=response.headers,
            status_code=response.status_code,
            model=self.model,
            binding=binding_digest(payload),
        )

    def stream(self, request: ModelRequest):
        payload = to_messages_payload(request, self.model, stream=True)
        chunks = self.http_client.post_json_stream(
            self._url,
            headers=self._headers(request),
            data=payload,
            timeout=_timeout(request),
        )
        yield from messages_stream_events(
            parse_sse_lines(chunks),
            headers={},
            status_code=200,
            model=self.model,
            binding=binding_digest(payload),
        )

    async def astream(self, request: ModelRequest):
        import asyncio
        if self.async_http_client is not None:
            payload = to_messages_payload(request, self.model, stream=True)
            chunks = self.async_http_client.post_json_stream(
                self._url,
                headers=self._headers(request),
                data=payload,
                timeout=_timeout(request),
            )
            async for event in messages_stream_events_async(
                parse_sse_lines_async(chunks),
                headers={},
                status_code=200,
                model=self.model,
                binding=binding_digest(payload),
            ):
                yield event
            return
        iterator = self.stream(request)
        while True:
            has_event, event = await asyncio.to_thread(_next_event, iterator)
            if not has_event:
                return
            yield event

    def _headers(self, request: ModelRequest) -> dict[str, str]:
        headers = {"anthropic-version": self.api_version, **self.default_headers}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        betas = beta_headers(request)
        if betas:
            headers["anthropic-beta"] = ",".join(betas)
        return headers


def _timeout(request: ModelRequest) -> float | None:
    return request.settings.timeout if request.settings else None


def _next_event(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None
