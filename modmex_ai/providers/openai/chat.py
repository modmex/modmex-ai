from __future__ import annotations

import asyncio

from modmex_ai.http import AsyncHttpClient, HttpClient, parse_sse_lines
from modmex_ai.models import ModelRequest, ModelResponse
from modmex_ai.providers.openai.mapper import chat_stream_events, from_chat_payload, to_chat_payload
from modmex_ai.providers.openai.profile import OPENAI_CHAT_PROFILE


class OpenAIChatModel:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        http_client: HttpClient | None = None,
        async_http_client: AsyncHttpClient | None = None,
    ) -> None:
        self.name = model
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.http_client = http_client or HttpClient()
        self._owns_async_http_client = async_http_client is None
        self.async_http_client = async_http_client or (
            AsyncHttpClient()
            if http_client is None and AsyncHttpClient.is_available()
            else None
        )
        self.profile = OPENAI_CHAT_PROFILE

    def complete(self, request: ModelRequest) -> ModelResponse:
        response = self.http_client.post_json(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            data=to_chat_payload(request, self.model),
            timeout=request.settings.timeout if request.settings else None,
        )
        return from_chat_payload(
            response.body,
            headers=response.headers,
            status_code=response.status_code,
            model=self.model,
        )

    async def aclose(self) -> None:
        """Release the internally created native async transport, if any."""
        if self._owns_async_http_client and self.async_http_client is not None:
            await self.async_http_client.close()

    async def acomplete(self, request: ModelRequest) -> ModelResponse:
        if self.async_http_client is None:
            return await asyncio.to_thread(self.complete, request)
        response = await self.async_http_client.post_json(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            data=to_chat_payload(request, self.model),
            timeout=request.settings.timeout if request.settings else None,
        )
        return from_chat_payload(
            response.body,
            headers=response.headers,
            status_code=response.status_code,
            model=self.model,
        )

    def stream(self, request: ModelRequest):
        payload = to_chat_payload(request, self.model)
        payload["stream"] = True
        chunks = self.http_client.post_json_stream(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            data=payload,
            timeout=request.settings.timeout if request.settings else None,
        )
        yield from chat_stream_events(
            parse_sse_lines(chunks),
            headers={},
            status_code=200,
            model=self.model,
        )

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
