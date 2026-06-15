from __future__ import annotations

import asyncio

from modmex_ai.http import AsyncHttpClient, HttpClient, parse_sse_lines
from modmex_ai.models import ModelRequest, ModelResponse
from modmex_ai.providers.openai.mapper import from_responses_payload, responses_stream_events, to_responses_payload
from modmex_ai.providers.openai.profile import OPENAI_RESPONSES_PROFILE


class OpenAIResponsesModel:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        http_client: HttpClient | None = None,
        async_http_client: AsyncHttpClient | None = None,
        organization: str | None = None,
        project: str | None = None,
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
        self.organization = organization
        self.project = project
        self.profile = OPENAI_RESPONSES_PROFILE

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload = to_responses_payload(request, self.model)
        response = self.http_client.post_json(
            f"{self.base_url}/responses",
            headers=self._headers(),
            data=payload,
            timeout=request.settings.timeout if request.settings else None,
        )
        return from_responses_payload(
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
        payload = to_responses_payload(request, self.model)
        response = await self.async_http_client.post_json(
            f"{self.base_url}/responses",
            headers=self._headers(),
            data=payload,
            timeout=request.settings.timeout if request.settings else None,
        )
        return from_responses_payload(
            response.body,
            headers=response.headers,
            status_code=response.status_code,
            model=self.model,
        )

    def stream(self, request: ModelRequest):
        payload = to_responses_payload(request, self.model)
        payload["stream"] = True
        chunks = self.http_client.post_json_stream(
            f"{self.base_url}/responses",
            headers=self._headers(),
            data=payload,
            timeout=request.settings.timeout if request.settings else None,
        )
        yield from responses_stream_events(
            parse_sse_lines(chunks),
            headers={},
            status_code=200,
            model=self.model,
        )

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project
        return headers
