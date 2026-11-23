from __future__ import annotations

from typing import Any

from modmex_ai.http import AsyncHttpClient, HttpClient
from modmex_ai.models import ModelRequest, ModelResponse
from modmex_ai.providers.http_model import HttpModel
from modmex_ai.providers.openai.mapper import (
    chat_stream_events,
    chat_stream_events_async,
    from_chat_payload,
    to_chat_payload,
)
from modmex_ai.providers.openai.profile import OPENAI_CHAT_PROFILE


class OpenAIChatModel(HttpModel):
    path = "/chat/completions"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        http_client: HttpClient | None = None,
        async_http_client: AsyncHttpClient | None = None,
        auth: Any = None,
    ) -> None:
        super().__init__(
            model,
            base_url=base_url,
            http_client=http_client,
            async_http_client=async_http_client,
            auth=auth,
        )
        self.api_key = api_key
        self.profile = OPENAI_CHAT_PROFILE

    def _headers(self, request: ModelRequest | None = None) -> dict[str, str]:
        return {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _payload(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        payload = to_chat_payload(request, self.model)
        if stream:
            payload["stream"] = True
        return payload

    def _parse(self, body, *, headers, status_code, payload) -> ModelResponse:
        return from_chat_payload(
            body, headers=headers, status_code=status_code, model=self.model
        )

    def _events(self, payloads, *, payload):
        return chat_stream_events(payloads, headers={}, status_code=200, model=self.model)

    def _aevents(self, payloads, *, payload):
        return chat_stream_events_async(
            payloads, headers={}, status_code=200, model=self.model
        )
