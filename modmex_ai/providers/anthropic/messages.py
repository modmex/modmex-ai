from __future__ import annotations

import os
from typing import Any

from modmex_ai.http import AsyncHttpClient, HttpClient, RetryConfig
from modmex_ai.models import ModelRequest, ModelResponse
from modmex_ai.providers.anthropic.mapper import (
    beta_headers,
    binding_digest,
    from_messages_payload,
    messages_stream_events,
    messages_stream_events_async,
    to_messages_payload,
)
from modmex_ai.providers.anthropic.profile import ANTHROPIC_MESSAGES_PROFILE
from modmex_ai.providers.http_model import HttpModel


API_VERSION = "2026-10-21"
# Anthropic signals temporary overload with 529 in addition to the usual codes.
RETRY_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}
# Thinking models can take minutes per request; bounded_timeout still caps every
# request at the remaining Lambda/execution budget.
DEFAULT_TIMEOUT_SECONDS = 300.0


def default_retry() -> RetryConfig:
    return RetryConfig(status_codes=set(RETRY_STATUS_CODES))


class AnthropicMessagesModel(HttpModel):
    """Claude via ``POST /v1/messages`` using the built-in zero-dependency transport.

    ``api_key`` falls back to the ``ANTHROPIC_API_KEY`` environment variable.
    """

    path = "/messages"
    default_timeout = DEFAULT_TIMEOUT_SECONDS

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
        super().__init__(
            model,
            base_url=base_url,
            http_client=http_client,
            async_http_client=async_http_client,
        )
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        self.api_version = api_version
        self.default_headers = dict(default_headers or {})
        self.profile = ANTHROPIC_MESSAGES_PROFILE

    def default_retry(self) -> RetryConfig:
        return default_retry()

    def _headers(self, request: ModelRequest) -> dict[str, str]:
        headers = {"anthropic-version": self.api_version, **self.default_headers}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        betas = beta_headers(request)
        if betas:
            headers["anthropic-beta"] = ",".join(betas)
        return headers

    def _payload(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        return to_messages_payload(request, self.model, stream=stream)

    def _parse(self, body, *, headers, status_code, payload) -> ModelResponse:
        return from_messages_payload(
            body,
            headers=headers,
            status_code=status_code,
            model=self.model,
            binding=binding_digest(payload),
        )

    def _events(self, payloads, *, payload):
        return messages_stream_events(
            payloads,
            headers={},
            status_code=200,
            model=self.model,
            binding=binding_digest(payload),
        )

    def _aevents(self, payloads, *, payload):
        return messages_stream_events_async(
            payloads,
            headers={},
            status_code=200,
            model=self.model,
            binding=binding_digest(payload),
        )
