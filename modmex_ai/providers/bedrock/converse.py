from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

from modmex_ai.http import AsyncHttpClient, HttpClient, RetryConfig
from modmex_ai.models import ModelProfile, ModelRequest, ModelResponse
from modmex_ai.providers.bedrock.mapper import (
    binding_digest,
    converse_stream_events,
    converse_stream_events_async,
    from_converse_payload,
    to_converse_payload,
)
from modmex_ai.providers.http_model import HttpModel


DEFAULT_TIMEOUT_SECONDS = 300.0


class BedrockConverseModel(HttpModel):
    """Any Bedrock text model through the Converse API (one adapter, many providers).

    Authentication, in order of preference:

    - default: ``AwsSigV4`` with botocore's credential chain (the Lambda
      execution role, a profile or SSO locally);
    - ``auth=AwsSigV4(...)`` to choose region, profile or credentials;
    - ``api_key=`` an Amazon Bedrock API key (bearer token), for software that
      cannot sign requests.

    ``model`` is a model id or inference profile id (e.g. ``us.<provider>...``),
    as shown in the model card. What a model supports (documents, prompt caching,
    structured output, reasoning) varies; check its card.
    """

    default_timeout = DEFAULT_TIMEOUT_SECONDS

    def __init__(
        self,
        model: str,
        *,
        region: str | None = None,
        auth: Any = None,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: HttpClient | None = None,
        async_http_client: AsyncHttpClient | None = None,
        s3_loader: Any = None,
        supports_structured_output: bool = True,
    ) -> None:
        if auth is not None and api_key is not None:
            raise ValueError("Pass either auth (SigV4) or api_key (Bedrock API key), not both")
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if base_url is None:
            if not self.region:
                raise ValueError("BedrockConverseModel needs a region (argument, AWS_REGION or AWS_DEFAULT_REGION)")
            base_url = f"https://bedrock-runtime.{self.region}.amazonaws.com"
        if auth is None and api_key is None:
            from modmex_ai.aws import AwsSigV4

            auth = AwsSigV4(region=self.region, service="bedrock")
        super().__init__(
            model,
            base_url=base_url,
            http_client=http_client,
            async_http_client=async_http_client,
            auth=auth,
        )
        self.api_key = api_key
        self._s3_loader = s3_loader
        self.profile = ModelProfile(
            supports_tools=True,
            supports_parallel_tool_calls=True,
            supports_structured_output=supports_structured_output,
            supports_system_messages=True,
            tool_schema_mode="json_schema",
            output_mode="json_schema" if supports_structured_output else "prompted_json",
        )

    def default_retry(self) -> RetryConfig:
        # 429 ThrottlingException and 503 ServiceUnavailable are the common
        # transient Bedrock errors; both are in the default set.
        return RetryConfig()

    @property
    def s3_loader(self) -> Any:
        if self._s3_loader is None:
            from modmex_ai.aws import S3ObjectLoader

            session = self.auth.session() if hasattr(self.auth, "session") else None
            self._s3_loader = S3ObjectLoader(session=session, region=self.region)
        return self._s3_loader

    def _endpoint(self, request: ModelRequest, *, stream: bool) -> str:
        model = quote(request.model or self.model, safe="")
        return f"{self.base_url}/model/{model}/{'converse-stream' if stream else 'converse'}"

    def _headers(self, request: ModelRequest) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        return to_converse_payload(request, self.model, load_s3=lambda uri: self.s3_loader.load(uri))

    def _parse(self, body, *, headers, status_code, payload) -> ModelResponse:
        return from_converse_payload(
            body, headers=headers, status_code=status_code, model=self.model, binding=binding_digest(payload)
        )

    def _decode_stream(self, chunks):
        from modmex_ai.aws.eventstream import decode_event_stream

        return decode_event_stream(chunks)

    def _adecode_stream(self, chunks):
        from modmex_ai.aws.eventstream import decode_event_stream_async

        return decode_event_stream_async(chunks)

    def _events(self, payloads, *, payload):
        return converse_stream_events(
            payloads, headers={}, status_code=200, model=self.model, binding=binding_digest(payload)
        )

    def _aevents(self, payloads, *, payload):
        return converse_stream_events_async(
            payloads, headers={}, status_code=200, model=self.model, binding=binding_digest(payload)
        )
