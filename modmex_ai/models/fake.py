from __future__ import annotations

from collections.abc import Iterable

from modmex_ai.models.profile import ModelProfile
from modmex_ai.models.request import ModelRequest
from modmex_ai.models.response import ModelResponse


class FakeModel:
    def __init__(
        self,
        responses: Iterable[ModelResponse | str],
        *,
        name: str = "fake",
        profile: ModelProfile | None = None,
    ) -> None:
        self.name = name
        self.profile = profile or ModelProfile(
            supports_tools=True,
            supports_structured_output=True,
            supports_parallel_tool_calls=True,
            output_mode="json_schema",
        )
        self.requests: list[ModelRequest] = []
        self._responses = list(responses)

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("FakeModel has no responses left")
        response = self._responses.pop(0)
        if isinstance(response, str):
            return ModelResponse(output_text=response, provider="fake", model=self.name)
        return response

    def stream(self, request: ModelRequest):
        # Backwards-compatible complete-response stream; Agent normalizes it.
        yield self.complete(request)
