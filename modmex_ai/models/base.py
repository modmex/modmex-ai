from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from modmex_ai.models.profile import ModelProfile
from modmex_ai.models.request import ModelRequest
from modmex_ai.models.response import ModelResponse
from modmex_ai.models.stream import ModelStreamEvent


class ModelClient(Protocol):
    name: str
    profile: ModelProfile

    def complete(self, request: ModelRequest) -> ModelResponse:
        ...

    def stream(self, request: ModelRequest):
        raise NotImplementedError("Streaming is not implemented by this model.")


class AsyncModelClient(ModelClient, Protocol):
    """A model that can complete requests without blocking an event loop."""

    async def acomplete(self, request: ModelRequest) -> ModelResponse:
        ...

    def astream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent | ModelResponse]:
        ...
