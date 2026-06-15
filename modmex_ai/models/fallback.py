from __future__ import annotations

from collections.abc import Iterable

from modmex_ai.models.base import ModelClient
from modmex_ai.models.profile import ModelProfile
from modmex_ai.models.request import ModelRequest
from modmex_ai.models.response import ModelResponse


class FallbackModel:
    def __init__(self, models: Iterable[ModelClient], *, name: str = "fallback") -> None:
        self.models = list(models)
        if not self.models:
            raise ValueError("FallbackModel requires at least one model")
        self.name = name
        self.profile = ModelProfile()

    def complete(self, request: ModelRequest) -> ModelResponse:
        last_error: Exception | None = None
        for model in self.models:
            try:
                return model.complete(request)
            except Exception as exc:  # pragma: no cover - intentionally broad wrapper
                last_error = exc
        assert last_error is not None
        raise last_error

    def stream(self, request: ModelRequest):
        return self.models[0].stream(request)

