from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from modmex_ai.errors import RunTimeoutError
from modmex_ai.models.base import ModelClient
from modmex_ai.models.profile import ModelProfile
from modmex_ai.models.request import ModelRequest
from modmex_ai.models.response import ModelResponse

# Errors that mean "stop", not "try the next model": the execution budget is spent.
_NO_FALLBACK = (RunTimeoutError,)


class FallbackModel(ModelClient):
    """Try models in order until one succeeds.

    Streaming falls back only while the failing model has not emitted anything;
    once output has reached the caller, switching models would duplicate or
    contradict it, so the error is raised. The profile is the intersection of
    the models' profiles, because any of them may end up answering.
    """

    def __init__(self, models: Iterable[ModelClient], *, name: str = "fallback") -> None:
        self.models = list(models)
        if not self.models:
            raise ValueError("FallbackModel requires at least one model")
        self.name = name
        self.profile = _shared_profile(self.models)

    def complete(self, request: ModelRequest) -> ModelResponse:
        last_error: Exception | None = None
        for model in self.models:
            try:
                return model.complete(request)
            except _NO_FALLBACK:
                raise
            except Exception as exc:  # intentionally broad: any model failure falls back
                last_error = exc
        assert last_error is not None
        raise last_error

    async def acomplete(self, request: ModelRequest) -> ModelResponse:
        import asyncio

        last_error: Exception | None = None
        for model in self.models:
            try:
                acomplete = getattr(model, "acomplete", None)
                if callable(acomplete):
                    return await acomplete(request)
                return await asyncio.to_thread(model.complete, request)
            except _NO_FALLBACK:
                raise
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def stream(self, request: ModelRequest):
        last_error: Exception | None = None
        for model in self.models:
            emitted = False
            try:
                for event in model.stream(request):
                    emitted = True
                    yield event
                return
            except _NO_FALLBACK:
                raise
            except Exception as exc:
                if emitted:
                    raise
                last_error = exc
        assert last_error is not None
        raise last_error

    async def astream(self, request: ModelRequest):
        last_error: Exception | None = None
        for model in self.models:
            emitted = False
            try:
                async for event in _astream(model, request):
                    emitted = True
                    yield event
                return
            except _NO_FALLBACK:
                raise
            except Exception as exc:
                if emitted:
                    raise
                last_error = exc
        assert last_error is not None
        raise last_error


async def _astream(model: ModelClient, request: ModelRequest):
    astream = getattr(model, "astream", None)
    if callable(astream):
        async for event in astream(request):
            yield event
        return
    import asyncio

    iterator = iter(model.stream(request))
    sentinel: Any = object()
    while True:
        event = await asyncio.to_thread(next, iterator, sentinel)
        if event is sentinel:
            return
        yield event


def _shared_profile(models: list[ModelClient]) -> ModelProfile:
    profiles = [getattr(model, "profile", None) or ModelProfile() for model in models]
    structured = all(profile.supports_structured_output for profile in profiles)
    modes = {profile.tool_schema_mode for profile in profiles}
    return ModelProfile(
        supports_tools=all(profile.supports_tools for profile in profiles),
        supports_parallel_tool_calls=all(
            profile.supports_parallel_tool_calls for profile in profiles
        ),
        supports_structured_output=structured,
        supports_system_messages=all(
            profile.supports_system_messages for profile in profiles
        ),
        tool_schema_mode=modes.pop() if len(modes) == 1 else "json_schema",
        output_mode=(
            profiles[0].output_mode
            if structured and len({p.output_mode for p in profiles}) == 1
            else "prompted_json"
        ),
    )
