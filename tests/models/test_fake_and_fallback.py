import pytest

from modmex_ai.models import FakeModel, FallbackModel, ModelRequest, ModelResponse


def test_fake_model_records_requests():
    model = FakeModel(["ok"])
    response = model.complete(ModelRequest(messages=[]))

    assert response.output_text == "ok"
    assert len(model.requests) == 1


def test_fake_model_raises_when_responses_are_exhausted():
    model = FakeModel([])

    with pytest.raises(AssertionError):
        model.complete(ModelRequest(messages=[]))


def test_fake_model_stream_yields_complete_response():
    model = FakeModel(["ok"])

    responses = list(model.stream(ModelRequest(messages=[])))

    assert responses[0].output_text == "ok"


def test_fallback_model_uses_next_model_after_failure():
    class BrokenModel:
        name = "broken"

        def complete(self, _request):
            raise RuntimeError("boom")

    fallback = FallbackModel([BrokenModel(), FakeModel([ModelResponse(output_text="ok")])])

    assert fallback.complete(ModelRequest(messages=[])).output_text == "ok"


def test_fallback_model_requires_models():
    with pytest.raises(ValueError):
        FallbackModel([])


def test_fallback_model_raises_last_error_when_all_fail():
    class BrokenModel:
        name = "broken"

        def complete(self, _request):
            raise RuntimeError("boom")

    fallback = FallbackModel([BrokenModel()])

    with pytest.raises(RuntimeError, match="boom"):
        fallback.complete(ModelRequest(messages=[]))


def test_fallback_model_stream_uses_first_model():
    fallback = FallbackModel([FakeModel(["ok"])])

    responses = list(fallback.stream(ModelRequest(messages=[])))

    assert responses[0].output_text == "ok"


import asyncio

from modmex_ai.errors import ProviderError, RunTimeoutError
from modmex_ai.models import ModelProfile, ModelStreamEvent, ModelStreamEventType


class _Broken:
    name = "broken"
    profile = ModelProfile(supports_structured_output=True)

    def __init__(self, error=None, emit_first=False):
        self.error = error or ProviderError("down", status_code=529)
        self.emit_first = emit_first
        self.calls = 0

    def complete(self, _request):
        self.calls += 1
        raise self.error

    def stream(self, _request):
        self.calls += 1
        if self.emit_first:
            yield ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text_delta="partial")
        raise self.error


def _ok(text="ok"):
    model = FakeModel([ModelResponse(output_text=text)])
    model.profile = ModelProfile(supports_structured_output=True, output_mode="json_schema")
    return model


def test_fallback_stream_uses_next_model_when_first_fails_before_output():
    fallback = FallbackModel([_Broken(), _ok()])
    events = list(fallback.stream(ModelRequest(messages=[])))
    assert events[-1].output_text == "ok"


def test_fallback_stream_does_not_switch_after_output_was_emitted():
    second = _ok()
    fallback = FallbackModel([_Broken(emit_first=True), second])
    with pytest.raises(ProviderError):
        list(fallback.stream(ModelRequest(messages=[])))
    assert second.requests == []


def test_fallback_does_not_retry_when_budget_is_exhausted():
    second = _ok()
    fallback = FallbackModel([_Broken(error=RunTimeoutError("budget")), second])
    with pytest.raises(RunTimeoutError):
        fallback.complete(ModelRequest(messages=[]))
    with pytest.raises(RunTimeoutError):
        list(fallback.stream(ModelRequest(messages=[])))
    assert second.requests == []


def test_fallback_async_complete_and_stream():
    async def run():
        response = await FallbackModel([_Broken(), _ok("async")]).acomplete(ModelRequest(messages=[]))
        events = [event async for event in FallbackModel([_Broken(), _ok("astream")]).astream(ModelRequest(messages=[]))]
        return response, events

    response, events = asyncio.run(run())
    assert response.output_text == "async"
    assert events[-1].output_text == "astream"


def test_fallback_profile_is_intersection_of_models():
    assert FallbackModel([_ok(), _ok()]).profile.supports_structured_output is True
    weak = FakeModel([])
    weak.profile = ModelProfile(supports_structured_output=False)
    mixed = FallbackModel([_ok(), weak]).profile
    assert mixed.supports_structured_output is False
    assert mixed.output_mode == "prompted_json"


def test_agent_async_run_falls_back():
    from modmex_ai import Agent

    agent = Agent(name="a", instructions="x", model=FallbackModel([_Broken(), _ok("from backup")]))
    assert asyncio.run(agent.run_async("hi")).output == "from backup"
