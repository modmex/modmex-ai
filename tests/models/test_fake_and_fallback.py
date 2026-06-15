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
