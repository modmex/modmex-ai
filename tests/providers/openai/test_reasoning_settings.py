import pytest
from modmex_ai.models import ModelRequest, ModelSettings
from modmex_ai.providers.openai.mapper import to_chat_payload, to_responses_payload


def test_reasoning_settings_are_optional_and_mapped_per_endpoint():
    request = ModelRequest(messages=[], settings=ModelSettings(reasoning_effort="low"))
    assert to_responses_payload(request, "test")["reasoning"] == {"effort": "low"}
    assert to_chat_payload(request, "test")["reasoning_effort"] == "low"
    default = ModelRequest(messages=[])
    assert "reasoning" not in to_responses_payload(default, "test")
    assert "reasoning_effort" not in to_chat_payload(default, "test")
    with pytest.raises(ValueError):
        to_responses_payload(
            ModelRequest(
                messages=[], settings=ModelSettings(reasoning_effort="invalid")
            ),
            "test",
        )
