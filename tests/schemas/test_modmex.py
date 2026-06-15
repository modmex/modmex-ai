import pytest
from modmex import BaseModel

from modmex_ai.errors import OutputValidationError, ToolValidationError
from modmex_ai.schemas import dumps, serialize, validate_model, validate_tool_args


class Payload(BaseModel):
    name: str


def test_validate_model_accepts_json_string():
    result = validate_model('{"name":"Ada"}', Payload)

    assert isinstance(result, Payload)
    assert result.name == "Ada"


def test_validate_model_wraps_invalid_values():
    with pytest.raises(OutputValidationError):
        validate_model("{bad json", Payload)


def test_serialize_and_dumps_use_modmex_dump_methods():
    payload = Payload(name="Ada")

    assert serialize(payload) == {"name": "Ada"}
    assert dumps(payload) == '{"name":"Ada"}'


def test_validate_model_returns_existing_instance():
    payload = Payload(name="Ada")

    assert validate_model(payload, Payload) is payload


def test_validate_model_handles_non_modmex_types():
    assert validate_model("1", int) == 1


def test_validate_tool_args_uses_tool_validation_error():
    with pytest.raises(ToolValidationError):
        validate_tool_args("{bad json", Payload)


def test_serialize_uses_to_dict_when_present():
    class ToDictValue:
        def to_dict(self):
            return {"ok": True}

    assert serialize(ToDictValue()) == {"ok": True}
