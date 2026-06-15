from enum import Enum
from typing import Any, Literal

from modmex import BaseModel

from modmex_ai.schemas import schema_for_model, schema_for_type


class Payload(BaseModel):
    name: str
    count: int
    kind: Literal["a", "b"]
    optional_note: str | None = None


class Kind(Enum):
    A = "a"
    B = "b"


def test_schema_for_model_uses_modmex_models():
    schema = schema_for_model(Payload)

    assert schema["title"] == "Payload"
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["count"]["type"] == "integer"
    assert schema["properties"]["kind"]["enum"] == ["a", "b"]
    assert "optional_note" not in schema["required"]


def test_schema_for_optional_type_marks_nullable():
    schema = schema_for_type(str | None)

    assert schema["type"] == "string"
    assert schema["nullable"] is True


def test_schema_for_misc_types():
    assert schema_for_type(Any) == {}
    assert schema_for_type(str | int)["anyOf"] == [{"type": "string"}, {"type": "integer"}]
    assert schema_for_type(list[str]) == {"type": "array", "items": {"type": "string"}}
    assert schema_for_type(dict[str, str]) == {"type": "object"}
    assert schema_for_type(Kind) == {"enum": ["a", "b"], "type": "string"}
    assert schema_for_type(float) == {"type": "number"}
    assert schema_for_type(bool) == {"type": "boolean"}
    assert schema_for_type(list) == {"type": "array"}
    assert schema_for_type(dict) == {"type": "object"}


def test_schema_for_model_accepts_custom_name_and_skips_private_fields():
    class PrivatePayload(BaseModel):
        _private: str
        public: str

    schema = schema_for_model(PrivatePayload, name="Custom")

    assert schema["title"] == "Custom"
    assert "_private" not in schema["properties"]
