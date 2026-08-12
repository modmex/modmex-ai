from __future__ import annotations

import pytest

from modmex_ai.mcp.headers import extract_mcp_parameter_headers, validate_mcp_input_schema


def test_extracts_nested_string_boolean_and_safe_integer_headers() -> None:
    schema = {
        "type": "object",
        "properties": {
            "region": {"type": "string", "x-mcp-header": "Region"},
            "enabled": {"type": "boolean", "x-mcp-header": "Enabled"},
            "count": {"type": "integer", "x-mcp-header": "Count"},
            "filter": {"type": "object", "properties": {
                "tenant": {"type": "string", "x-mcp-header": "Tenant_Id"},
            }},
        },
    }
    assert extract_mcp_parameter_headers(
        input_schema=schema,
        arguments={"region": "us-west1", "enabled": True, "count": 7, "filter": {"tenant": "acme"}},
    ) == {
        "Mcp-Param-Region": "us-west1",
        "Mcp-Param-Enabled": "true",
        "Mcp-Param-Count": "7",
        "Mcp-Param-Tenant_Id": "acme",
    }


@pytest.mark.parametrize("value", ["number", "array", "null"])
def test_rejects_unsupported_header_property_types(value: str) -> None:
    with pytest.raises(ValueError):
        extract_mcp_parameter_headers(
            input_schema={"type": "object", "properties": {"value": {"type": value, "x-mcp-header": "Value"}}},
            arguments={},
        )


def test_rejects_duplicate_header_names_case_insensitively() -> None:
    with pytest.raises(ValueError):
        extract_mcp_parameter_headers(
            input_schema={"type": "object", "properties": {
                "first": {"type": "string", "x-mcp-header": "Region"},
                "second": {"type": "string", "x-mcp-header": "region"},
            }},
            arguments={},
        )


def test_encodes_unsafe_values_and_rejects_unsafe_integers() -> None:
    result = extract_mcp_parameter_headers(
        input_schema={"type": "object", "properties": {"value": {"type": "string", "x-mcp-header": "Value"}}},
        arguments={"value": " Hello, 世界\n"},
    )
    assert result["Mcp-Param-Value"].startswith("=?base64?")

    with pytest.raises(ValueError):
        extract_mcp_parameter_headers(
            input_schema={"type": "object", "properties": {"value": {"type": "integer", "x-mcp-header": "Value"}}},
            arguments={"value": 2**53},
        )


def test_rejects_invalid_header_name() -> None:
    with pytest.raises(ValueError):
        extract_mcp_parameter_headers(
            input_schema={"type": "object", "properties": {"value": {"type": "string", "x-mcp-header": "Bad Header"}}},
            arguments={},
        )


def test_accepts_rfc9110_tchar_names_and_safe_integer_boundaries() -> None:
    schema = {"type": "object", "properties": {
        "lower": {"type": "integer", "x-mcp-header": "X_!#$%&'*+.^`|~"},
        "upper": {"type": "integer", "x-mcp-header": "Upper"},
    }}
    result = extract_mcp_parameter_headers(
        input_schema=schema,
        arguments={"lower": -(2**53) + 1, "upper": 2**53 - 1},
    )
    assert result["Mcp-Param-X_!#$%&'*+.^`|~"] == str(-(2**53) + 1)
    assert result["Mcp-Param-Upper"] == str(2**53 - 1)


def test_rejects_invalid_nested_schema_and_runtime_values() -> None:
    with pytest.raises(ValueError):
        extract_mcp_parameter_headers(
            input_schema={"type": "object", "properties": {"nested": {"type": "object", "properties": []}}},
            arguments={"nested": {}},
        )
    with pytest.raises(ValueError):
        extract_mcp_parameter_headers(
            input_schema={"type": "object", "properties": {"enabled": {"type": "boolean", "x-mcp-header": "Enabled"}}},
            arguments={"enabled": "true"},
        )
    with pytest.raises(ValueError):
        extract_mcp_parameter_headers(
            input_schema={"type": "object", "properties": {"count": {"type": "integer", "x-mcp-header": "Count"}}},
            arguments={"count": True},
        )


def test_schema_validation_helper_returns_false_for_invalid_schema() -> None:
    assert validate_mcp_input_schema({"type": "object", "properties": {}})
    assert not validate_mcp_input_schema(None)
    assert not validate_mcp_input_schema({"type": "object", "properties": []})


@pytest.mark.parametrize("schema", [
    {"type": "object", "oneOf": [{"properties": {"region": {"type": "string", "x-mcp-header": "Region"}}}]},
    {"type": "object", "properties": {"items": {"type": "array", "items": {"properties": {"region": {"type": "string", "x-mcp-header": "Region"}}}}}},
    {"type": "object", "$defs": {"Region": {"type": "string", "x-mcp-header": "Region"}}},
])
def test_rejects_headers_outside_static_properties_paths(schema) -> None:
    assert not validate_mcp_input_schema(schema)
