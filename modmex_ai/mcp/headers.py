"""MCP parameter-header extraction and validation."""

from __future__ import annotations

import re
import base64
from typing import Any


# RFC 9110 tchar.
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_SAFE_INTEGER_MIN = -(2**53) + 1
_SAFE_INTEGER_MAX = (2**53) - 1
_SAFE_HEADER_VALUE = frozenset("!#$%&'*+-.^_`|~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def extract_mcp_parameter_headers(*, input_schema: dict[str, Any], arguments: dict[str, Any]) -> dict[str, str]:
    """Extract ``x-mcp-header`` properties from nested tool arguments."""
    headers: dict[str, str] = {}
    header_names: set[str] = set()

    def visit(schema: dict[str, Any], value: Any) -> None:
        if not isinstance(schema, dict):
            raise ValueError("inputSchema must contain objects")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("inputSchema.properties must be an object")
        value = value if isinstance(value, dict) else {}
        for name, property_schema in properties.items():
            if not isinstance(property_schema, dict):
                raise ValueError(f"Invalid schema for property {name}")
            header_name = property_schema.get("x-mcp-header")
            if header_name is not None:
                if (
                    not isinstance(header_name, str)
                    or not header_name
                    or not _HEADER_NAME.fullmatch(header_name)
                    or property_schema.get("type") not in {"string", "boolean", "integer"}
                ):
                    raise ValueError(f"Invalid x-mcp-header for property {name}")
                normalized_header_name = header_name.lower()
                if normalized_header_name in header_names:
                    raise ValueError(f"Duplicate x-mcp-header: {header_name}")
                header_names.add(normalized_header_name)
                if name in value:
                    item = value[name]
                    if property_schema.get("type") == "integer":
                        if isinstance(item, bool) or not isinstance(item, int) or not (_SAFE_INTEGER_MIN <= item <= _SAFE_INTEGER_MAX):
                            raise ValueError(f"Invalid integer value for x-mcp-header: {name}")
                    elif property_schema.get("type") == "boolean" and not isinstance(item, bool):
                        raise ValueError(f"Invalid boolean value for x-mcp-header: {name}")
                    elif property_schema.get("type") == "string" and not isinstance(item, str):
                        raise ValueError(f"Invalid string value for x-mcp-header: {name}")
                    serialized = str(item).lower() if isinstance(item, bool) else str(item)
                    headers[f"Mcp-Param-{header_name}"] = encode_mcp_header_value(serialized)
            if property_schema.get("type") == "object":
                visit(property_schema, value.get(name))

    visit(input_schema, arguments)
    return headers


def validate_mcp_input_schema(schema: Any) -> bool:
    try:
        extract_mcp_parameter_headers(input_schema=schema, arguments={})
    except (TypeError, ValueError):
        return False
    return isinstance(schema, dict)


def encode_mcp_header_value(value: str, *, force: bool = False) -> str:
    """Encode values that cannot safely travel as an HTTP header value."""
    if not force and value and all(char in _SAFE_HEADER_VALUE for char in value) and not value.startswith("=?base64?"):
        return value
    return "=?base64?" + base64.b64encode(value.encode()).decode() + "?="
