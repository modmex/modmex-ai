"""MCP parameter-header extraction and validation."""

from __future__ import annotations

import re
from typing import Any


_HEADER_NAME = re.compile(r"^[A-Za-z0-9-]+$")


def extract_mcp_parameter_headers(*, input_schema: dict[str, Any], arguments: dict[str, Any]) -> dict[str, str]:
    """Extract ``x-mcp-header`` properties from nested tool arguments."""
    headers: dict[str, str] = {}

    def visit(schema: dict[str, Any], value: Any) -> None:
        if not isinstance(schema, dict):
            raise ValueError("inputSchema must contain objects")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("inputSchema.properties must be an object")
        if not isinstance(value, dict):
            return
        for name, property_schema in properties.items():
            if not isinstance(property_schema, dict):
                raise ValueError(f"Invalid schema for property {name}")
            header_name = property_schema.get("x-mcp-header")
            if header_name is not None:
                if (
                    not isinstance(header_name, str)
                    or not header_name
                    or not _HEADER_NAME.fullmatch(header_name)
                    or property_schema.get("type") not in {"string", "boolean"}
                ):
                    raise ValueError(f"Invalid x-mcp-header for property {name}")
                if name in value:
                    item = value[name]
                    headers[f"Mcp-Param-{header_name}"] = (
                        str(item).lower() if isinstance(item, bool) else str(item)
                    )
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
