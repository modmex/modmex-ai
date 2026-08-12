"""Typed errors returned by MCP clients."""

from __future__ import annotations

from typing import Any


class MCPError(Exception):
    def __init__(self, code: int, message: str, *, data: Any = None, http_status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data
        self.http_status = http_status


class MCPInputRequired(MCPError):
    def __init__(self, input_requests: list[Any], request_state: str | None = None, *, data: Any = None) -> None:
        super().__init__(-32001, "MCP input required", data=data, http_status=None)
        self.input_requests = input_requests
        self.request_state = request_state
