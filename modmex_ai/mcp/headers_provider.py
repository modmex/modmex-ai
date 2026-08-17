from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Protocol


@dataclass(frozen=True)
class MCPRequest:
    url: str
    method: str
    headers: dict[str, str]
    payload: dict[str, Any]


class MCPHeadersProvider(Protocol):
    def __call__(self, request: MCPRequest) -> dict[str, str] | Awaitable[dict[str, str]]:
        ...
