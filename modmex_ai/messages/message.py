from __future__ import annotations

from dataclasses import field
from typing import Any, Literal

from modmex import BaseModel


Role = Literal["system", "developer", "user", "assistant", "tool"]


class Message(BaseModel):
    role: Role
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
