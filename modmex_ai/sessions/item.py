from __future__ import annotations

from typing import Any, Literal

from modmex import BaseModel

from modmex_ai.messages import Message, Role
from modmex_ai.schemas import dumps


SessionItemType = Literal[
    "message",
    "function_call",
    "function_call_output",
    "handoff_call",
    "handoff_call_output",
]


class SessionItem(BaseModel):
    type: SessionItemType = "message"
    role: Role | None = None
    content: str | list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    arguments: dict[str, Any] | str | None = None
    output: Any = None

    @classmethod
    def from_message(cls, message: Message) -> "SessionItem":
        return cls(
            type="message",
            role=message.role,
            content=message.content,
            name=message.name,
            tool_call_id=message.tool_call_id,
        )

    def to_message(self) -> Message:
        if self.type != "message":
            raise ValueError("Only message session items can be converted to Message")
        if self.role is None or self.content is None:
            raise ValueError("Message session items require role and content")
        return Message(
            role=self.role,
            content=_as_content(self.content),
        )

    def to_input(self) -> dict[str, Any]:
        if self.type == "function_call":
            return {
                "type": self.type,
                "tool_call_id": self.tool_call_id,
                "name": self.name,
                "arguments": _as_arguments(self.arguments),
            }
        if self.type == "function_call_output":
            return {
                "type": self.type,
                "tool_call_id": self.tool_call_id,
                "output": _as_string(self.output),
            }
        if self.type == "handoff_call":
            return {
                "type": self.type,
                "tool_call_id": self.tool_call_id,
                "name": self.name,
                "arguments": _as_arguments(self.arguments),
            }
        if self.type == "handoff_call_output":
            return {
                "type": self.type,
                "tool_call_id": self.tool_call_id,
                "output": _as_string(self.output),
            }
        return {
            "role": self.role,
            "content": _as_content(self.content),
        }


def _as_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return dumps(value)


def _as_arguments(value: Any) -> str:
    if value is None:
        return "{}"
    return _as_string(value)


def _as_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return dumps(value)
