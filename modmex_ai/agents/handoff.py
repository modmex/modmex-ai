from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from modmex_ai.models import ToolCall
from modmex_ai.schemas import schema_for_model, validate_model
from modmex_ai.sessions import SessionItem


RECOMMENDED_PROMPT_PREFIX = (
    "# System context\n"
    "You are part of a multi-agent system called modmex-ai, designed to make "
    "agent coordination and execution easy. The system uses two primary "
    "abstractions: Agents and Handoffs. An agent includes instructions and "
    "tools, and can hand off a conversation to another agent when appropriate. "
    "Handoffs are achieved by calling a handoff function, generally named "
    "`transfer_to_<agent_name>`. Transfers between agents are handled "
    "seamlessly in the background; do not mention or draw attention to these "
    "transfers in your conversation with the user."
)


def prompt_with_handoff_instructions(prompt: str) -> str:
    return f"{RECOMMENDED_PROMPT_PREFIX}\n\n{prompt}"


class HandoffInputData:
    def __init__(self, *, history: list[SessionItem], input: Any) -> None:
        self.history = history
        self.input = input


HandoffCallback = Callable[[Any, Any], None]
HandoffInputFilter = Callable[[HandoffInputData], HandoffInputData | list[SessionItem]]


class Handoff:
    def __init__(
        self,
        agent: str,
        *,
        name: str | None = None,
        description: str | None = None,
        input_type: type[Any] | None = None,
        on_handoff: HandoffCallback | None = None,
        input_filter: HandoffInputFilter | None = None,
    ) -> None:
        if input_type is not None and on_handoff is None:
            raise ValueError("Handoff input_type requires an on_handoff callback")
        self.agent = agent
        self.name = name or self.default_tool_name(agent)
        self.description = description or self.default_tool_description(agent)
        self.input_type = input_type
        self.on_handoff = on_handoff
        self.input_filter = input_filter

    @staticmethod
    def default_tool_name(agent: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", agent).strip("_").lower()
        return f"transfer_to_{normalized}"

    @staticmethod
    def default_tool_description(agent: str) -> str:
        return f"Handoff to the {agent} agent to handle the request."

    def schema(self) -> dict[str, Any]:
        parameters = (
            schema_for_model(self.input_type)
            if self.input_type
            else {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
        )
        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }

    def parse_input(self, value: Any) -> Any:
        if self.input_type is None:
            return value
        return validate_model(value, self.input_type)

    def invoke(self, context: Any, value: Any) -> Any:
        parsed = self.parse_input(value)
        if self.on_handoff is not None:
            self.on_handoff(context, parsed)
        return parsed

    def filter_history(self, history: list[SessionItem], input: Any) -> list[SessionItem]:
        if self.input_filter is None:
            return history
        filtered = self.input_filter(HandoffInputData(history=history, input=input))
        if isinstance(filtered, HandoffInputData):
            return filtered.history
        return filtered


def normalize_handoffs(values: list[str | Handoff] | None) -> list[Handoff]:
    return [
        value if isinstance(value, Handoff) else Handoff(value)
        for value in (values or [])
    ]


def find_handoff(tool_call: ToolCall, handoffs: list[Handoff]) -> Handoff | None:
    for handoff in handoffs:
        if tool_call.name == handoff.name:
            return handoff
    return None
