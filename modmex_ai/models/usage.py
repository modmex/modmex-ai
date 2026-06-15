from __future__ import annotations

from dataclasses import field
from typing import Any

from modmex import BaseModel


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def add(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.reasoning_output_tokens += other.reasoning_output_tokens
        self._add_details(other)
        return self

    def copy(self) -> "Usage":
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            cached_input_tokens=self.cached_input_tokens,
            reasoning_output_tokens=self.reasoning_output_tokens,
            details=self.details.copy(),
        )

    def _add_details(self, other: "Usage") -> None:
        if not other.details:
            return
        raw_items = self.details.setdefault("raw_items", [])
        raw = other.details.get("raw", other.details)
        raw_items.append(raw)
