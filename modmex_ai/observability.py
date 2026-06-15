from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any

from modmex import BaseModel


class ObservabilityEvent(BaseModel):
    """One normalized lifecycle event emitted by an agent execution."""

    type: str
    name: str
    data: dict[str, Any] = field(default_factory=dict)


class ObservabilityObserver(ABC):
    """Receives lifecycle events for logs, metrics or tracing exporters."""

    @abstractmethod
    def on_event(self, event: ObservabilityEvent) -> None:
        """Handle one event without changing agent execution."""

