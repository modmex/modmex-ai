from __future__ import annotations

from dataclasses import field
from typing import Any

from modmex import BaseModel


class ProviderState(BaseModel):
    provider: str | None = None
    conversation_id: str | None = None
    previous_response_id: str | None = None
    values: dict[str, Any] = field(default_factory=dict)

    @property
    def has_remote_state(self) -> bool:
        return bool(self.conversation_id or self.previous_response_id or self.values)
