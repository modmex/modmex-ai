from __future__ import annotations

from typing import Any

from dataclasses import field
from modmex import BaseModel

from modmex_ai.messages import Message
from modmex_ai.models.provider_state import ProviderState
from modmex_ai.models.settings import ModelSettings
from modmex_ai.sessions import SessionItem


class ModelRequest(BaseModel):
    messages: list[Message | SessionItem]
    model: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    output_strict: bool = True
    settings: ModelSettings | None = None
    provider_state: ProviderState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
