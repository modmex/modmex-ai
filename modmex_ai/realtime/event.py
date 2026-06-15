from __future__ import annotations

from typing import Any

from modmex import BaseModel


class RealtimeEvent(BaseModel):
    """A provider event received by a realtime session."""

    type: str
    data: dict[str, Any]
    raw: dict[str, Any]
