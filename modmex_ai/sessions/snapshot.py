from __future__ import annotations

from dataclasses import field

from modmex import BaseModel

from modmex_ai.sessions.item import SessionItem
from modmex_ai.sessions.memory import InMemorySession
from modmex_ai.sessions.session import Session


class SessionSnapshot(BaseModel):
    """Serializable history for hosts that do not provide durable Session storage."""

    session_id: str
    items: list[SessionItem] = field(default_factory=list)
    schema_version: int = 1
    revision: int = 0
    summary: str | None = None
    continuation: dict | None = None
    provider_state: dict | None = None

    @classmethod
    def from_session(cls, session: Session) -> "SessionSnapshot":
        return cls(session_id=session.id, items=session.get_items())

    def to_memory_session(self) -> InMemorySession:
        return InMemorySession(session_id=self.session_id, items=self.items)

    def compact(self, *, max_items: int, summary: str | None = None) -> "SessionSnapshot":
        """Return a portable snapshot retaining only recent history and a summary."""
        if max_items < 0:
            raise ValueError("max_items must be non-negative")
        return SessionSnapshot(**{
            **self.model_dump(),
            "items": self.items[-max_items:] if max_items else [],
            "summary": summary if summary is not None else self.summary,
        })
