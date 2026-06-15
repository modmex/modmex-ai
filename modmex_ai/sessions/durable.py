from __future__ import annotations

from abc import ABC, abstractmethod

from modmex_ai.sessions.snapshot import SessionSnapshot


class SessionConflictError(RuntimeError):
    """A durable session changed since the caller loaded it."""


class DurableSessionStore(ABC):
    """Persists versioned session snapshots with optimistic concurrency."""

    @abstractmethod
    def load(self, session_id: str) -> SessionSnapshot | None:
        ...

    @abstractmethod
    def save(self, snapshot: SessionSnapshot, *, expected_revision: int) -> SessionSnapshot:
        ...


class InMemoryDurableSessionStore(DurableSessionStore):
    def __init__(self) -> None:
        self._snapshots: dict[str, SessionSnapshot] = {}

    def load(self, session_id: str) -> SessionSnapshot | None:
        return self._snapshots.get(session_id)

    def save(self, snapshot: SessionSnapshot, *, expected_revision: int) -> SessionSnapshot:
        current = self._snapshots.get(snapshot.session_id)
        revision = current.revision if current else 0
        if revision != expected_revision:
            raise SessionConflictError(f"Session {snapshot.session_id!r} revision conflict")
        stored = SessionSnapshot(**{**snapshot.model_dump(), "revision": revision + 1})
        self._snapshots[snapshot.session_id] = stored
        return stored
