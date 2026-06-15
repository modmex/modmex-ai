from __future__ import annotations

from uuid import uuid4

from modmex_ai.sessions.item import SessionItem
from modmex_ai.sessions.session import Session


class InMemorySession(Session):
    def __init__(
        self,
        session_id: str | None = None,
        items: list[SessionItem] | None = None,
    ) -> None:
        self._id = session_id or str(uuid4())
        self._items = list(items or [])

    @property
    def id(self) -> str:
        return self._id

    def get_items(self, *, limit: int | None = None) -> list[SessionItem]:
        if limit is None:
            return list(self._items)
        if limit <= 0:
            return []
        return self._items[-limit:]

    def add_items(self, items: list[SessionItem]) -> None:
        self._items.extend(items)

    def pop_item(self) -> SessionItem | None:
        if not self._items:
            return None
        return self._items.pop()

    def clear_session(self) -> None:
        self._items.clear()
