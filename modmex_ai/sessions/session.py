from __future__ import annotations

from abc import ABC, abstractmethod

from modmex_ai.sessions.item import SessionItem


class Session(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        ...

    @abstractmethod
    def get_items(self, *, limit: int | None = None) -> list[SessionItem]:
        ...

    @abstractmethod
    def add_items(self, items: list[SessionItem]) -> None:
        ...

    @abstractmethod
    def pop_item(self) -> SessionItem | None:
        ...

    @abstractmethod
    def clear_session(self) -> None:
        ...
