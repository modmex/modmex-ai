from __future__ import annotations

from typing import Protocol


class RealtimeTransport(Protocol):
    """Minimal async transport required by a realtime session."""

    async def send(self, message: str) -> None:
        ...

    async def recv(self) -> str | bytes:
        ...

    async def close(self) -> None:
        ...
