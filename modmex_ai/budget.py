"""Cooperative deadlines shared by nested agent and HTTP executions."""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from contextvars import ContextVar

from modmex_ai.errors import RunTimeoutError

_deadline: ContextVar[float | None] = ContextVar("modmex_ai_deadline", default=None)


def remaining_seconds() -> float | None:
    deadline = _deadline.get()
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RunTimeoutError("Agent execution time budget exhausted")
    return remaining


def bounded_timeout(timeout: float) -> float:
    remaining = remaining_seconds()
    return min(timeout, remaining) if remaining is not None else timeout


@contextmanager
def execution_budget(seconds: float | None):
    parent = _deadline.get()
    deadline = time.monotonic() + seconds if seconds is not None else parent
    if parent is not None and deadline is not None:
        deadline = min(parent, deadline)
    token = _deadline.set(deadline)
    try:
        remaining_seconds()
        yield
        remaining_seconds()
    finally:
        _deadline.reset(token)


async def bounded_await(awaitable):
    try:
        remaining = remaining_seconds()
    except RunTimeoutError:
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise
    if remaining is None:
        return await awaitable
    # timeout() distinguishes the budget expiring from a tool's own TimeoutError.
    timer = asyncio.timeout(remaining)
    try:
        async with timer:
            return await awaitable
    except TimeoutError as error:
        if timer.expired():
            raise RunTimeoutError("Agent execution time budget exhausted") from error
        raise
