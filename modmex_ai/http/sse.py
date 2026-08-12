from __future__ import annotations

import json
from collections.abc import AsyncIterable, Iterable
from typing import Any


def parse_sse_lines(lines: Iterable[bytes | str]):
    data: list[str] = []
    buffer = ""
    for chunk in lines:
        buffer += chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            text = line.strip()
            if not text:
                if data:
                    payload = "\n".join(data)
                    data = []
                    if payload == "[DONE]":
                        return
                    yield json.loads(payload)
                continue
            if text.startswith("data:"):
                data.append(text[5:].strip())
    if data:
        payload = "\n".join(data)
        if payload != "[DONE]":
            yield json.loads(payload)
    elif buffer.strip():
        # Streamable HTTP servers may legally return one JSON response when
        # no event stream is needed, even when the client requested events.
        yield json.loads(buffer.strip())


async def parse_sse_lines_async(lines: AsyncIterable[bytes | str]):
    """Asynchronously parse Server-Sent Events without buffering a response."""
    data: list[str] = []
    buffer = ""
    async for chunk in lines:
        buffer += chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            text = line.strip()
            if not text:
                if data:
                    payload = "\n".join(data)
                    data = []
                    if payload == "[DONE]":
                        return
                    yield json.loads(payload)
                continue
            if text.startswith("data:"):
                data.append(text[5:].strip())
    if data:
        payload = "\n".join(data)
        if payload != "[DONE]":
            yield json.loads(payload)
    elif buffer.strip():
        yield json.loads(buffer.strip())


def event_data(event: dict[str, Any]) -> Any:
    return event.get("data", event)
