from __future__ import annotations

import json
from collections.abc import Iterable
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


def event_data(event: dict[str, Any]) -> Any:
    return event.get("data", event)
