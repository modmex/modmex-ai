"""Decode AWS binary event streams (``application/vnd.amazon.eventstream``).

ConverseStream does not use SSE: each event is a binary frame with headers
(``:message-type``, ``:event-type``) and a JSON payload. Framing and CRC checks are
delegated to botocore's parser.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from modmex_ai.errors import ProviderError, RateLimitError

# Stream exceptions arrive after HTTP 200; map them to the HTTP status they stand for.
_EXCEPTION_STATUS = {
    "throttlingException": 429,
    "serviceUnavailableException": 503,
    "internalServerException": 500,
    "validationException": 400,
    "modelStreamErrorException": 424,
    "modelTimeoutException": 408,
}


def decode_event_stream(chunks: Iterator[bytes]) -> Iterator[dict[str, Any]]:
    from botocore.eventstream import EventStreamBuffer

    buffer = EventStreamBuffer()
    for chunk in chunks:
        buffer.add_data(chunk)
        for message in buffer:
            yield _event(message)


async def decode_event_stream_async(chunks: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
    from botocore.eventstream import EventStreamBuffer

    buffer = EventStreamBuffer()
    async for chunk in chunks:
        buffer.add_data(chunk)
        for message in buffer:
            yield _event(message)


def _event(message: Any) -> dict[str, Any]:
    headers = message.headers
    payload = json.loads(message.payload.decode("utf-8")) if message.payload else {}
    kind = headers.get(":message-type")
    if kind == "event":
        return {"type": headers.get(":event-type"), **payload}
    name = headers.get(":exception-type") or headers.get(":error-code") or "error"
    detail = payload.get("message") or payload.get("Message") or headers.get(":error-message") or name
    status = _EXCEPTION_STATUS.get(name)
    error_cls = RateLimitError if status == 429 else ProviderError
    raise error_cls(f"Bedrock stream {name}: {detail}", status_code=status, response_body=payload)
