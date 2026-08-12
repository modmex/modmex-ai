from __future__ import annotations

import asyncio

from modmex_ai.http.sse import event_data, parse_sse_lines, parse_sse_lines_async


def test_sse_parser_handles_bytes_done_and_plain_json() -> None:
    assert list(parse_sse_lines([b'data: {"a":', b' 1}\n\n'])) == [{"a": 1}]
    assert list(parse_sse_lines(["data: [DONE]\n\n"])) == []
    assert list(parse_sse_lines(['{"plain": true}'])) == [{"plain": True}]
    assert event_data({"data": "value"}) == "value"
    assert event_data({"event": "value"}) == {"event": "value"}


def test_async_sse_parser_handles_split_chunks_and_done() -> None:
    async def chunks():
        yield b'data: {"a":'
        yield ' 1}\n\n'
        yield 'data: [DONE]\n\n'

    async def run():
        return [event async for event in parse_sse_lines_async(chunks())]

    assert asyncio.run(run()) == [{"a": 1}]
