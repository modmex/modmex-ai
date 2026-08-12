from __future__ import annotations

import asyncio

from modmex_ai.http.client import HttpResponse
from modmex_ai.mcp import MCPClient, MCPClientConnection, MCPClientManager
from modmex_ai.agents import Agent
from modmex_ai.models import FakeModel, ModelResponse, ToolCall


class FakeHttp:
    def __init__(self):
        self.calls = []
        self.responses = [
            HttpResponse(status_code=200, headers={}, body={
                "jsonrpc": "2.0", "id": 1, "result": {
                    "resultType": "complete",
                    "supportedVersions": ["2026-07-28"],
                    "capabilities": {"tools": {}},
                    "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "loads", "version": "1"}},
                },
            }),
            HttpResponse(status_code=200, headers={}, body={
                "jsonrpc": "2.0", "id": 2, "result": {"tools": [{
                    "name": "lookup", "description": "Lookup a load",
                    "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
                }]},
            }),
            HttpResponse(status_code=200, headers={}, body={
                "jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "{\"id\":\"A-1\"}"}], "isError": False},
            }),
        ]

    async def post_json(self, url, *, headers=None, data=None, timeout=None):
        self.calls.append((url, headers, data))
        return self.responses.pop(0)


def test_client_initializes_discovers_and_calls_remote_tool() -> None:
    async def run() -> None:
        http = FakeHttp()
        client = MCPClient("https://loads.example/mcp", http=http)
        initialized = await client.initialize()
        tools = await client.list_tools()
        result = await client.call_tool("lookup", {"id": "A-1"})

        assert initialized["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == "loads"
        assert len(tools) == 1
        assert tools[0].schema()["parameters"]["required"] == ["id"]
        assert result == {"id": "A-1"}
        assert http.calls[1][1]["MCP-Protocol-Version"] == "2026-07-28"
        assert http.calls[1][1]["Mcp-Method"] == "tools/list"

    asyncio.run(run())


def test_remote_tool_runs_inside_async_agent_with_sync_model() -> None:
    async def run() -> None:
        http = FakeHttp()
        client = MCPClient("https://loads.example/mcp", http=http)
        await client.initialize()
        tools = await client.list_tools()
        agent = Agent(
            name="lookup-agent",
            instructions="Use the remote lookup tool.",
            tools=tools,
            model=FakeModel([
                ModelResponse(tool_calls=[ToolCall(
                    tool_call_id="call-1",
                    name="lookup",
                    arguments={"id": "A-1"},
                )]),
                ModelResponse(output_text="done"),
            ]),
        )

        result = await agent.run_async("Look up A-1")

        assert result.output == "done"
        assert http.calls[2][2]["params"]["arguments"] == {"id": "A-1"}
        await client.close()

    asyncio.run(run())


def test_agent_loads_tools_from_mcp_client_connections() -> None:
    async def run() -> None:
        http = FakeHttp()
        connection = MCPClientConnection(MCPClient("https://loads.example/mcp", http=http))
        agent = Agent(
            name="lookup-agent",
            instructions="Use the remote lookup tool.",
            mcp_clients=[connection],
            model=FakeModel([
                ModelResponse(tool_calls=[ToolCall(
                    tool_call_id="call-1", name="lookup", arguments={"id": "A-1"},
                )]),
                ModelResponse(output_text="done"),
            ]),
        )

        result = await agent.run_async("Look up A-1")

        assert result.output == "done"
        assert connection.connected is True
        assert len(connection.tools) == 1
        assert http.calls[2][2]["params"]["arguments"] == {"id": "A-1"}
        await connection.close()

    asyncio.run(run())


def test_stream_events_accepts_missing_params() -> None:
    async def run() -> None:
        class StreamingFakeHttp(FakeHttp):
            def post_json_stream(self, url, *, headers=None, data=None, timeout=None):
                self.calls.append((url, headers, data))

                async def chunks():
                    yield 'data: {"resultType":"complete"}\n\n'

                return chunks()

        http = StreamingFakeHttp()
        client = MCPClient("https://loads.example/mcp", http=http)
        events = [event async for event in client.stream_events("tools/list")]
        assert events == [{"resultType": "complete"}]
        assert http.calls[0][2]["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"

    asyncio.run(run())


def test_client_follows_pagination_and_returns_structured_content() -> None:
    async def run() -> None:
        class PaginatedHttp:
            def __init__(self):
                self.calls = []
                self.responses = [
                    HttpResponse(status_code=200, headers={}, body={"jsonrpc": "2.0", "id": 1, "result": {
                        "resultType": "complete", "tools": [{"name": "one", "inputSchema": {}}], "nextCursor": "1",
                    }}),
                    HttpResponse(status_code=200, headers={}, body={"jsonrpc": "2.0", "id": 2, "result": {
                        "resultType": "complete", "tools": [{"name": "two", "inputSchema": {}}],
                    }}),
                    HttpResponse(status_code=200, headers={}, body={"jsonrpc": "2.0", "id": 3, "result": {
                        "resultType": "complete", "structuredContent": {"id": "A-1"}, "content": [], "isError": False,
                    }}),
                ]

            async def post_json(self, url, *, headers=None, data=None, timeout=None):
                self.calls.append(data)
                return self.responses.pop(0)

        http = PaginatedHttp()
        client = MCPClient("https://example.test/mcp", http=http)
        tools = await client.list_tools()
        result = await client.call_tool("lookup")

        assert [tool.name for tool in tools] == ["one", "two"]
        assert http.calls[1]["params"]["cursor"] == "1"
        assert result == {"id": "A-1"}

    asyncio.run(run())


def test_connection_accepts_injected_client_without_owning_its_lifecycle() -> None:
    http = FakeHttp()
    client = MCPClient("https://loads.example/mcp", http=http)
    connection = MCPClientConnection(client)

    async def run() -> None:
        await connection.connect()
        await connection.close()
        assert connection.connected is False
        assert http.responses  # the injected HTTP client was not closed by the connection

    asyncio.run(run())


def test_manager_connects_and_closes_multiple_connections() -> None:
    async def run() -> None:
        first = MCPClientConnection(MCPClient("https://one.example/mcp", http=FakeHttp()))
        second = MCPClientConnection(MCPClient("https://two.example/mcp", http=FakeHttp()))

        async with MCPClientManager([first, second]) as clients:
            assert clients == [first, second]
            assert all(connection.connected for connection in clients)

        assert not first.connected
        assert not second.connected

    asyncio.run(run())
