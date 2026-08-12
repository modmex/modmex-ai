from __future__ import annotations

import asyncio
import pytest

from modmex_ai.http.client import HttpResponse
from modmex_ai.mcp import MCPClient, MCPClientConnection, MCPClientManager, MCPError, MCPInputRequired
from modmex_ai.mcp.client import _content_value, _content_text, _header
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


def test_client_encodes_names_and_extracts_parameter_headers() -> None:
    client = MCPClient("https://example.test/mcp", http=FakeHttp())
    headers = client._headers("tools/call", {
        "name": "tool/name",
        "arguments": {"region": "Hello, 世界"},
        "_schema": {"type": "object", "properties": {"region": {"type": "string", "x-mcp-header": "Region"}}},
    })
    assert headers["Mcp-Name"].startswith("=?base64?")
    assert headers["Mcp-Param-Region"].startswith("=?base64?")


def test_client_raises_input_required_for_resources_and_prompts() -> None:
    async def run() -> None:
        class InputRequiredHttp:
            def __init__(self):
                self.responses = [
                    HttpResponse(status_code=200, headers={}, body={"jsonrpc": "2.0", "id": 1, "result": {"resultType": "input_required", "inputRequests": [{"id": "x"}]} }),
                    HttpResponse(status_code=200, headers={}, body={"jsonrpc": "2.0", "id": 2, "result": {"resultType": "input_required", "inputRequests": [{"id": "y"}]} }),
                ]

            async def post_json(self, url, *, headers=None, data=None, timeout=None):
                return self.responses.pop(0)

        client = MCPClient("https://example.test/mcp", http=InputRequiredHttp())
        with pytest.raises(MCPInputRequired):
            await client.read_resource("x://resource")
        with pytest.raises(MCPInputRequired):
            await client.get_prompt("prompt")

    asyncio.run(run())


def test_client_correlates_jsonrpc_response_id() -> None:
    async def run() -> None:
        class WrongIdHttp:
            async def post_json(self, url, *, headers=None, data=None, timeout=None):
                return HttpResponse(status_code=200, headers={}, body={"jsonrpc": "2.0", "id": 999, "result": {"resultType": "complete"}})

        client = MCPClient("https://example.test/mcp", http=WrongIdHttp())
        with pytest.raises(MCPError, match="id"):
            await client.discover()

    asyncio.run(run())


def test_client_follows_resource_and_prompt_pagination() -> None:
    async def run() -> None:
        class Pages:
            def __init__(self):
                self.responses = [
                    {"resources": [{"uri": "a://one"}], "nextCursor": "next"},
                    {"resources": [{"uri": "a://two"}]},
                    {"prompts": [{"name": "one"}], "nextCursor": "p2"},
                    {"prompts": [{"name": "two"}]},
                ]

            async def post_json(self, url, *, headers=None, data=None, timeout=None):
                result = {"resultType": "complete", **self.responses.pop(0)}
                return HttpResponse(status_code=200, headers={}, body={"jsonrpc": "2.0", "id": data["id"], "result": result})

        client = MCPClient("https://example.test/mcp", http=Pages())
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        assert [item["uri"] for item in resources["resources"]] == ["a://one", "a://two"]
        assert [item["name"] for item in prompts["prompts"]] == ["one", "two"]

    asyncio.run(run())


def test_client_accepts_sse_response_after_notification() -> None:
    async def run() -> None:
        class SseHttp:
            async def post_json(self, url, *, headers=None, data=None, timeout=None, raise_for_status=True):
                body = 'data: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n'
                body += 'data: {"jsonrpc":"2.0","id":1,"result":{"resultType":"complete","tools":[]}}\n\n'
                return HttpResponse(status_code=200, headers={"content-type": "text/event-stream"}, body=body)

        result = await MCPClient("https://example.test/mcp", http=SseHttp()).list_tools()
        assert result == []

    asyncio.run(run())


def test_client_preserves_jsonrpc_error_over_http_error_status() -> None:
    async def run() -> None:
        class ErrorHttp:
            async def post_json(self, url, *, headers=None, data=None, timeout=None, raise_for_status=True):
                return HttpResponse(status_code=400, headers={}, body={"jsonrpc": "2.0", "id": data["id"], "error": {"code": -32020, "message": "header mismatch", "data": {"field": "Mcp-Name"}}})

        with pytest.raises(MCPError) as captured:
            await MCPClient("https://example.test/mcp", http=ErrorHttp()).discover()
        assert captured.value.code == -32020
        assert captured.value.http_status == 400
        assert captured.value.data == {"field": "Mcp-Name"}

    asyncio.run(run())


def test_client_rejects_unknown_result_type() -> None:
    async def run() -> None:
        class UnknownResultHttp:
            async def post_json(self, url, *, headers=None, data=None, timeout=None, raise_for_status=True):
                return HttpResponse(status_code=200, headers={}, body={"jsonrpc": "2.0", "id": data["id"], "result": {"resultType": "future"}})

        with pytest.raises(MCPError, match="resultType"):
            await MCPClient("https://example.test/mcp", http=UnknownResultHttp()).read_resource("a://one")

    asyncio.run(run())


@pytest.mark.parametrize("body", [
    None,
    {"jsonrpc": "1.0", "id": 1, "result": {}},
    {"jsonrpc": "2.0", "id": 1},
    {"jsonrpc": "2.0", "id": 1, "result": {}, "error": {}},
])
def test_client_rejects_malformed_jsonrpc_envelopes(body) -> None:
    async def run() -> None:
        class MalformedHttp:
            async def post_json(self, url, *, headers=None, data=None, timeout=None, raise_for_status=True):
                return HttpResponse(status_code=200, headers={}, body=body)

        with pytest.raises(MCPError):
            await MCPClient("https://example.test/mcp", http=MalformedHttp()).discover()

    asyncio.run(run())


def test_client_filters_invalid_tool_definitions_and_handles_tool_errors() -> None:
    async def run() -> None:
        class ToolsHttp:
            def __init__(self):
                self.calls = 0

            async def post_json(self, url, *, headers=None, data=None, timeout=None, raise_for_status=True):
                self.calls += 1
                if data["method"] == "tools/list":
                    result = {"resultType": "complete", "tools": [
                        {"name": "valid", "inputSchema": {"type": "object", "properties": {}}},
                        {"name": "invalid", "inputSchema": {"type": "object", "properties": {"x": {"type": "number", "x-mcp-header": "X"}}}},
                        "not-a-tool",
                    ]}
                else:
                    result = {"resultType": "complete", "isError": True, "content": [{"type": "text", "text": "permission denied"}]}
                return HttpResponse(status_code=200, headers={}, body={"jsonrpc": "2.0", "id": data["id"], "result": result})

        client = MCPClient("https://example.test/mcp", http=ToolsHttp())
        assert [tool.name for tool in await client.list_tools()] == ["valid"]
        with pytest.raises(MCPError, match="permission denied"):
            await client.call_tool("valid")

    asyncio.run(run())


def test_client_handles_empty_content_and_explicit_pages() -> None:
    async def run() -> None:
        class PagesHttp:
            async def post_json(self, url, *, headers=None, data=None, timeout=None, raise_for_status=True):
                method = data["method"]
                if method == "resources/list":
                    result = {"resultType": "complete", "resources": []}
                elif method == "resources/templates/list":
                    result = {"resultType": "complete", "resourceTemplates": [{"uriTemplate": "x://{id}"}]}
                elif method == "prompts/list":
                    result = {"resultType": "complete", "prompts": []}
                elif method == "resources/read":
                    result = {"resultType": "complete", "contents": []}
                else:
                    result = {"resultType": "complete", "messages": []}
                return HttpResponse(status_code=200, headers={}, body={"jsonrpc": "2.0", "id": data["id"], "result": result})

        client = MCPClient("https://example.test/mcp", http=PagesHttp())
        assert await client.list_resources_page(page_size=10) == {"resultType": "complete", "resources": []}
        assert await client.list_resource_templates_page(cursor="abc") == {"resultType": "complete", "resourceTemplates": [{"uriTemplate": "x://{id}"}]}
        assert await client.list_prompts_page(page_size=5) == {"resultType": "complete", "prompts": []}
        assert await client.read_resource("x://empty") == []
        assert await client.get_prompt("empty") == {"resultType": "complete", "messages": []}

    asyncio.run(run())


def test_client_rejects_sse_without_correlated_response() -> None:
    async def run() -> None:
        class SseHttp:
            async def post_json(self, url, *, headers=None, data=None, timeout=None, raise_for_status=True):
                return HttpResponse(status_code=200, headers={"content-type": "text/event-stream"}, body='data: {"jsonrpc":"2.0","id":99,"result":{}}\n\n')

        with pytest.raises(MCPError, match="requested JSON-RPC id"):
            await MCPClient("https://example.test/mcp", http=SseHttp()).discover()

    asyncio.run(run())


def test_client_streams_tool_progress_and_final_result() -> None:
    async def run() -> None:
        class ProgressHttp:
            def post_json_stream(self, url, *, headers=None, data=None, timeout=None):
                async def chunks():
                    yield 'event: progress\ndata: {"progress":1,"total":2}\n\n'
                    yield 'data: {"jsonrpc":"2.0","id":1,"result":{"resultType":"complete","structuredContent":{"status":"done"}}}\n\n'
                return chunks()

        client = MCPClient("https://example.test/mcp", http=ProgressHttp())
        events = [event async for event in client.stream_tool_call("long_running")]
        assert events == [
            {"kind": "progress", "data": {"progress": 1, "total": 2}},
            {"kind": "result", "result": {"resultType": "complete", "structuredContent": {"status": "done"}}},
        ]

    asyncio.run(run())


def test_client_stream_tool_call_rejects_unmatched_final_id() -> None:
    async def run() -> None:
        class ProgressHttp:
            def post_json_stream(self, url, *, headers=None, data=None, timeout=None):
                async def chunks():
                    yield 'data: {"jsonrpc":"2.0","id":99,"result":{"resultType":"complete"}}\n\n'
                return chunks()

        client = MCPClient("https://example.test/mcp", http=ProgressHttp())
        with pytest.raises(MCPError, match="id"):
            _ = [event async for event in client.stream_tool_call("long_running")]

    asyncio.run(run())


def test_client_helpers_handle_header_and_non_json_content() -> None:
    assert _header({"X-Test": "ok"}, "x-test") == "ok"
    assert _header({}, "missing") is None
    assert _content_text([{"type": "image", "data": "x"}]).startswith("[")
    assert _content_value([{"type": "text", "text": "plain text"}]) == "plain text"
    assert _content_value([{"type": "image", "data": "x"}])


def test_client_closes_owned_http_and_supports_explicit_tool_page() -> None:
    async def run() -> None:
        class OwnedHttp(FakeHttp):
            def __init__(self):
                super().__init__()
                self.closed = False

            async def close(self):
                self.closed = True

            async def post_json(self, url, *, headers=None, data=None, timeout=None, raise_for_status=True):
                response = await super().post_json(url, headers=headers, data=data, timeout=timeout)
                return response

        http = OwnedHttp()
        client = MCPClient("https://example.test/mcp", http=http)
        client._owns_http = True
        page = await client.list_tools_page(page_size=3)
        assert page["supportedVersions"] == ["2026-07-28"]
        await client.close()
        assert http.closed

    asyncio.run(run())
