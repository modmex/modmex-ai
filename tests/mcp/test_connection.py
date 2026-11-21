from __future__ import annotations

import asyncio

import pytest

from modmex_ai.agents import Agent
from modmex_ai.http.client import HttpResponse
from modmex_ai.mcp import MCPClient, MCPClientConnection
from modmex_ai.models import FakeModel, ModelResponse


class ConnectionHttp:
    def __init__(self):
        self.calls = 0

    async def post_json(self, url, *, headers=None, data=None, timeout=None):
        self.calls += 1
        method = data["method"]
        if method == "server/discover":
            result = {"resultType": "complete", "capabilities": {}}
        else:
            result = {"resultType": "complete", "tools": [{"name": "lookup", "inputSchema": {}}]}
        return HttpResponse(status_code=200, headers={}, body={"jsonrpc": "2.0", "id": data["id"], "result": result})


def test_connection_reconnects_after_cache_invalidation_and_refreshes() -> None:
    async def run() -> None:
        http = ConnectionHttp()
        connection = MCPClientConnection(MCPClient("https://example.test/mcp", http=http))
        with pytest.raises(RuntimeError):
            connection.tools
        await connection.connect()
        first_call_count = http.calls
        assert await connection.connect() is connection
        assert http.calls == first_call_count
        await connection.refresh_tools()
        assert http.calls == first_call_count + 1
        connection.invalidate_tools_cache()
        assert connection.connected is False
        await connection.connect()
        assert connection.connected is True
        await connection.close()

    asyncio.run(run())


def test_agent_rejects_duplicate_remote_tool_names() -> None:
    async def run() -> None:
        first = MCPClientConnection(MCPClient("https://one.test/mcp", http=ConnectionHttp()))
        second = MCPClientConnection(MCPClient("https://two.test/mcp", http=ConnectionHttp()))
        agent = Agent(name="agent", instructions="", mcp_clients=[first, second], model=FakeModel([ModelResponse(output_text="done")]))
        with pytest.raises(ValueError, match="Duplicate tool name"):
            await agent.run_async("go")
        await first.close()
        await second.close()

    asyncio.run(run())


def test_connection_constructor_rejects_ambiguous_sources_and_context_manager() -> None:
    client = MCPClient("https://example.test/mcp", http=ConnectionHttp())
    with pytest.raises(ValueError):
        MCPClientConnection(client, client=client)
    with pytest.raises(ValueError):
        MCPClientConnection(client, headers={"Authorization": "x"})
    with pytest.raises(TypeError):
        MCPClientConnection()

    async def run() -> None:
        connection = MCPClientConnection(client, discover=False)
        async with connection as active:
            assert active is connection
            assert connection.connected
        assert not connection.connected

    asyncio.run(run())


def test_manager_closes_connected_connections_when_a_later_connection_fails() -> None:
    async def run() -> None:
        first = MCPClientConnection(MCPClient("https://one.test/mcp", http=ConnectionHttp()))

        class FailingHttp(ConnectionHttp):
            async def post_json(self, *args, **kwargs):
                raise RuntimeError("server unavailable")

        second = MCPClientConnection(MCPClient("https://two.test/mcp", http=FailingHttp()))
        from modmex_ai.mcp import MCPClientManager
        manager = MCPClientManager([first, second])
        with pytest.raises(RuntimeError, match="unavailable"):
            await manager.connect()
        assert not first.connected
        assert not second.connected

    asyncio.run(run())


@pytest.mark.parametrize("use_manager", [False, True])
def test_discovery_overlaps_between_servers(use_manager):
    async def run():
        from modmex_ai.mcp import MCPClientManager
        started = set()
        ready = asyncio.Event()

        class ConcurrentHttp(ConnectionHttp):
            async def post_json(self, url, *, headers=None, data=None, timeout=None):
                kwargs = {"headers": headers, "data": data, "timeout": timeout}
                if kwargs["data"]["method"] == "server/discover":
                    started.add(url)
                    if len(started) == 2:
                        ready.set()
                    await asyncio.wait_for(ready.wait(), 1)
                response = await super().post_json(url, **kwargs)
                if kwargs["data"]["method"] == "tools/list":
                    response.body["result"]["tools"][0]["name"] = url
                return response

        connections = [MCPClientConnection(MCPClient(url, http=ConcurrentHttp())) for url in ("one", "two")]
        if use_manager:
            async with MCPClientManager(connections):
                assert all(connection.connected for connection in connections)
        else:
            agent = Agent(name="test", instructions="", mcp_clients=connections,
                          model=FakeModel([ModelResponse(output_text="done")]))
            await agent.run_async("go")
            assert len(agent.tools) == 2

    asyncio.run(run())


def test_client_validates_ids_for_overlapping_requests():
    async def run():
        ready = asyncio.Event()
        requests = []

        class ConcurrentHttp:
            async def post_json(self, url, *, headers, data, **kwargs):
                requests.append(data)
                if len(requests) == 2:
                    ready.set()
                await ready.wait()
                if data["id"] == 1:
                    await asyncio.sleep(0.01)
                return HttpResponse(status_code=200, headers={}, body={
                    "jsonrpc": "2.0", "id": data["id"],
                    "result": {"structuredContent": data["params"]["name"]},
                })

        client = MCPClient("test", http=ConcurrentHttp())
        assert await asyncio.gather(client.call_tool("one"), client.call_tool("two")) == ["one", "two"]
        assert [request["id"] for request in requests] == [1, 2]

    asyncio.run(run())


def test_owned_client_reconnects_after_close_on_a_new_event_loop():
    from modmex_ai.http.async_client import AsyncHttpClient
    from modmex_ai.mcp import MCPClientManager

    connection = MCPClientConnection("https://example.test/mcp")
    first_transport = connection.client.http

    async def invocation():
        connection.client.http.post_json = ConnectionHttp().post_json  # stub the wire
        async with MCPClientManager([connection]):
            return [tool.name for tool in connection.tools]

    assert asyncio.run(invocation()) == ["lookup"]
    assert first_transport.is_closed
    assert isinstance(connection.client.http, AsyncHttpClient) and not connection.client.http.is_closed
    assert asyncio.run(invocation()) == ["lookup"]  # second warm invocation


def test_agent_drops_mcp_tools_the_server_no_longer_lists():
    class CatalogHttp(ConnectionHttp):
        def __init__(self):
            super().__init__()
            self.names = ["lookup", "delete_all"]

        async def post_json(self, url, *, headers=None, data=None, timeout=None):
            if data["method"] == "tools/list":
                tools = [{"name": name, "inputSchema": {}} for name in self.names]
                return HttpResponse(status_code=200, headers={}, body={"jsonrpc": "2.0", "id": data["id"], "result": {"resultType": "complete", "tools": tools}})
            return await super().post_json(url, headers=headers, data=data, timeout=timeout)

    async def run():
        http = CatalogHttp()
        connection = MCPClientConnection(MCPClient("https://example.test/mcp", http=http))
        agent = Agent(name="a", instructions="", mcp_clients=[connection],
                      model=FakeModel([ModelResponse(output_text="one"), ModelResponse(output_text="two")]))
        await agent.run_async("go")
        assert sorted(tool.name for tool in agent.tools) == ["delete_all", "lookup"]

        http.names = ["lookup"]
        await connection.refresh_tools()
        await agent.run_async("again")
        assert [tool.name for tool in agent.tools] == ["lookup"]

    asyncio.run(run())
